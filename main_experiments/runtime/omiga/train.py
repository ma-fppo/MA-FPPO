"""Fresh continuous MAC-Flow pretraining followed by centralized-critic PPO."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
from torch import nn
from common import read, write, save, sha, seed_all, grad_norm, finite, memory_available
from data import Dataset
from model import ContinuousMACFlow, GaussianStudent, mlp, gae
from mamujoco_env import Pool

PAUSE = False
def pause(signum, frame):
    global PAUSE
    PAUSE = True


class Run:
    def __init__(self, cfg_path, phase):
        self.cfg_path = str(cfg_path); self.cfg = read(cfg_path); self.phase = phase
        self.out = Path(self.cfg['output']) / phase
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out / 'checkpoint_0.pt').exists(): raise FileExistsError('Refusing an implicit restart: ' + str(self.out))
        self.log = (self.out / 'metrics.jsonl').open('a', buffering=1)
        self.started = time.monotonic(); self.step = 0
        signal.signal(signal.SIGTERM, pause); signal.signal(signal.SIGINT, pause)
        self.status('initializing')

    def emit(self, kind, **values):
        record = dict(kind=kind, phase=self.phase, step=self.step, time=time.time(), elapsed_s=time.monotonic()-self.started, **values)
        assert finite(record), record
        line = json.dumps(record, allow_nan=False); self.log.write(line + '\n'); print(line, flush=True)

    def status(self, status, **values):
        write(self.out / 'status.json', dict(status=status, phase=self.phase, step=self.step, pid=os.getpid(), updated=time.time(), **values))

    def evaluation(self, ck, episodes=20, seed=10000, label='eval'):
        if self.cfg['smoke']: episodes = 4
        output = self.out / ('%s_%d.json' % (label, self.step))
        subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate.py')), '--config', self.cfg_path,
            '--checkpoint', str(ck), '--output', str(output), '--episodes', str(episodes), '--seed', str(seed)], check=True)
        result = read(output); self.emit(label, policies=result['policies'], checkpoint_sha256=result['checkpoint_sha256'])
        return result

    def guard(self, state):
        if PAUSE or memory_available() < float(os.environ.get('MA_FPPO_MIN_FREE_RAM_GIB', '2')):
            save(self.out / 'checkpoint_paused.pt', state); self.status('paused', reason='signal or host memory below the configured reserve')
            raise SystemExit(75)


def pretrain(run):
    c = run.cfg; seed_all(c['seed']); device = 'cuda'
    dataset = Dataset(c['data_root'], c['task'], c['split'], c['seed'])
    assert dataset.obs_dim == c['obs_dim']
    model = ContinuousMACFlow(c['obs_dim'],action_dim=c['action_dim']).to(device)
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=c['pretrain_lr'])
    def state():
        return dict(phase='pretrain', step=run.step, model=model.state_dict(), optimizer=optim.state_dict(), config=c,
                    data_rng=dataset.rng.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
    ck = run.out / 'checkpoint_0.pt'; save(ck, state()); run.evaluation(ck)
    run.status('running')
    for step in range(1, c['pretrain_steps']+1):
        metrics = model.learn(dataset.sample(device, c['batch_size'], c['sequence_length']), optim,
                              c['pretrain_discount'], c['alpha'], c['tau'])
        run.step = step
        if step == 1 or step % c['log_every'] == 0:
            run.emit('train', **metrics, updates_per_s=step/(time.monotonic()-run.started)); run.status('running'); run.guard(state())
        if step % c['pretrain_eval_every'] == 0 or step == c['pretrain_steps']:
            ck = run.out / ('checkpoint_%d.pt' % step); save(ck, state()); run.evaluation(ck)
    run.status('complete', checkpoint=str(ck), checkpoint_sha256=sha(ck)); run.emit('complete', checkpoint=str(ck))


def ppo(run):
    c = run.cfg; seed_all(c['seed']); device = 'cuda'; pool = None
    source = Path(c['output']) / 'pretrain' / ('checkpoint_%d.pt' % c['pretrain_steps'])
    source_sha = sha(source); ck = torch.load(source, map_location=device)
    assert ck['phase'] == 'pretrain' and ck['step'] == c['pretrain_steps']
    model = ContinuousMACFlow(c['obs_dim'],action_dim=c['action_dim']).to(device); model.load_state_dict(ck['model'])
    actor = GaussianStudent(model.student, c['initial_std'],action_dim=c['action_dim']).to(device)
    reference = copy.deepcopy(actor).eval().requires_grad_(False)
    critic = mlp(c['state_dim'], 1, (256, 256), norm=True).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=c['actor_lr'])
    critic_opt = torch.optim.Adam(critic.parameters(), lr=c['critic_lr'])
    assert not actor_opt.state and not critic_opt.state
    del model, ck
    initial = torch.cat([p.detach().flatten() for p in actor.parameters()]).clone()
    reference_initial = {k: v.clone() for k, v in reference.state_dict().items()}
    iteration = 0; updates = 0
    def state():
        return dict(phase='ppo', step=run.step, iteration=iteration, actor_updates=updates,
                    actor=actor.state_dict(), critic=critic.state_dict(), reference=reference.state_dict(),
                    actor_optimizer=actor_opt.state_dict(), critic_optimizer=critic_opt.state_dict(), config=c,
                    source_checkpoint=str(source), source_sha256=source_sha,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), numpy_rng=np.random.get_state())
    def value(state): return critic(state).squeeze(-1)
    def full_kl(obs, means, stds):
        with torch.no_grad():
            sums = 0.
            for start in range(0, len(obs), 1024):
                idx = slice(start, start+1024)
                d = actor(obs[idx]); old = torch.distributions.Normal(means[idx], stds[idx])
                sums += float(torch.distributions.kl_divergence(old, d).sum(-1).sum())
            return sums / (len(obs)*c['n_agents'])
    try:
        initial_ck = run.out / 'checkpoint_0.pt'; save(initial_ck, state())
        initial_eval = run.evaluation(initial_ck)
        offline_eval = read(source.parent / ('eval_%d.json' % c['pretrain_steps']))
        assert initial_eval['policies'] == offline_eval['policies'], 'Pretraining/PPO zero boundary mismatch'
        run.evaluation(initial_ck, c.get('endpoint_eval_episodes', 20), 90000, 'test')
        run.emit('fresh_initialization', source_sha256=source_sha, fresh_optimizers=True, zero_boundary_exact=True)
        pool = Pool(c['task'], c['data_root'], c['n_envs'], c['workers'], c['seed']+1000)
        episode_returns = np.zeros(c['n_envs']); run.status('running')
        while run.step < c['ppo_steps']:
            iteration += 1
            remaining = c['ppo_steps'] - run.step
            active_envs = min(c['n_envs'], remaining)
            horizon = min(c['warmup_steps'] if iteration == 1 else c['rollout_steps'], remaining // active_envs)
            assert horizon > 0
            records = []; episode_results = []
            for _ in range(horizon):
                with torch.no_grad():
                    obs = torch.tensor(pool.current[:active_envs], device=device); state_tensor = torch.tensor(pool.states[:active_envs], device=device); dist = actor(obs); action = dist.sample()
                    logp = dist.log_prob(action).sum(-1); val = value(state_tensor)
                    nxt, next_state, final_state, rewards, done, terminated = pool.step(action.clamp(-1, 1).cpu().numpy())
                    next_val = value(torch.tensor(final_state, device=device)) * (1-torch.tensor(terminated, device=device))
                    team_reward = rewards.mean(-1); active_returns = episode_returns[:active_envs]; active_returns += team_reward
                    mask = done.astype(bool); episode_results.extend(active_returns[mask].tolist()); active_returns[mask] = 0
                    records.append((obs, state_tensor, action, logp, val, next_val, torch.tensor(team_reward, device=device),
                        torch.tensor(done, device=device), dist.mean, dist.stddev))
                    run.step += active_envs
            obs, states, actions, oldlog, vals, nextvals, rewards, dones, means, stds = [torch.stack(x) for x in zip(*records)]
            adv, returns = gae(rewards, vals, nextvals, dones, c['gamma'], c['gae_lambda'])
            adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
            obs, actions, oldlog, means, stds = [x.flatten(0, 1) for x in (obs, actions, oldlog, means, stds)]
            adv = adv.flatten(); returns = returns.flatten(); states = states.flatten(0, 1)
            with torch.no_grad():
                replay_error = max(float((actor(obs[i:i+1024]).log_prob(actions[i:i+1024]).sum(-1)-oldlog[i:i+1024]).abs().max()) for i in range(0, len(obs), 1024))
            assert replay_error < 2e-3, replay_error
            update_actor = iteration > 1; stop_actor = False; metrics = []; checked_kl = 0.
            for epoch in range(c['epochs']):
                order = torch.randperm(len(obs), device=device)
                for indices in order.split(c['ppo_batch_size']):
                    o = obs[indices]; a = actions[indices]; ret = returns[indices]
                    critic_loss = .5 * (value(states[indices])-ret).square().mean()
                    critic_opt.zero_grad(set_to_none=True); critic_loss.backward()
                    critic_grad = float(nn.utils.clip_grad_norm_(critic.parameters(), c['max_grad_norm'])); critic_opt.step()
                    metric = dict(critic_loss=float(critic_loss.detach()), critic_grad=critic_grad)
                    if update_actor and not stop_actor:
                        dist = actor(o); logp = dist.log_prob(a).sum(-1)
                        ratio = (logp-oldlog[indices]).exp(); advantage = adv[indices, None]
                        policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(1-c['clip'], 1+c['clip']) * advantage).mean()
                        with torch.no_grad(): ref = reference(o)
                        anchor = torch.distributions.kl_divergence(dist, ref).sum(-1).mean()
                        entropy = dist.entropy().sum(-1).mean()
                        loss = policy_loss + c['anchor_coef']*anchor - c['entropy_coef']*entropy
                        actor_opt.zero_grad(set_to_none=True); loss.backward()
                        actor_grad = float(nn.utils.clip_grad_norm_(actor.parameters(), c['max_grad_norm']))
                        if not math.isfinite(actor_grad): raise FloatingPointError('actor gradient')
                        actor_opt.step()
                        with torch.no_grad(): actor.log_std.clamp_(-4., 0.)
                        updates += 1
                        metric.update(actor_loss=float(loss.detach()), actor_grad=actor_grad,
                            reference_kl=float(anchor.detach()), entropy=float(entropy.detach()),
                            clip_fraction=float(((ratio-1).abs()>c['clip']).float().mean()))
                        with torch.no_grad():
                            local_kl = float(torch.distributions.kl_divergence(torch.distributions.Normal(means[indices], stds[indices]), actor(o)).sum(-1).mean())
                        if local_kl > c['target_kl']:
                            checked_kl = full_kl(obs, means, stds)
                            stop_actor = checked_kl > c['target_kl']
                    metrics.append(metric)
                if update_actor and not stop_actor:
                    checked_kl = full_kl(obs, means, stds); stop_actor = checked_kl > c['target_kl']
            summary = {k: float(np.mean([m[k] for m in metrics if k in m])) for k in set().union(*(m.keys() for m in metrics))}
            with torch.no_grad(): delta = float((torch.cat([p.flatten() for p in actor.parameters()])-initial).norm())
            run.emit('train', iteration=iteration, active_envs=active_envs, rollout_env_steps=horizon*active_envs, actor_updates=updates, actor_parameter_delta=delta,
                replay_error=replay_error, old_new_kl=full_kl(obs, means, stds), early_stop=stop_actor,
                sampling_std=actor.log_std.exp().detach().cpu().tolist(), env_steps_per_s=run.step/(time.monotonic()-run.started),
                rollout_return=float(np.mean(episode_results)) if episode_results else None, **summary)
            assert all(torch.equal(v, reference.state_dict()[k]) for k, v in reference_initial.items())
            run.status('running'); run.guard(state())
            if run.step == c['ppo_steps'] or run.step // c['ppo_eval_every'] != (run.step-horizon*active_envs) // c['ppo_eval_every']:
                ck = run.out / ('checkpoint_%d.pt' % run.step); save(ck, state()); run.evaluation(ck)
        assert run.step == c['ppo_steps'] and updates > 0 and delta > 0 and sha(source) == source_sha
        restored = torch.load(ck, map_location=device); old = actor(obs[:128]).mean.detach().clone(); actor.load_state_dict(restored['actor'])
        assert torch.equal(old, actor(obs[:128]).mean.detach())
        run.evaluation(ck, c.get('endpoint_eval_episodes', 20), 90000, 'test')
        run.emit('complete', actor_updates=updates, actor_parameter_delta=delta, reference_frozen=True, roundtrip_exact=True, source_unchanged=True)
        run.status('complete', checkpoint=str(ck), checkpoint_sha256=sha(ck))
    finally:
        if pool: pool.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--phase', choices=['pretrain', 'ppo'], required=True)
    args = p.parse_args(); run = Run(args.config, args.phase)
    try: globals()[args.phase](run)
    except Exception as exc: run.status('failed', error=repr(exc)); raise
