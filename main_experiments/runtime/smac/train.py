#!/usr/bin/env python
"""PPO fine-tuning of the four existing SMACv2 action-policy checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np

from common import (atomic_json, compute_gae, finite_metrics, load_environment_factory,
                    normalize_advantages, observe, seed_environment, sha256, stack_records, transition_flags)


DEFAULTS = dict(seed=0, device='cuda', n_envs=4, rollout_steps=128, iterations=300,
    update_epochs=4, batch_size=64, actor_lr=2e-5, critic_lr=3e-4, gamma=.99,
    gae_lambda=.95, clip_coef=.05, target_kl=.02, max_grad_norm=.5,
    entropy_coef=.001, anchor_coef=.01, critic_warmup_iters=5,
    flow_steps=8, flow_std=.1, temperature=1., torch_history='recurrent',
    eval_episodes=20, eval_every=25, save_every=25, eval_seed=10000)


def evaluate(backend, make_env, cfg, episodes, deterministic):
    saved_carry = backend.carry
    saved_rng = (random.getstate(), np.random.get_state(), backend.rng_state())
    results = []
    try:
        for ep in range(episodes):
            seed = cfg['eval_seed']+ep
            random.seed(seed); np.random.seed(seed); backend.seed(seed)
            env = make_env(cfg['task'], seed=seed)
            try:
                seed_environment(env, seed)
                env.reset(); backend.reset()
                reward_sum, length, done = 0., 0, False
                while not done:
                    obs, _, legal = observe(env)
                    actions, _ = backend.act(obs, legal, deterministic=deterministic)
                    if not legal[np.arange(len(actions)), actions].all():
                        raise AssertionError('Illegal action in evaluation')
                    reward, done, info = env.step(actions.tolist())
                    reward_sum += float(reward); length += 1
                results.append((reward_sum, float(info.get('battle_won', False)), length))
            finally:
                env.close()
    finally:
        backend.carry = saved_carry
        random.setstate(saved_rng[0]); np.random.set_state(saved_rng[1]); backend.restore_rng(saved_rng[2])
    arr = np.asarray(results)
    return dict(avg_return=float(arr[:, 0].mean()), win_rate=float(arr[:, 1].mean()),
                avg_length=float(arr[:, 2].mean()), episodes=episodes,
                policy='deployment' if deterministic else 'ppo_sampling')


def run(cfg, resume=None, eval_only=False):
    for key in ['n_envs', 'rollout_steps', 'iterations', 'update_epochs', 'batch_size', 'eval_episodes', 'eval_every', 'save_every']:
        if int(cfg[key]) <= 0:
            raise ValueError(key + ' must be positive')
    if cfg['temperature'] <= 0:
        raise ValueError('PPO categorical temperature must be positive')
    out = Path(cfg['output_dir'])
    if out.exists() and any(out.iterdir()) and not (resume or eval_only):
        raise FileExistsError('Output directory is nonempty; choose a fresh directory or --resume')
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('SC2PATH', str(Path.home() / 'StarCraftII'))
    os.environ.setdefault('WANDB_MODE', 'offline')
    random.seed(cfg['seed']); np.random.seed(cfg['seed'])
    make_env = load_environment_factory(cfg['project_root'])
    envs = []
    env_steps, start_iteration, completed_episodes = 0, 0, 0
    log = open(out / ('evaluation.jsonl' if eval_only else 'metrics.jsonl'), 'a', buffering=1)
    def emit(kind, **fields):
        record = dict(kind=kind, variant=cfg['variant'], task=cfg['task'], **fields)
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        print(kind.upper() + ' ' + line, flush=True); log.write(line + '\n')
    try:
        for i in range(cfg['n_envs'] if not eval_only else 1):
            env = make_env(cfg['task'], seed=cfg['seed']+i)
            envs.append(env); env.reset()
        info = envs[0].get_env_info()
        if cfg['variant'] == 'jax':
            from jax_backend import JaxBackend
            backend = JaxBackend(cfg, info)
        else:
            from torch_backend import TorchBackend
            backend = TorchBackend(cfg, info)
        cfg['source_checkpoint_sha256'] = sha256(cfg['checkpoint'])
        if resume:
            start_iteration, env_steps = backend.restore(resume)
        checkpoint_hash = cfg['source_checkpoint_sha256']
        atomic_json(out / 'run_config.json', dict(cfg, source_checkpoint_sha256=checkpoint_hash,
                    resumed_from=resume, env_info=info, evaluation_only=eval_only))
        for deterministic in [True, False]:
            emit('baseline' if not resume else 'restored_eval',
                 **evaluate(backend, make_env, cfg, cfg['eval_episodes'], deterministic))
        if eval_only:
            return
        if start_iteration >= cfg['iterations']:
            raise ValueError('--iterations must exceed the resumed iteration; use --eval-only for evaluation')
        initial_actor = backend.actor_vector() if cfg.get('smoke') else None
        total_actor_updates = 0
        carries = [None] * len(envs)
        episode_returns = np.zeros(len(envs)); episode_lengths = np.zeros(len(envs), int)
        started = time.monotonic()
        for iteration in range(start_iteration, cfg['iterations']):
            records, rewards, values, next_values, terms, truncs = [], [], [], [], [], []
            episode_results = []
            for _ in range(cfg['rollout_steps']):
                for i, env in enumerate(envs):
                    obs, state, legal = observe(env)
                    backend.carry = carries[i]
                    actions, cache = backend.act(obs, legal)
                    carries[i] = backend.carry
                    if not legal[np.arange(len(actions)), actions].all():
                        raise AssertionError('Illegal action in rollout')
                    value = backend.value(state)
                    reward, done, info_step = env.step(actions.tolist())
                    terminated, truncated = transition_flags(env, done, info_step)
                    # Obtain the final observation BEFORE resetting; truncations
                    # bootstrap it, true terminals bootstrap zero.
                    next_value = 0. if terminated else backend.value(np.asarray(env.get_state(), np.float32))
                    cache['state'] = state
                    records.append(cache); rewards.append(float(reward)); values.append(value)
                    next_values.append(next_value); terms.append(terminated); truncs.append(truncated)
                    env_steps += 1
                    episode_returns[i] += reward; episode_lengths[i] += 1
                    if done:
                        completed_episodes += 1
                        episode_results.append((episode_returns[i], float(info_step.get('battle_won', False)), episode_lengths[i]))
                        episode_returns[i] = 0.; episode_lengths[i] = 0
                        env.reset(); carries[i] = None
            shape = (cfg['rollout_steps'], len(envs))
            arrays = [np.asarray(x).reshape(shape) for x in [rewards, values, next_values, terms, truncs]]
            advantages, returns = compute_gae(*arrays, gamma=cfg['gamma'], lam=cfg['gae_lambda'])
            batch = stack_records(records)
            batch['advantages'] = normalize_advantages(advantages).reshape(-1)
            batch['returns'] = returns.reshape(-1)
            replay_error = backend.replay_error(batch)
            if replay_error > 2e-3:
                raise AssertionError('Unchanged policy failed likelihood replay: ' + str(replay_error))
            metrics = []
            actor_update = iteration >= cfg['critic_warmup_iters']
            stop = False
            for epoch in range(cfg['update_epochs']):
                indices = np.random.permutation(len(records))
                for offset in range(0, len(records), cfg['batch_size']):
                    idx = indices[offset:offset+cfg['batch_size']]
                    m = backend.update({k: v[idx] for k, v in batch.items()}, actor_update)
                    total_actor_updates += int(m['actor_updated'])
                    metrics.append(m)
                    if actor_update and m['approx_kl'] > cfg['target_kl']:
                        stop = True; break
                if stop:
                    break
            summary = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
            summary.update(iteration=iteration+1, env_steps=env_steps, replay_error=replay_error,
                           early_stop=float(stop), elapsed_s=time.monotonic()-started,
                           completed_episodes=completed_episodes)
            if episode_results:
                summary['rollout_return'] = float(np.mean([r[0] for r in episode_results]))
                summary['rollout_win_rate'] = float(np.mean([r[1] for r in episode_results]))
            emit('train', **finite_metrics(summary))
            ext = '.pkl' if cfg['variant'] == 'jax' else '.pt'
            if (iteration+1) % cfg['save_every'] == 0:
                backend.save(out / ('checkpoint_%d%s' % (iteration+1, ext)), iteration+1, env_steps)
            if (iteration+1) % cfg['eval_every'] == 0:
                for deterministic in [True, False]:
                    emit('eval', iteration=iteration+1,
                         **evaluate(backend, make_env, cfg, cfg['eval_episodes'], deterministic))
        final_path = out / ('checkpoint_final' + ('.pkl' if cfg['variant'] == 'jax' else '.pt'))
        backend.save(final_path, cfg['iterations'], env_steps)
        before = backend.log_probs_numpy(batch).copy()
        backend.restore(final_path)
        after = backend.log_probs_numpy(batch)
        roundtrip_error = float(np.max(np.abs(before-after)))
        if roundtrip_error > 2e-3:
            raise AssertionError('Checkpoint round-trip changed likelihood')
        if sha256(cfg['checkpoint']) != checkpoint_hash:
            raise AssertionError('Source pretrained checkpoint was modified')
        actor_delta = float(np.linalg.norm(backend.actor_vector()-initial_actor)) if initial_actor is not None else None
        if cfg.get('smoke') and (total_actor_updates == 0 or actor_delta <= 0):
            raise AssertionError('Smoke run failed to update the pretrained actor')
        emit('complete', checkpoint=str(final_path), env_steps=env_steps,
             checkpoint_roundtrip_error=roundtrip_error, source_checkpoint_unchanged=True,
             actor_updates=total_actor_updates, actor_parameter_delta=actor_delta)
        atomic_json(out / 'status.json', {'status': 'complete', 'env_steps': env_steps,
                    'checkpoint': str(final_path), 'smoke': cfg.get('smoke', False)})
    except BaseException as e:
        atomic_json(out / 'status.json', {'status': 'failed', 'error': repr(e), 'env_steps': env_steps})
        raise
    finally:
        for env in envs:
            env.close()
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--project-root')
    parser.add_argument('--resume')
    parser.add_argument('--smoke', action='store_true', help='2 PPO updates, 1 environment; no production training')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--iterations', type=int)
    args = parser.parse_args()
    cfg = dict(DEFAULTS, **json.loads(Path(args.config).read_text()))
    if args.project_root: cfg['project_root'] = args.project_root
    if args.output_dir: cfg['output_dir'] = args.output_dir
    if args.smoke:
        cfg.update(smoke=True, n_envs=1, rollout_steps=64, iterations=2, update_epochs=2,
                   batch_size=32, eval_episodes=1, eval_every=2, save_every=1, critic_warmup_iters=0)
    if args.iterations is not None: cfg['iterations'] = args.iterations
    run(cfg, args.resume, args.eval_only)


if __name__ == '__main__':
    main()
