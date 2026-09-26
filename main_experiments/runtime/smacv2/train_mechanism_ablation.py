#!/usr/bin/env python
"""Isolated student-noise and recurrent-encoder PPO experiments."""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np

from common import atomic_json, compute_gae, finite_metrics, load_environment_factory, normalize_advantages, sha256
from train import DEFAULTS, evaluate
from vector_env import ParallelEnv
from mechanism_evaluation import evaluate_parallel
from partial_rollout import act_prefix, step_prefix


def run(cfg, resume=None):
    out = Path(cfg['output_dir'])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose a fresh output directory, including when resuming')
    out.mkdir(parents=True)
    os.environ.setdefault('SC2PATH', str(Path.home() / 'StarCraftII'))
    os.environ.setdefault('WANDB_MODE', 'offline')
    random.seed(cfg['seed']); np.random.seed(cfg['seed'])
    env = None
    steps, iteration, total_updates = 0, 0, 0
    log = open(out/'metrics.jsonl', 'w', buffering=1)
    def emit(kind, **fields):
        record = dict(kind=kind, variant=cfg['variant'], task=cfg['task'], **fields)
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        print(kind.upper()+' '+line, flush=True); log.write(line+'\n')
    try:
        initialization_started = time.monotonic()
        env = ParallelEnv(cfg['project_root'], cfg['task'], cfg['n_envs'], cfg['seed'],
                          out/'workers', startup_batch=cfg.get('startup_batch', 4))
        if cfg['variant']=='jax':
            from jax_backend import JaxBackend
            backend = JaxBackend(cfg, env.info)
        else:
            from mechanism_backend import MechanismBackend as TorchBackend
            backend = TorchBackend(cfg, env.info)
        cfg['source_checkpoint_sha256'] = sha256(cfg['checkpoint'])
        if resume:
            iteration, steps = backend.restore(resume)
        initial_steps, initial_iteration = steps, iteration
        if cfg['total_env_steps'] <= steps:
            raise ValueError('Target environment budget has already been reached')
        atomic_json(out/'run_config.json', dict(cfg, resumed_from=resume, env_info=env.info,
                   initial_env_steps=steps, initial_iteration=iteration,
                   environment_rng='independent worker; seedsequence_tree_v1 for nested SMACv2 generators'))
        ext = '.pkl' if cfg['variant']=='jax' else '.pt'
        initial_actor = backend.actor_vector()
        start_checkpoint = out/('checkpoint_initial'+ext)
        backend.save(start_checkpoint, iteration, steps)
        make_env = load_environment_factory(cfg['project_root'])
        best = None
        def assess(checkpoint, kind, count):
            nonlocal best
            for deterministic in [True, False]:
                metric = evaluate_parallel(backend, cfg, count, deterministic)
                emit(kind, iteration=iteration, env_steps=steps, checkpoint=str(checkpoint), **metric)
                score = (metric['win_rate'], metric['avg_return'])
                if deterministic and (best is None or score > tuple(best['score'])):
                    best = dict(score=score, checkpoint=str(checkpoint), env_steps=steps, **metric)
                    atomic_json(out/'best_validation.json', best)
        if not cfg.get('skip_initial_eval', False):
            assess(start_checkpoint, 'restored_eval' if resume else 'baseline', cfg['eval_episodes'])
        backend.reset()
        emit('initialized', elapsed_s=time.monotonic()-initialization_started, n_envs=cfg['n_envs'],
             initial_env_steps=steps, resumed_from=resume)
        episode_returns = np.zeros(cfg['n_envs'])
        episode_lengths = np.zeros(cfg['n_envs'], int)
        completed_episodes = 0
        eval_interval = cfg.get('eval_interval_steps', 65536)
        save_interval = cfg.get('save_interval_steps', 16384)
        next_eval = (steps//eval_interval+1)*eval_interval
        next_save = (steps//save_interval+1)*save_interval
        started = time.monotonic()
        while steps < cfg['total_env_steps']:
            iteration_started = time.monotonic()
            before_steps = steps
            remaining = cfg['total_env_steps'] - steps
            active_envs = min(cfg['n_envs'], remaining)
            horizon = min(cfg['rollout_steps'], remaining // active_envs)
            warmup = cfg.get('critic_warmup_steps', 2560)
            if steps < warmup:
                horizon = min(horizon, int(math.ceil((warmup-steps)/cfg['n_envs'])))
            records, rewards, values, next_values, terms, truncs = [], [], [], [], [], []
            episodes = []
            timings = dict(action_seconds=0., value_seconds=0., environment_seconds=0.)
            for _ in range(horizon):
                state = env.state[:active_envs].copy()
                tick = time.monotonic()
                actions, cache = act_prefix(backend, env, active_envs)
                timings['action_seconds'] += time.monotonic()-tick
                if not np.take_along_axis(env.legal[:active_envs], actions[..., None], -1).all():
                    raise AssertionError('Illegal batched action')
                tick = time.monotonic()
                value = backend.values_batch(state)
                timings['value_seconds'] += time.monotonic()-tick
                tick = time.monotonic()
                transition = step_prefix(env, actions)
                timings['environment_seconds'] += time.monotonic()-tick
                tick = time.monotonic()
                next_value = backend.values_batch(transition['bootstrap_state'])*(~transition['terminated'])
                timings['value_seconds'] += time.monotonic()-tick
                cache['state'] = state
                records.append(cache); rewards.append(transition['reward']); values.append(value)
                next_values.append(next_value); terms.append(transition['terminated']); truncs.append(transition['truncated'])
                steps += active_envs
                episode_returns[:active_envs] += transition['reward']; episode_lengths[:active_envs] += 1
                for i in np.flatnonzero(transition['done']):
                    episodes.append((episode_returns[i], transition['win'][i], episode_lengths[i]))
                    episode_returns[i] = 0.; episode_lengths[i] = 0
                    completed_episodes += 1
            advantages, returns = compute_gae(rewards, values, next_values, terms, truncs,
                                               cfg['gamma'], cfg['gae_lambda'])
            batch = backend.prepare_batch(records)
            batch['advantages'] = normalize_advantages(advantages).reshape(-1)
            batch['returns'] = returns.reshape(-1)
            tick = time.monotonic()
            replay_error = backend.replay_error(batch)
            if replay_error > 2e-3:
                raise AssertionError('Batched likelihood replay failed: '+str(replay_error))
            timings['replay_seconds'] = time.monotonic()-tick
            tick = time.monotonic()
            backend.set_progress(before_steps)
            objective_diagnostics = backend.objective_diagnostics(batch)
            old_policy_logits = backend.policy_snapshot(batch)
            metrics, stop = [], False
            actor_update = before_steps >= warmup
            for _ in range(cfg['update_epochs']):
                indices = np.random.permutation(len(batch['returns']))
                for offset in range(0, len(indices), cfg['batch_size']):
                    idx = indices[offset:offset+cfg['batch_size']]
                    item = backend.update({k: v[idx] for k,v in batch.items()}, actor_update)
                    metrics.append(item); total_updates += int(item['actor_updated'])
                    if actor_update and item['approx_kl'] > cfg['target_kl']:
                        stop = True; break
                if stop: break
            timings['update_seconds'] = time.monotonic()-tick
            iteration += 1
            summary = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
            summary.update(backend.post_update_diagnostics(batch, old_policy_logits))
            summary.update(objective_diagnostics)
            backend.refresh_carry(batch, active_envs)
            duration = time.monotonic()-iteration_started
            summary.update(timings, iteration=iteration, env_steps=steps, samples=steps-before_steps, active_envs=active_envs,
                           replay_error=replay_error, early_stop=float(stop),
                           elapsed_s=time.monotonic()-started, iteration_seconds=duration,
                           env_steps_per_second=(steps-before_steps)/duration, completed_episodes=completed_episodes)
            if episodes:
                summary.update(rollout_return=float(np.mean([x[0] for x in episodes])),
                               rollout_win_rate=float(np.mean([x[1] for x in episodes])))
            emit('train', **finite_metrics(summary))
            stop_requested = (out/'stop_after_checkpoint').exists()
            evaluate_now = steps >= next_eval
            if steps >= next_save or evaluate_now or stop_requested:
                path = out/('checkpoint_step_%d%s' % (steps, ext))
                backend.save(path, iteration, steps)
                next_save = (steps//save_interval+1)*save_interval
                if evaluate_now and not cfg.get('skip_periodic_eval', False):
                    assess(path, 'eval', cfg['eval_episodes'])
                next_eval = (steps//eval_interval+1)*eval_interval if evaluate_now else next_eval
                if stop_requested:
                    atomic_json(out/'status.json', dict(status='paused', env_steps=steps, checkpoint=str(path)))
                    emit('paused', checkpoint=str(path), env_steps=steps)
                    return
        assert steps == cfg['total_env_steps']
        final = out/('checkpoint_final'+ext)
        backend.save(final, iteration, steps)
        before = backend.log_probs_numpy(batch)
        backend.restore(final)
        roundtrip = float(np.max(np.abs(before-backend.log_probs_numpy(batch))))
        delta = float(np.linalg.norm(backend.actor_vector()-initial_actor))
        if roundtrip > 2e-3 or not total_updates or not delta > 0:
            raise AssertionError('Invalid checkpoint roundtrip or missing actor update')
        if sha256(cfg['checkpoint']) != cfg['source_checkpoint_sha256']:
            raise AssertionError('Original checkpoint was modified')
        if not cfg.get('skip_periodic_eval', False):
            assess(final, 'eval', cfg['eval_episodes'])
        emit('complete', checkpoint=str(final), env_steps=steps, new_env_steps=steps-initial_steps,
             new_iterations=iteration-initial_iteration, actor_updates=total_updates,
             actor_parameter_delta=delta, checkpoint_roundtrip_error=roundtrip,
             source_checkpoint_unchanged=True)
        atomic_json(out/'status.json', dict(status='complete', env_steps=steps, checkpoint=str(final)))
    except BaseException as error:
        atomic_json(out/'status.json', dict(status='failed', error=repr(error), env_steps=steps))
        raise
    finally:
        if env is not None: env.close()
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir')
    parser.add_argument('--resume')
    args = parser.parse_args()
    cfg = dict(DEFAULTS, **json.loads(Path(args.config).read_text()))
    if args.output_dir: cfg['output_dir'] = args.output_dir
    for key in ['n_envs', 'rollout_steps', 'total_env_steps', 'batch_size', 'update_epochs']:
        if int(cfg[key]) <= 0: raise ValueError(key+' must be positive')
    run(cfg, args.resume)


if __name__=='__main__':
    main()
