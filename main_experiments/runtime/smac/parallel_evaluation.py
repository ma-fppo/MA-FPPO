"""Evaluate independent seeded episodes with batched actions and parallel SC2."""
import random
from pathlib import Path
import numpy as np
from vector_env import ParallelEnv


def evaluate_parallel(backend, cfg, episodes, deterministic):
    saved = backend.carry, backend.rng_state(), random.getstate(), np.random.get_state()
    result = []
    try:
        for start in range(0, episodes, cfg.get('eval_n_envs', 20)):
            count = min(cfg.get('eval_n_envs', 20), episodes-start)
            seed = cfg['eval_seed']+start
            env = ParallelEnv(cfg['project_root'], cfg['task'], count, seed,
                              Path(cfg['output_dir'])/'eval_workers',
                              startup_batch=cfg.get('startup_batch', 4), auto_reset=False)
            try:
                backend.reset()
                rngs = backend.make_rngs(range(seed, seed+count))
                active = np.ones(count, bool)
                returns, wins = np.zeros(count), np.zeros(count)
                lengths = np.zeros(count, int)
                while active.any():
                    actions, _ = backend.act_batch(env.obs, env.legal, env.resets,
                                                    deterministic=deterministic, rngs=rngs)
                    transition = env.step(actions, active)
                    returns += transition['reward']*active
                    lengths += active
                    wins = np.where(active & transition['done'], transition['win'], wins)
                    active &= ~transition['done']
                result.extend(zip(returns, wins, lengths))
            finally:
                env.close()
    finally:
        backend.carry = saved[0]; backend.restore_rng(saved[1])
        random.setstate(saved[2]); np.random.set_state(saved[3])
    data = np.asarray(result)
    return dict(avg_return=float(data[:,0].mean()),win_rate=float(data[:,1].mean()),
                avg_length=float(data[:,2].mean()),episodes=episodes,
                policy='deployment' if deterministic else 'ppo_sampling')
