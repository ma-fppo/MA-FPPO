"""Numerical and environment utilities shared by native Torch and JAX PPO."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path

import numpy as np


def seed_environment(env, seed):
    """Seed SMACv2's nested default_rng instances, not just legacy NumPy RNG."""
    random.seed(seed); np.random.seed(seed)
    sequence = np.random.SeedSequence(seed)
    seen, generators = set(), {}
    def visit(obj):
        if isinstance(obj, np.random.Generator):
            if id(obj) not in generators:
                generators[id(obj)] = np.random.default_rng(sequence.spawn(1)[0])
            return generators[id(obj)]
        if id(obj) in seen: return obj
        seen.add(id(obj))
        if isinstance(obj, dict):
            for key in sorted(obj, key=str): obj[key] = visit(obj[key])
        elif isinstance(obj, list):
            for i in range(len(obj)): obj[i] = visit(obj[i])
        elif isinstance(obj, tuple):
            return tuple(visit(item) for item in obj)
        elif hasattr(obj, '__dict__'):
            for key,value in sorted(vars(obj).items()): setattr(obj, key, visit(value))
        return obj
    visit(getattr(env, 'env_key_to_distribution_map', {}))
    return len(generators)


def load_environment_factory(project_root):
    # Load this torch-free module without importing either training framework.
    path = Path(project_root) / 'mac_flow_pytorch_smacv2/macflow_torch/smacv2_env.py'
    spec = importlib.util.spec_from_file_location('macflow_env_factory', str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_smacv2_env


def observe(env):
    obs = np.asarray(env.get_obs(), dtype=np.float32)
    state = np.asarray(env.get_state(), dtype=np.float32)
    legal = np.asarray(env.get_avail_actions(), dtype=bool)
    if not np.isfinite(obs).all() or not np.isfinite(state).all():
        raise FloatingPointError('Non-finite environment observation/state')
    if not legal.any(axis=-1).all():
        raise ValueError('An agent has no legal action; refusing an undefined policy')
    return obs, state, legal


def transition_flags(env, done, info):
    # SMAC's default finite episode limit is a terminal task horizon. Only an
    # explicit episode_limit flag denotes a continuing-task time truncation.
    truncated = bool(done and info.get('episode_limit', False))
    return bool(done and not truncated), truncated


def compute_gae(rewards, values, next_values, terminated, truncated, gamma=.99, lam=.95):
    """Bootstrap truncations from the final state, but never cross a reset."""
    rewards, values, next_values = [np.asarray(x, np.float32) for x in (rewards, values, next_values)]
    terminated, truncated = np.asarray(terminated, bool), np.asarray(truncated, bool)
    if not (rewards.shape == values.shape == next_values.shape == terminated.shape == truncated.shape):
        raise ValueError('GAE arrays must have identical shapes')
    advantage = np.zeros_like(rewards)
    carry = np.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * (~terminated[t]) - values[t]
        carry = delta + gamma * lam * (~(terminated[t] | truncated[t])) * carry
        advantage[t] = carry
    return advantage, advantage + values


def normalize_advantages(advantages):
    x = np.asarray(advantages, np.float32)
    return (x - x.mean()) / max(float(x.std()), 1e-8)


def stack_records(records):
    return {k: np.stack([r[k] for r in records]) for k in records[0]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(str(tmp), str(path))


def finite_metrics(metrics):
    if not all(np.isfinite(float(v)) for v in metrics.values()):
        raise FloatingPointError('Non-finite PPO metrics: ' + str(metrics))
    return {k: float(v) for k, v in metrics.items()}
