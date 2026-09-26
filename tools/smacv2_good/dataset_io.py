"""Lossless episode shards and checks for the post-trained SMACv2 dataset."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

FIELDS = ('obs', 'next_obs', 'states', 'next_states', 'actions', 'rewards',
          'legals', 'next_legals', 'terminals', 'truncations', 'discounts')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def validate_episode(data, meta):
    length = int(meta['length'])
    if length <= 0 or set(data) != set(FIELDS):
        raise ValueError('Incomplete episode fields or empty episode')
    n = data['actions'].shape[1]
    for k, x in data.items():
        if len(x) != length or not np.isfinite(x).all():
            raise ValueError('Invalid array: ' + k)
    if data['actions'].dtype.kind not in 'iu':
        raise ValueError('Actions must be integer')
    for prefix in ('', 'next_'):
        legal = data[prefix + 'legals']
        if legal.shape[:2] != (length, n) or not legal.any(-1).all():
            raise ValueError('Invalid legal masks')
        obs = data[prefix + 'obs']
        if obs.shape[:2] != (length, n):
            raise ValueError('Invalid observations')
        if not np.array_equal(obs[..., -n:], np.broadcast_to(np.eye(n), (length, n, n))):
            raise ValueError('Expected exactly the existing suffix agent IDs')
    actions = data['actions']
    if (actions < 0).any() or (actions >= data['legals'].shape[-1]).any():
        raise ValueError('Action outside environment action space')
    if not np.take_along_axis(data['legals'], actions[..., None], -1).all():
        raise ValueError('Recorded illegal action')
    for k in ('rewards', 'terminals', 'truncations', 'discounts'):
        if data[k].shape != (length, n):
            raise ValueError('Invalid agent dimension: ' + k)
    if not np.array_equal(data['rewards'], np.repeat(data['rewards'][:, :1], n, axis=1)):
        raise ValueError('Team reward must be shared, not summed over agents')
    term, trunc = data['terminals'], data['truncations']
    if term.dtype != np.bool_ or trunc.dtype != np.bool_:
        raise ValueError('Boundary flags must be boolean')
    for mask in (term, trunc):
        if not np.array_equal(mask, np.repeat(mask[:, :1], n, axis=1)):
            raise ValueError('Environment boundaries must agree across agents')
    if (term & trunc).any() or (term | trunc)[:-1].any() or not (term | trunc)[-1].all():
        raise ValueError('Episode must have exactly one final boundary')
    if not np.array_equal(data['discounts'], 1 - term.astype(np.float32)):
        raise ValueError('Discount must preserve time-limit bootstrap')
    for left, right in [('next_obs', 'obs'), ('next_states', 'states'), ('next_legals', 'legals')]:
        if not np.array_equal(data[left][:-1], data[right][1:]):
            raise ValueError('Broken transition alignment: ' + left)
    ret = float(data['rewards'][:, 0].sum(dtype=np.float64))
    if abs(ret - meta['return']) > 1e-6 or meta['win'] not in (0, 1):
        raise ValueError('Invalid episode outcome')
    if bool(meta['terminated']) != bool(term[-1, 0]) or bool(meta['truncated']) != bool(trunc[-1, 0]):
        raise ValueError('Outcome and transition boundary disagree')


def write_shard(path, episodes):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    for data, meta in episodes:
        validate_episode(data, meta)
    arrays = {k: np.concatenate([d[k] for d, _ in episodes], axis=0) for k in FIELDS}
    arrays['path_lengths'] = np.asarray([m['length'] for _, m in episodes], np.int64)
    arrays['metadata_json'] = np.asarray([json.dumps(m, sort_keys=True, allow_nan=False) for _, m in episodes])
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        # Uncompressed shards avoid spending scarce rollout CPU on compression.
        np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(str(tmp), str(path))


def read_metadata(path):
    with np.load(path, allow_pickle=False) as z:
        return [json.loads(str(s)) for s in z['metadata_json']]


def summary(records):
    returns = np.asarray([r['return'] for r in records], np.float64)
    return dict(episodes=len(records), transitions=sum(r['length'] for r in records),
                wins=sum(r['win'] for r in records),
                win_rate=float(np.mean([r['win'] for r in records])),
                mean_return=float(returns.mean()), std_return=float(returns.std()),
                return_quantiles=dict(zip(['min', 'p25', 'p50', 'p75', 'max'],
                                         np.quantile(returns, [0, .25, .5, .75, 1]).tolist())),
                mean_length=float(np.mean([r['length'] for r in records])))
