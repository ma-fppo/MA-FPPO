"""Select whole episodes: wins first, then team return; export existing-loader arrays."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from dataset_io import FIELDS, atomic_json, digest, read_metadata, summary, validate_episode


def quality_key(record):
    # A deterministic pseudo-random tie break avoids preferring early episodes.
    tie = hashlib.sha256(str(record['episode_id']).encode()).hexdigest()
    return -record['win'], -record['return'], tie


def select(root):
    root = Path(root)
    cfg = json.loads((root / 'collection_config.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    if status['status'] != 'collected':
        raise ValueError('Collection must be complete before global selection')
    target, keep = cfg['target_episodes'], cfg['keep_episodes']
    if target != 4 * keep:
        raise ValueError('Expected exactly fourfold collection and top quarter')
    manifest = json.loads((root / 'raw_manifest.json').read_text())
    if manifest['collection_config_sha256'] != digest(root / 'collection_config.json'):
        raise ValueError('Collection configuration changed')
    records, shards = [], []
    for entry in manifest['shards']:
        path = root / 'raw' / entry['file']
        if digest(path) != entry['sha256']:
            raise ValueError('Raw shard changed: ' + str(path))
        metas = read_metadata(path)
        if any(m['source_checkpoint_sha256'] != cfg['ppo_sha256'] or m['policy'] != 'deployment' for m in metas):
            raise ValueError('Wrong collection policy provenance')
        records.extend(metas); shards.append((path, metas))
    ids = [m['episode_id'] for m in records]
    if len(ids) != target or set(ids) != set(range(target)):
        raise ValueError('Missing/duplicate episodes; refusing biased partial selection')
    if {m['capability_seed'] for m in records} != set(range(cfg['seed_start'], cfg['seed_start'] + target)):
        raise ValueError('Missing/duplicate collection seeds')
    ranked = sorted(records, key=quality_key)[:keep]
    chosen = {r['episode_id'] for r in ranked}
    # Store by source shard order; membership is determined by global quality.
    kept = [m for _, metas in shards for m in metas if m['episode_id'] in chosen]
    total = sum(m['length'] for m in kept)
    out = root / ('Good' if cfg['mode'] == 'formal' else 'Good_preflight_only')
    if out.exists():
        raise FileExistsError('Never overwrite an existing dataset: ' + str(out))
    temp = root / (out.name + '.building')
    temp.mkdir(exist_ok=False)
    with np.load(shards[0][0], allow_pickle=False) as z:
        destinations = {k: np.lib.format.open_memmap(str(temp / (k + '.npy')), mode='w+',
                         dtype=z[k].dtype, shape=(total,) + z[k].shape[1:]) for k in FIELDS}
    cursor = 0
    for path, metas in shards:
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in FIELDS}
            lengths = z['path_lengths']
            if list(lengths) != [m['length'] for m in metas] or any(len(x) != sum(lengths) for x in arrays.values()):
                raise ValueError('Shard length mismatch')
            offset = 0
            for meta in metas:
                length = meta['length']
                episode = {k: x[offset:offset + length] for k, x in arrays.items()}
                # Audit all raw episodes, including the three discarded quarters.
                validate_episode(episode, meta)
                if meta['episode_id'] in chosen:
                    for k in FIELDS:
                        destinations[k][cursor:cursor + length] = episode[k]
                    cursor += length
                offset += length
    if cursor != total:
        raise AssertionError('Wrong number of exported transitions')
    for x in destinations.values():
        x.flush()
    destinations.clear()
    np.save(temp / 'path_lengths.npy', np.asarray([m['length'] for m in kept], np.int64))
    np.save(temp / 'episode_wins.npy', np.asarray([m['win'] for m in kept], np.uint8))
    np.save(temp / 'episode_returns.npy', np.asarray([m['return'] for m in kept], np.float64))
    np.save(temp / 'episode_ids.npy', np.asarray([m['episode_id'] for m in kept], np.int64))
    with open(temp / 'episodes.jsonl', 'w') as f:
        for meta in kept:
            f.write(json.dumps(meta, sort_keys=True, allow_nan=False) + '\n')
    # Read back the flat export and audit its boundaries and episode statistics.
    flat = {k: np.load(temp / (k + '.npy'), mmap_mode='r') for k in FIELDS}
    pos = 0
    for meta in kept:
        end = pos + meta['length']
        validate_episode({k: x[pos:end] for k, x in flat.items()}, meta)
        pos = end
    flat.clear()
    report = dict(status='verified', dataset_kind='self_generated_posttrained_top_quarter',
                  official_og_marl_dataset=False, task=cfg['task'], mode=cfg['mode'],
                  collection_config_sha256=digest(root / 'collection_config.json'),
                  collector_hashes=cfg.get('collector_hashes', {}),
                  source_checkpoint=cfg['ppo_checkpoint'], source_checkpoint_sha256=cfg['ppo_sha256'],
                  selection='battle_won descending, undiscounted shared team return descending, SHA256 episode-ID tie break',
                  raw=summary(records), selected=summary(kept),
                  selected_episode_ids_in_quality_order=[r['episode_id'] for r in ranked],
                  quality_cutoff={k: ranked[-1][k] for k in ('win', 'return')},
                  observation_format='raw local observation plus one suffix one-hot agent ID',
                  reward_format='one shared team reward repeated per agent; never summed across agents',
                  terminal_semantics='true terminal and time truncation retained separately; next observations precede reset',
                  evaluation_note='Selected-data win rate is selection-conditioned, not an unbiased policy evaluation. Use fresh unfiltered environment seeds for downstream evaluation.',
                  files={p.name: digest(p) for p in temp.iterdir() if p.is_file()})
    atomic_json(temp / 'dataset_manifest.json', report)
    os.rename(str(temp), str(out))
    atomic_json(root / 'selection_report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    args = parser.parse_args()
    with open(Path(args.root) / '.selection.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = select(args.root)
    print(json.dumps({k: report[k] for k in ('status', 'raw', 'selected')}, indent=2))
