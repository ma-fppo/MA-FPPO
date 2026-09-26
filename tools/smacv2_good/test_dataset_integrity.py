"""Synthetic boundary/selection tests; these are NOT real SMACv2 rollouts."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dataset_io import atomic_json, digest, validate_episode, write_shard
from select_good import quality_key, select


def fixture(episode_id, win=1, ret=10.0, truncated=False):
    length = 1 + episode_id % 5
    rng = np.random.default_rng(episode_id)
    obs = np.concatenate([rng.normal(size=(length + 1, 2, 3)).astype(np.float32),
                          np.broadcast_to(np.eye(2, dtype=np.float32), (length + 1, 2, 2))], -1)
    states = rng.normal(size=(length + 1, 4)).astype(np.float32)
    term, trunc = np.zeros((length, 2), bool), np.zeros((length, 2), bool)
    (trunc if truncated else term)[-1] = True
    rewards = np.zeros((length, 2), np.float32); rewards[-1] = ret
    d = dict(obs=obs[:-1].copy(), next_obs=obs[1:].copy(), states=states[:-1].copy(), next_states=states[1:].copy(),
             actions=np.zeros((length, 2), np.int64), rewards=rewards,
             legals=np.ones((length, 2, 3), bool), next_legals=np.ones((length, 2, 3), bool),
             terminals=term, truncations=trunc, discounts=1.0 - term.astype(np.float32))
    m = dict(episode_id=episode_id, capability_seed=1000000 + episode_id,
             length=length, win=win, terminated=not truncated, truncated=truncated,
             source_checkpoint_sha256='synthetic-only', policy='deployment', **{'return': ret})
    return d, m


class Integrity(unittest.TestCase):
    def test_true_final_observation_and_timeout(self):
        for truncated in (False, True):
            d, m = fixture(7, truncated=truncated)
            validate_episode(d, m)
            self.assertTrue(np.all(d['discounts'][-1] == (1 if truncated else 0)))
            d['next_obs'][0, 0, 0] += 100
            with self.assertRaisesRegex(ValueError, 'alignment'):
                validate_episode(d, m)

    def test_illegal_action_and_false_reward(self):
        d, m = fixture(2); d['legals'][:, :, 0] = False
        with self.assertRaisesRegex(ValueError, 'illegal'):
            validate_episode(d, m)
        d, m = fixture(3); m['return'] += 1
        with self.assertRaisesRegex(ValueError, 'outcome'):
            validate_episode(d, m)

    def test_global_top_quarter_and_flat_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'raw').mkdir()
            episodes = [fixture(i, win=int(i % 3 == 0), ret=float(100 - i if i % 3 else i),
                                truncated=bool(i % 2)) for i in range(16)]
            # Winning trajectories deliberately have smaller raw rewards than losses.
            # Membership must still be the global best four winning trajectories.
            cfg = dict(task='synthetic', mode='preflight', target_episodes=16, keep_episodes=4,
                       seed_start=1000000, ppo_checkpoint='synthetic-only', ppo_sha256='synthetic-only')
            atomic_json(root / 'collection_config.json', cfg)
            atomic_json(root / 'status.json', dict(status='collected'))
            shards = []
            for start in range(0, 16, 5):
                p = root / 'raw' / ('shard_%05d.npz' % len(shards))
                write_shard(p, episodes[start:start + 5])
                shards.append(dict(file=p.name, sha256=digest(p)))
            atomic_json(root / 'raw_manifest.json', dict(shards=shards,
                        collection_config_sha256=digest(root / 'collection_config.json')))
            report = select(root)
            self.assertEqual(report['raw']['episodes'], 16)
            self.assertEqual(report['selected']['episodes'], 4)
            self.assertEqual(report['selected']['win_rate'], 1.0)
            out = root / 'Good_preflight_only'
            self.assertEqual(set(np.load(out / 'episode_ids.npy')), {6, 9, 12, 15})
            lengths = np.load(out / 'path_lengths.npy')
            self.assertEqual(int(lengths.sum()), report['selected']['transitions'])
            self.assertEqual(len(np.load(out / 'actions.npy')), int(lengths.sum()))
            with self.assertRaises(FileExistsError):
                select(root)
            # A changed raw artifact cannot silently enter the training dataset.
            with open(root / 'raw' / shards[0]['file'], 'ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                select(root)

    def test_tie_break_is_repeatable(self):
        records = [fixture(i, win=1, ret=20.0)[1] for i in range(20)]
        self.assertEqual(sorted(records, key=quality_key), sorted(records[::-1], key=quality_key))


if __name__ == '__main__':
    unittest.main(verbosity=2)
