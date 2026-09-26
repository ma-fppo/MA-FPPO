"""Memory-mapped OMAR arrays, all five data seeds, three controlled agents."""
from pathlib import Path
import numpy as np
import torch


def with_ids(obs):
    ids = np.broadcast_to(np.eye(3, dtype=np.float32), (*obs.shape[:-2], 3, 3))
    return np.concatenate((obs, ids), -1).astype(np.float32)


class Dataset:
    def __init__(self, root, task, split, seed=0):
        self.path = Path(root) / task / split
        self.seeds = sorted(self.path.glob('seed_*_data'))
        assert [p.name for p in self.seeds] == ['seed_%d_data' % i for i in range(5)]
        self.arrays = [{k: [np.load(p / ('%s_%d.npy' % (k, i)), mmap_mode='r') for i in range(3)]
                        for k in ('obs', 'next_obs', 'acs', 'rews', 'dones')} for p in self.seeds]
        self.rng = np.random.RandomState(seed)
        self.raw_obs_dim = self.arrays[0]['obs'][0].shape[-1]
        self.obs_dim = self.raw_obs_dim + 3
        self.lengths = np.array([len(a['obs'][0]) for a in self.arrays])
        assert all(self.lengths == self.lengths[0]) and all(self.lengths > 0) and all(self.lengths % 25 == 0)

    def sample(self, device, batch_size=32, sequence_length=20):
        choices = self.rng.randint(0, len(self.arrays), size=batch_size)
        starts = self.rng.randint(0, int(self.lengths[0] // 25), size=batch_size) * 25 + self.rng.randint(0, 26-sequence_length, size=batch_size)
        result = {k: [] for k in ('obs', 'next_obs', 'actions', 'rewards', 'terminals')}
        for s, start in zip(choices, starts):
            a = self.arrays[s]; rows = np.arange(start, start + sequence_length)
            for key, src in [('obs', 'obs'), ('next_obs', 'next_obs'), ('actions', 'acs'), ('rewards', 'rews'), ('terminals', 'dones')]:
                value = np.stack([v[rows] for v in a[src]], axis=1)
                if key in ('obs', 'next_obs'): value = with_ids(value)
                result[key].append(value)
        return {k: torch.as_tensor(np.stack(v), dtype=torch.float32, device=device) for k, v in result.items()}
