"""Read original Poor trajectories and match PPO's trailing agent-ID convention."""
import numpy as np
from macflow_torch.data import ConvertedSMACv2Sequences, SMACv2DatasetSpec


class PoorSequences(ConvertedSMACv2Sequences):
    def __init__(self, root, task, sequence_length=20, split='Poor'):
        super().__init__(SMACv2DatasetSpec(task=task, split=split, data_root=root), sequence_length)
        self.ids = np.eye(self.num_agents, dtype=np.float32)

    def sample_numpy(self, batch_size):
        batch = super().sample_numpy(batch_size)
        obs = batch['obs']
        if not np.all(obs[..., :self.num_agents] == self.ids):
            raise ValueError('Source observations must begin with ordered one-hot agent IDs')
        batch['obs'] = np.concatenate([obs[..., self.num_agents:], obs[..., :self.num_agents]], axis=-1)
        return batch
