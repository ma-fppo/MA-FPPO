from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


@dataclass
class SMACv2DatasetSpec:
    task: str
    split: str = "Replay"
    data_root: str = "data/smacv2"

    @property
    def path(self) -> str:
        return os.path.join(self.data_root, self.task, self.split)


class ConvertedSMACv2Sequences:
    """Memory-mapped sampler for the flattened converted OG-MARL SMACv2 arrays."""

    def __init__(self, spec: SMACv2DatasetSpec, sequence_length: int = 20):
        self.spec = spec
        self.sequence_length = int(sequence_length)
        self.obs = np.load(os.path.join(spec.path, "obs.npy"), mmap_mode="r")
        self.actions = np.load(os.path.join(spec.path, "actions.npy"), mmap_mode="r")
        self.rewards = np.load(os.path.join(spec.path, "rewards.npy"), mmap_mode="r")
        self.legals = np.load(os.path.join(spec.path, "legals.npy"), mmap_mode="r")
        self.path_lengths = np.load(os.path.join(spec.path, "path_lengths.npy"))

        self.episode_offsets = np.concatenate([[0], np.cumsum(self.path_lengths[:-1])]).astype(np.int64)
        eligible = np.where(self.path_lengths >= self.sequence_length)[0]
        if len(eligible) == 0:
            raise ValueError(f"No episodes have length >= {self.sequence_length}")
        self.eligible_episodes = eligible.astype(np.int64)

        self.num_agents = int(self.obs.shape[1])
        self.obs_dim = int(self.obs.shape[2])
        self.action_dim = int(self.legals.shape[2])
        self.aug_obs_dim = self.obs_dim + self.num_agents

    def __len__(self) -> int:
        return int(np.sum(np.maximum(self.path_lengths - self.sequence_length + 1, 0)))

    def sample_numpy(self, batch_size: int) -> Dict[str, np.ndarray]:
        batch_size = int(batch_size)
        obs, actions, rewards, legals, terminals = [], [], [], [], []
        eps = np.random.choice(self.eligible_episodes, size=batch_size, replace=True)
        for ep in eps:
            ep_len = int(self.path_lengths[ep])
            offset = int(self.episode_offsets[ep])
            start = offset + np.random.randint(0, ep_len - self.sequence_length + 1)
            end = start + self.sequence_length
            obs.append(np.asarray(self.obs[start:end], dtype=np.float32))
            actions.append(np.asarray(self.actions[start:end], dtype=np.int64))
            rewards.append(np.asarray(self.rewards[start:end], dtype=np.float32))
            legals.append(np.asarray(self.legals[start:end], dtype=np.float32))

            term = np.zeros((self.sequence_length, self.num_agents), dtype=np.float32)
            if end == offset + ep_len:
                term[-1] = 1.0
            terminals.append(term)

        return {
            "obs": np.stack(obs, axis=0),
            "actions": np.stack(actions, axis=0),
            "rewards": np.stack(rewards, axis=0),
            "legals": np.stack(legals, axis=0),
            "terminals": np.stack(terminals, axis=0),
        }

    def sample_torch(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        batch = self.sample_numpy(batch_size)
        return {
            "obs": torch.as_tensor(batch["obs"], device=device, dtype=torch.float32),
            "actions": torch.as_tensor(batch["actions"], device=device, dtype=torch.long),
            "rewards": torch.as_tensor(batch["rewards"], device=device, dtype=torch.float32),
            "legals": torch.as_tensor(batch["legals"], device=device, dtype=torch.float32),
            "terminals": torch.as_tensor(batch["terminals"], device=device, dtype=torch.float32),
        }


def append_agent_id(obs: torch.Tensor) -> torch.Tensor:
    """Append one-hot agent ids to tensors shaped (..., num_agents, obs_dim)."""
    n_agents = obs.shape[-2]
    eye = torch.eye(n_agents, dtype=obs.dtype, device=obs.device)
    ids = eye.view(*([1] * (obs.ndim - 2)), n_agents, n_agents)
    ids = ids.expand(*obs.shape[:-1], n_agents)
    return torch.cat([obs, ids], dim=-1)


def iter_tasks(tasks: str) -> Iterable[str]:
    return (task.strip() for task in tasks.split(",") if task.strip())

