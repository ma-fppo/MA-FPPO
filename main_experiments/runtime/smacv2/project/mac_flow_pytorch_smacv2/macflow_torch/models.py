from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def mlp(in_dim: int, hidden_dims: Iterable[int], out_dim: int, layer_norm: bool = False) -> nn.Sequential:
    layers = []
    last = in_dim
    for hidden in hidden_dims:
        layers.append(nn.Linear(last, hidden))
        layers.append(nn.GELU())
        if layer_norm:
            layers.append(nn.LayerNorm(hidden))
        last = hidden
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class SequenceLSTMEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 64,
        pre_mlp_dims: Tuple[int, ...] = (128,),
        lstm_layers: int = 1,
        layer_norm: bool = True,
    ):
        super().__init__()
        pre_out = pre_mlp_dims[-1] if pre_mlp_dims else obs_dim
        self.pre = mlp(obs_dim, pre_mlp_dims[:-1], pre_out, layer_norm=False) if pre_mlp_dims else nn.Identity()
        self.lstm = nn.LSTM(pre_out, hidden_dim, num_layers=lstm_layers, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity()
        self.hidden_dim = hidden_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch, time, n_agents, obs_dim = obs.shape
        x = obs.permute(0, 2, 1, 3).reshape(batch * n_agents, time, obs_dim)
        x = self.pre(x)
        x, _ = self.lstm(x)
        x = self.norm(x)
        return x.reshape(batch, n_agents, time, self.hidden_dim).permute(0, 2, 1, 3)


class ActorVectorField(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        time_dim: int = 16,
        hidden_dims: Tuple[int, ...] = (256, 256, 256, 256),
        layer_norm: bool = False,
        include_time: bool = True,
    ):
        super().__init__()
        self.include_time = include_time
        input_dim = obs_dim + action_dim + (time_dim if include_time else 0)
        self.net = mlp(input_dim, hidden_dims, action_dim, layer_norm=layer_norm)

    def forward(self, obs: torch.Tensor, actions_or_noise: torch.Tensor, time_embed: torch.Tensor | None = None) -> torch.Tensor:
        inputs = [obs, actions_or_noise]
        if self.include_time:
            if time_embed is None:
                raise ValueError("time_embed is required for this vector field")
            inputs.append(time_embed)
        return self.net(torch.cat(inputs, dim=-1))


class EnsembleQ(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 256, 256),
        num_ensembles: int = 2,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.qs = nn.ModuleList(
            [mlp(obs_dim + action_dim, hidden_dims, 1, layer_norm=layer_norm) for _ in range(num_ensembles)]
        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, actions], dim=-1)
        values = [q(x).squeeze(-1) for q in self.qs]
        return torch.stack(values, dim=0)


@dataclass
class MACFlowConfig:
    obs_dim: int
    num_agents: int
    action_dim: int
    lr: float = 3e-4
    discount: float = 0.99
    tau: float = 0.005
    alpha: float = 3.0
    q_weight: float = 1.0
    flow_steps: int = 10
    t_embed_frequencies: int = 8
    actor_hidden_dims: Tuple[int, ...] = (256, 256, 256, 256)
    value_hidden_dims: Tuple[int, ...] = (256, 256, 256, 256)
    lstm_hidden_dim: int = 64
    lstm_pre_mlp_dims: Tuple[int, ...] = (128,)
    lstm_layers: int = 1


class MACFlowTorch(nn.Module):
    def __init__(self, cfg: MACFlowConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SequenceLSTMEncoder(
            obs_dim=cfg.obs_dim,
            hidden_dim=cfg.lstm_hidden_dim,
            pre_mlp_dims=cfg.lstm_pre_mlp_dims,
            lstm_layers=cfg.lstm_layers,
            layer_norm=True,
        )
        time_dim = 2 * cfg.t_embed_frequencies
        self.actor_bc_flow = ActorVectorField(
            cfg.lstm_hidden_dim,
            cfg.action_dim,
            time_dim=time_dim,
            hidden_dims=cfg.actor_hidden_dims,
            layer_norm=False,
            include_time=True,
        )
        self.actor_onestep_flow = ActorVectorField(
            cfg.lstm_hidden_dim,
            cfg.action_dim,
            time_dim=time_dim,
            hidden_dims=cfg.actor_hidden_dims,
            layer_norm=False,
            include_time=False,
        )
        self.q = EnsembleQ(
            cfg.lstm_hidden_dim,
            cfg.action_dim,
            hidden_dims=cfg.value_hidden_dims,
            num_ensembles=2,
            layer_norm=True,
        )
        self.target_q = EnsembleQ(
            cfg.lstm_hidden_dim,
            cfg.action_dim,
            hidden_dims=cfg.value_hidden_dims,
            num_ensembles=2,
            layer_norm=True,
        )
        self.target_q.load_state_dict(self.q.state_dict())

    def time_sin_embed(self, t: torch.Tensor) -> torch.Tensor:
        freqs = torch.tensor(
            [2**i for i in range(self.cfg.t_embed_frequencies)],
            dtype=t.dtype,
            device=t.device,
        ).view(*([1] * (t.ndim - 1)), -1)
        angles = t * freqs * math.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    @torch.no_grad()
    def soft_update_target(self) -> None:
        for param, target_param in zip(self.q.parameters(), self.target_q.parameters()):
            target_param.data.mul_(1.0 - self.cfg.tau).add_(param.data, alpha=self.cfg.tau)

    @torch.no_grad()
    def compute_flow_actions(self, obs_enc: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        actions = noise
        steps = int(self.cfg.flow_steps)
        for i in range(steps):
            t_scalar = torch.full((*actions.shape[:-1], 1), float(i) / steps, device=actions.device)
            time_embed = self.time_sin_embed(t_scalar)
            actions = actions + self.actor_bc_flow(obs_enc, actions, time_embed) / steps
        return actions.argmax(dim=-1)

    def sample_actions(self, obs: torch.Tensor, legal_actions: torch.Tensor | None = None) -> torch.Tensor:
        if obs.ndim == 2:
            obs = obs.unsqueeze(0).unsqueeze(0)
        elif obs.ndim == 3:
            obs = obs.unsqueeze(1)
        enc = self.encoder(obs)[:, -1]
        noise = torch.zeros((*enc.shape[:-1], self.cfg.action_dim), dtype=enc.dtype, device=enc.device)
        logits = self.actor_onestep_flow(enc, noise)
        if legal_actions is not None:
            if legal_actions.ndim == 2:
                legal_actions = legal_actions.unsqueeze(0)
            logits = logits.masked_fill(legal_actions <= 0, -1e9)
        return logits.argmax(dim=-1)


def one_hot(actions: torch.Tensor, action_dim: int) -> torch.Tensor:
    return F.one_hot(actions.long(), num_classes=action_dim).float()

