from __future__ import annotations

from dataclasses import asdict
from typing import Dict

import torch
from torch.nn import functional as F

from .models import MACFlowConfig, MACFlowTorch, one_hot


class MACFlowLearner:
    def __init__(self, cfg: MACFlowConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.model = MACFlowTorch(cfg).to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        action_dim = self.cfg.action_dim

        obs_enc = self.model.encoder(obs)

        with torch.no_grad():
            next_obs = obs_enc[:, 1:]
            next_noise = torch.randn((*next_obs.shape[:-1], action_dim), device=self.device)
            next_action_idx = self.model.compute_flow_actions(next_obs, next_noise)
            next_action_oh = one_hot(next_action_idx, action_dim)
            target_qs = self.model.target_q(next_obs, next_action_oh).mean(dim=0)
            target_q = rewards[:, :-1] + self.cfg.discount * (1.0 - terminals[:, 1:]) * target_qs
            mixed_target_q = target_q.sum(dim=-1)

        cur_obs = obs_enc[:, :-1]
        cur_action_oh = one_hot(actions[:, :-1], action_dim)
        q_cur = self.model.q(cur_obs, cur_action_oh).mean(dim=0)
        mixed_q = q_cur.sum(dim=-1)
        critic_loss = 0.5 * F.mse_loss(mixed_q, mixed_target_q)

        x1 = cur_action_oh
        x0 = torch.randn_like(x1)
        t_scalar = torch.rand((*x1.shape[:-1], 1), device=self.device)
        xt = (1.0 - t_scalar) * x0 + t_scalar * x1
        velocity = x1 - x0
        pred_velocity = self.model.actor_bc_flow(cur_obs, xt, self.model.time_sin_embed(t_scalar))
        bc_flow_loss = F.mse_loss(pred_velocity, velocity)

        with torch.no_grad():
            distill_noise = torch.randn_like(x1)
            target_flow_actions = self.model.compute_flow_actions(cur_obs, distill_noise)
        actor_logits = self.model.actor_onestep_flow(cur_obs, distill_noise)
        distill_loss = F.cross_entropy(actor_logits.reshape(-1, action_dim), target_flow_actions.reshape(-1))

        probs = torch.softmax(actor_logits, dim=-1)
        with torch.no_grad():
            all_q = []
            for action_id in range(action_dim):
                candidate = torch.zeros_like(x1)
                candidate[..., action_id] = 1.0
                all_q.append(self.model.q(cur_obs, candidate).mean(dim=0))
            all_q_tensor = torch.stack(all_q, dim=-1)
        expected_q = (probs * all_q_tensor).sum(dim=-1).sum(dim=-1)
        q_guidance_loss = -expected_q.mean()

        actor_loss = bc_flow_loss + self.cfg.alpha * distill_loss + self.cfg.q_weight * q_guidance_loss
        loss = critic_loss + actor_loss

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optim.step()
        self.model.soft_update_target()

        with torch.no_grad():
            pred_actions = actor_logits.argmax(dim=-1)
            bc_acc = (pred_actions == actions[:, :-1]).float().mean()
        return {
            "loss": float(loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "bc_flow_loss": float(bc_flow_loss.detach().cpu()),
            "distill_loss": float(distill_loss.detach().cpu()),
            "q_guidance_loss": float(q_guidance_loss.detach().cpu()),
            "mixed_q": float(mixed_q.mean().detach().cpu()),
            "bc_acc": float(bc_acc.detach().cpu()),
        }

    def save(self, path: str, extra: Dict | None = None) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),
            "config": asdict(self.cfg),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str, map_location: str | torch.device = "cpu") -> Dict:
        payload = torch.load(path, map_location=map_location)
        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optim.load_state_dict(payload["optimizer"])
        return payload
