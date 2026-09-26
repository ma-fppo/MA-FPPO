"""PyTorch continuous MAC-Flow: FM teacher, MSE student distillation and Q guidance.

Uses the official continuous defaults (512x4 MLP, no recurrent encoder).
Episode boundaries and critic masks prevent bootstrapping across dataset resets.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from common import grad_norm


def mlp(inputs, outputs, hidden=(512, 512, 512, 512), norm=False):
    layers = []
    for size in hidden:
        layer = nn.Linear(inputs, size); nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        layers += [layer, nn.GELU(approximate='tanh')]
        if norm: layers.append(nn.LayerNorm(size, eps=1e-6))
        inputs = size
    layer = nn.Linear(inputs, outputs); nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
    return nn.Sequential(*layers, layer)


class ContinuousMACFlow(nn.Module):
    def __init__(self, obs_dim, action_dim=4, hidden=(512, 512, 512, 512)):
        super().__init__()
        self.obs_dim = obs_dim; self.action_dim = action_dim
        self.teacher = mlp(obs_dim + action_dim + 1, action_dim, hidden)
        self.student = mlp(obs_dim + action_dim, action_dim, hidden)
        self.q = nn.ModuleList([mlp(obs_dim + action_dim, 1, hidden, norm=True) for _ in range(2)])
        self.target_q = copy.deepcopy(self.q).requires_grad_(False)

    def student_action(self, obs, noise=None):
        if noise is None: noise = torch.zeros_like(obs[..., :self.action_dim])
        return self.student(torch.cat((obs, noise), -1))

    @torch.no_grad()
    def teacher_action(self, obs, noise):
        action = noise
        for i in range(10):
            t = torch.full_like(action[..., :1], i / 10)
            action = action + self.teacher(torch.cat((obs, action, t), -1)) / 10
        return action.clamp(-1, 1)

    def values(self, obs, actions, target=False):
        return torch.stack([q(torch.cat((obs, actions), -1)).squeeze(-1) for q in (self.target_q if target else self.q)])

    def learn(self, batch, optim, discount=.995, alpha=1., tau=.005):
        obs, act, rew, nxt, terminal = [batch[k] for k in ('obs', 'actions', 'rewards', 'next_obs', 'terminals')]
        with torch.no_grad():
            next_actions = self.student_action(nxt, torch.randn_like(act)).clamp(-1, 1)
            target = (rew + discount * (1 - terminal) * self.values(nxt, next_actions, True).mean(0)).mean(-1)
        current = self.values(obs, act).mean(-1)
        critic_mask = batch['critic_mask']
        critic_loss = ((current-target.unsqueeze(0)).square()*critic_mask.unsqueeze(0)).sum() / (len(self.q)*critic_mask.sum().clamp_min(1))
        x0 = torch.randn_like(act); t = torch.rand_like(act[..., :1])
        teacher_pred = self.teacher(torch.cat((obs, (1 - t) * x0 + t * act, t), -1))
        fm = F.mse_loss(teacher_pred, act - x0)
        noise = torch.randn_like(act)
        target_action = self.teacher_action(obs, noise)
        student_action = self.student_action(obs, noise)
        distill = F.mse_loss(student_action, target_action)
        # Freeze critic weights for the actor objective, while retaining dQ/da.
        self.q.requires_grad_(False)
        actor_q = self.values(obs, student_action.clamp(-1, 1)).mean(0).mean(-1)
        self.q.requires_grad_(True)
        q_guidance = -actor_q.mean() / actor_q.detach().abs().mean().clamp_min(1e-6)
        loss = critic_loss + fm + alpha * distill + q_guidance
        optim.zero_grad(set_to_none=True); loss.backward()
        grads = {k: grad_norm(getattr(self, k)) for k in ('teacher', 'student', 'q')}
        if not all(math.isfinite(v) and v > 0 for v in grads.values()): raise FloatingPointError(grads)
        nn.utils.clip_grad_norm_([p for p in self.parameters() if p.requires_grad], 10.)
        optim.step()
        with torch.no_grad():
            for p, target_p in zip(self.q.parameters(), self.target_q.parameters()): target_p.mul_(1 - tau).add_(p, alpha=tau)
        return dict(loss=float(loss.detach()), critic_loss=float(critic_loss.detach()), bc_flow_loss=float(fm.detach()),
                    distill_loss=float(distill.detach()), q_guidance_loss=float(q_guidance.detach()), gradient_norms=grads)


class GaussianStudent(nn.Module):
    """Gaussian pre-clipping action policy; store the sampled raw action for PPO.

    Clipping is the deterministic action transform of the environment. Likelihoods
    are of the sampled Gaussian action, never of the clipped value.
    """
    def __init__(self, student, std=.2, action_dim=4):
        super().__init__(); self.student = copy.deepcopy(student); self.action_dim = action_dim
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(std)))

    def forward(self, obs):
        mean = self.student(torch.cat((obs, torch.zeros_like(obs[..., :self.action_dim])), -1))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))


def gae(rewards, values, next_values, done, gamma=.99, lam=.95):
    advantage = torch.zeros_like(rewards); carry = torch.zeros_like(rewards[0])
    # Time limits bootstrap the final observation but cut GAE across the reset.
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] - values[t]
        carry = delta + gamma * lam * (1 - done[t]) * carry
        advantage[t] = carry
    return advantage, advantage + values
