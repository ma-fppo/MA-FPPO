"""Checkpoint-preserving PPO adapters, with explicit sampled path likelihoods.

Independent one-step MAC-Flow: masked categorical MAPPO-style surrogate.
Coupled DiT/Graph: ReinFlow-style joint latent-path PPO. This density is NOT
the marginal probability of the discrete argmax action. Argmax/masking is a
fixed environment-facing map of the sampled latent chain.
"""
from __future__ import annotations

import copy
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal, kl_divergence

from common import finite_metrics


def masked_categorical(logits, legal, temperature=1.0):
    if temperature <= 0 or not bool(legal.any(-1).all()):
        raise ValueError('Positive temperature and nonempty legal masks required')
    return Categorical(logits=(logits / temperature).masked_fill(~legal, -1e9), validate_args=False)


def ppo_terms(new_lp, old_lp, adv, mask, clip):
    log_ratio = new_lp - old_lp
    if not bool(torch.isfinite(log_ratio).all()) or float(log_ratio.abs().max()) > 60:
        raise FloatingPointError('PPO likelihood ratio diverged; aborting update')
    ratio = log_ratio.exp()
    weights = mask.to(ratio.dtype)
    denom = weights.sum().clamp_min(1)
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1-clip, 1+clip) * adv)
    loss = -(surrogate * weights).sum() / denom
    kl = (((ratio - 1) - log_ratio) * weights).sum() / denom
    clipfrac = (((ratio - 1).abs() > clip).float() * weights).sum() / denom
    return loss, kl, clipfrac


class CentralValue(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.Tanh(),
                                 nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state):
        return self.net(state).squeeze(-1)


class FlowPathPolicy(nn.Module):
    def __init__(self, model, variant, steps=8, std=.1):
        super().__init__()
        if steps <= 0 or not .01 < std < .5:
            raise ValueError('Invalid flow steps/std')
        self.model, self.variant, self.steps = model, variant, steps
        self.horizon = model.action_horizon if variant == 'dit' else 1
        # Bounded, learned per-step exploration scale. Same value in sampling
        # and replay; no clipping of Gaussian draws or of log densities.
        initial = math.log((std - .01) / (.5 - std))
        self.noise_raw = nn.Parameter(torch.full((steps,), initial))

    def std(self, k):
        return .01 + .49 * torch.sigmoid(self.noise_raw[k])

    def mean(self, obs, x, k):
        t = (k + .5) / self.steps if self.variant == 'graph' else k / self.steps
        times = torch.full((obs.shape[0],), t, device=obs.device, dtype=obs.dtype)
        velocity = self.model(x, obs, times) if self.variant == 'dit' else self.model(obs, x[:, 0], times)[:, None]
        return x + velocity / self.steps

    @torch.no_grad()
    def sample(self, obs, rngs=None):
        b, n, _ = obs.shape
        shape = (self.horizon, n, self.model.action_dim)
        x = torch.randn(b, *shape, device=obs.device, dtype=obs.dtype) if rngs is None else torch.stack([
            torch.randn(shape, device=obs.device, dtype=obs.dtype, generator=g) for g in rngs])
        chain, lp, entropy = [x], [], []
        for k in range(self.steps):
            dist = Normal(self.mean(obs, x, k), self.std(k), validate_args=False)
            x = dist.sample() if rngs is None else torch.stack([
                torch.normal(dist.loc[i], dist.scale[i], generator=g) for i,g in enumerate(rngs)])
            chain.append(x)
            # Reuse the sampling mean instead of running the velocity model twice.
            lp.append(dist.log_prob(x).flatten(1).sum(-1))
            entropy.append(dist.entropy().flatten(1).sum(-1))
        # Keep EVERY coordinate and horizon slot: cross-agent/time coupling
        # means apparently unexecuted coordinates can affect later actions.
        chain = torch.stack(chain, 1)
        return chain, torch.stack(lp, 1).sum(1), torch.stack(entropy, 1).sum(1)

    def log_prob(self, obs, chain):
        # Parameter-independent p(x0) cancels from old/new ratios.
        chain = chain.detach()
        lp, entropy = [], []
        for k in range(self.steps):
            dist = Normal(self.mean(obs, chain[:, k], k), self.std(k), validate_args=False)
            lp.append(dist.log_prob(chain[:, k+1]).flatten(1).sum(-1))
            entropy.append(dist.entropy().flatten(1).sum(-1))
        return torch.stack(lp, 1).sum(1), torch.stack(entropy, 1).sum(1)

    def reference_kl(self, reference, obs, chain):
        terms = []
        for k in range(self.steps):
            with torch.no_grad():
                old = Normal(reference.mean(obs, chain[:, k], k), reference.std(k), validate_args=False)
            new = Normal(self.mean(obs, chain[:, k], k), self.std(k), validate_args=False)
            terms.append(kl_divergence(old, new).flatten(1).sum(-1))
        return torch.stack(terms, 1).sum(1).mean()


class TorchBackend:
    def __init__(self, cfg, info):
        self.cfg = cfg
        self.device = torch.device(cfg.get('device', 'cuda'))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(int(cfg.get('torch_threads', 2)))
        torch.manual_seed(cfg['seed'])
        root = Path(cfg['project_root'])
        for name in ['mac_flow_pytorch_smacv2', 'mac_flow_torch_transformer_smacv2', 'mac_flow_torch_graph_smacv2']:
            sys.path.insert(0, str(root / name))
        self.variant = cfg['variant']
        self.n_agents, self.action_dim = info['n_agents'], info['n_actions']
        payload = torch.load(cfg['checkpoint'], map_location=self.device)
        self.normalizer = None
        if self.variant == 'torch':
            from macflow_torch.models import MACFlowConfig, MACFlowTorch
            self.model = MACFlowTorch(MACFlowConfig(**payload['config'])).to(self.device)
            self.model.load_state_dict(payload['model'], strict=True)
            self.model.requires_grad_(False)
            self.actor = self.model.actor_onestep_flow
            self.actor.requires_grad_(True)
            self.input_dim = self.model.cfg.obs_dim
        elif self.variant == 'dit':
            from macflow_torch_transformer.model import AgentTimeFlowTransformer
            c = payload['config']
            self.normalizer = payload.get('normalizer')
            self.input_dim = int(payload['model']['obs_proj.1.weight'].shape[1])
            self.model = AgentTimeFlowTransformer(self.input_dim, self.action_dim, self.n_agents,
                **{k: c[k] for k in ['action_horizon', 'model_dim', 'depth', 'n_heads', 'mlp_ratio', 'dropout'] if k in c}).to(self.device)
            self.model.load_state_dict(payload['model'], strict=True)
            self.actor = FlowPathPolicy(self.model, 'dit', cfg['flow_steps'], cfg['flow_std']).to(self.device)
        elif self.variant == 'graph':
            from graph_flow_smacv2.model import GraphActionFlow
            meta = payload['meta']; args, data = meta['args'], meta['dataset']
            self.input_dim = int(data['obs_dim'])
            self.model = GraphActionFlow(self.input_dim, self.action_dim, self.n_agents,
                hidden_dim=args.get('hidden_dim', 192), edge_dim=args.get('edge_dim', 128),
                n_layers=args.get('layers', 4)).to(self.device)
            self.model.load_state_dict(payload['model'], strict=True)
            self.actor = FlowPathPolicy(self.model, 'graph', cfg['flow_steps'], cfg['flow_std']).to(self.device)
        else:
            raise ValueError(self.variant)
        if self.input_dim != info['obs_shape'] + self.n_agents:
            raise ValueError('Checkpoint obs dimensions do not match environment + agent IDs')
        # The pretrained graph has very large intermediate gate activations.
        # FP32 GEMM rounding is amplified by successive message-passing blocks.
        # FP64 keeps its sampled path density independent of minibatch shape.
        self.policy_dtype = torch.float64 if self.variant=='graph' and cfg.get('graph_policy_dtype', 'float64')=='float64' else torch.float32
        if self.policy_dtype == torch.float64:
            self.actor.double()
            self.cfg['graph_policy_dtype'] = 'float64'
        self.model.eval(); self.actor.eval()  # deterministic dropout behavior in PPO replay
        self.reference = copy.deepcopy(self.actor).requires_grad_(False).eval()
        self.critic = CentralValue(info['state_shape']).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg['actor_lr'], eps=1e-5)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg['critic_lr'], eps=1e-5)
        self.carry = None

    def tensor(self, a, dtype=torch.float32):
        return torch.as_tensor(a, device=self.device, dtype=dtype)

    def reset(self):
        self.carry = None

    def actor_vector(self):
        return np.concatenate([p.detach().cpu().numpy().ravel().copy() for p in self.actor.parameters()])

    def rng_state(self):
        return torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def restore_rng(self, state):
        torch.set_rng_state(state[0])
        if state[1] is not None: torch.cuda.set_rng_state_all(state[1])

    def seed(self, seed):
        torch.manual_seed(seed)

    def make_rngs(self, seeds):
        return [torch.Generator(device=self.device).manual_seed(int(seed)) for seed in seeds]

    @torch.no_grad()
    def features(self, obs, legal):
        return self.features_batch(np.asarray(obs)[None], np.asarray(legal)[None])

    @torch.no_grad()
    def features_batch(self, obs, legal, resets=None):
        obs = self.tensor(obs, self.policy_dtype)
        b = obs.shape[0]
        ids = torch.eye(self.n_agents, device=self.device).expand(b, -1, -1)
        obs = torch.cat([obs, ids], -1)
        if self.normalizer is not None:
            obs = (obs - self.tensor(self.normalizer['obs_mean'])) / self.tensor(self.normalizer['obs_std']).clamp_min(1e-6)
        if self.variant != 'torch':
            return obs
        if self.cfg.get('torch_history', 'recurrent') == 'stateless':
            return self.model.encoder(obs[:, None])[:, -1]
        encoder = self.model.encoder
        if self.carry is not None:
            # Dead SMAC units are identifiable by their sole no-op action.
            active = ~np.asarray(legal)[:, :, 0]
            if resets is not None:
                active = active & ~np.asarray(resets)[:, None]
            active = self.tensor(active.reshape(-1), torch.bool)[None, :, None]
            self.carry = tuple(c * active for c in self.carry)
        x = encoder.pre(obs.reshape(b*self.n_agents, -1))[:, None, :]
        x, self.carry = encoder.lstm(x, self.carry)
        return encoder.norm(x[:, -1]).reshape(b, self.n_agents, -1)

    def categorical(self, actor, features, legal):
        noise = torch.zeros(*features.shape[:-1], self.action_dim, device=self.device)
        return masked_categorical(actor(features, noise), legal, self.cfg['temperature'])

    @torch.no_grad()
    def act(self, obs, legal, deterministic=False):
        actions, cache = self.act_batch(np.asarray(obs)[None], np.asarray(legal)[None],
                                        deterministic=deterministic)
        return actions[0], {k: v[0] for k, v in cache.items()}

    @torch.no_grad()
    def act_batch(self, obs, legal, resets=None, deterministic=False, rngs=None):
        features = self.features_batch(obs, legal, resets)
        masks = self.tensor(legal, torch.bool)
        if self.variant == 'torch':
            dist = self.categorical(self.actor, features, masks)
            if deterministic:
                actions = dist.logits.argmax(-1)
            elif rngs is None:
                actions = dist.sample()
            else:
                actions = torch.stack([torch.multinomial(dist.probs[i], 1, replacement=True, generator=g)[:, 0]
                                       for i,g in enumerate(rngs)])
            cache = {'features': features, 'legal': masks, 'actions': actions,
                     'old_lp': dist.log_prob(actions)}
        else:
            if deterministic:
                # Preserve pretrained inference, including its random initial x0.
                if rngs is not None:
                    x = torch.stack([torch.randn(self.actor.horizon, self.n_agents, self.action_dim,
                                                  device=self.device, dtype=self.policy_dtype, generator=g) for g in rngs])
                    for k in range(self.actor.steps):
                        x = self.actor.mean(features, x, k)
                    scores = x[:, 0]
                elif self.variant == 'dit':
                    scores = self.model.sample(features, steps=self.cfg['flow_steps'])[:, 0]
                elif self.policy_dtype == torch.float64:
                    x = torch.randn(features.shape[0], 1, self.n_agents, self.action_dim,
                                    device=self.device, dtype=self.policy_dtype)
                    for k in range(self.actor.steps): x = self.actor.mean(features, x, k)
                    scores = x[:, 0]
                else:
                    scores = self.model.sample_logits(features, steps=self.cfg['flow_steps'])
                actions = scores.masked_fill(~masks, -1e9).argmax(-1)
                return actions.cpu().numpy(), {}
            chain, lp, _ = self.actor.sample(features, rngs)
            actions = chain[:, -1, 0].masked_fill(~masks, -1e9).argmax(-1)
            cache = {'features': features, 'legal': masks, 'actions': actions,
                     'chain': chain, 'old_lp': lp}
        return actions.cpu().numpy(), {k: v.cpu().numpy() for k, v in cache.items()}

    @torch.no_grad()
    def value(self, state):
        return float(self.values_batch(np.asarray(state)[None])[0])

    @torch.no_grad()
    def values_batch(self, states):
        return self.critic(self.tensor(states)).cpu().numpy()

    def likelihood(self, batch):
        f = self.tensor(batch['features'], self.policy_dtype); legal = self.tensor(batch['legal'], torch.bool)
        if self.variant == 'torch':
            dist = self.categorical(self.actor, f, legal)
            lp = dist.log_prob(self.tensor(batch['actions'], torch.long))
            mask = legal.sum(-1) > 1
            entropy = (dist.entropy() * mask).sum() / mask.sum().clamp_min(1)
        else:
            lp, ent = self.actor.log_prob(f, self.tensor(batch['chain'], self.policy_dtype))
            mask = torch.ones_like(lp, dtype=torch.bool)
            # Scale entropy bonus per latent coordinate; keep likelihood SUMS exact.
            entropy = ent.mean() / (self.actor.steps * self.actor.horizon * self.n_agents * self.action_dim)
        return lp, entropy, mask

    @torch.no_grad()
    def replay_error(self, batch):
        return float(np.max(np.abs(self.log_probs_numpy(batch)-batch['old_lp'])))

    @torch.no_grad()
    def log_probs_numpy(self, batch):
        size = self.cfg.get('batch_size', 128)
        result = []
        for start in range(0, len(batch['features']), size):
            lp, _, _ = self.likelihood({k: v[start:start+size] for k, v in batch.items()})
            result.append(lp.cpu().numpy())
        return np.concatenate(result)

    def update(self, batch, actor_update=True):
        values = self.critic(self.tensor(batch['state']))
        targets = self.tensor(batch['returns'])
        v_loss = .5 * (values - targets).square().mean()
        self.critic_opt.zero_grad(set_to_none=True)
        v_loss.backward()
        cg = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg['max_grad_norm'])
        if not bool(torch.isfinite(cg)):
            raise FloatingPointError('Non-finite critic gradient')
        self.critic_opt.step()
        result = {'value_loss': float(v_loss.detach()), 'critic_grad_norm': float(cg),
                  'policy_loss': 0., 'approx_kl': 0., 'clip_fraction': 0., 'actor_grad_norm': 0.,
                  'anchor_kl': 0., 'entropy': 0., 'actor_updated': 0.}
        if not actor_update:
            return finite_metrics(result)
        lp, entropy, mask = self.likelihood(batch)
        adv = self.tensor(batch['advantages'])
        if lp.ndim == 2:
            adv = adv[:, None]
        pg, kl, cf = ppo_terms(lp, self.tensor(batch['old_lp'], lp.dtype), adv, mask, self.cfg['clip_coef'])
        result.update(policy_loss=float(pg.detach()), approx_kl=float(kl.detach()),
                      clip_fraction=float(cf), entropy=float(entropy.detach()))
        if float(kl.detach()) > self.cfg['target_kl']:
            return finite_metrics(result)
        f = self.tensor(batch['features'], self.policy_dtype)
        if self.variant == 'torch':
            legal = self.tensor(batch['legal'], torch.bool)
            with torch.no_grad():
                reference = self.categorical(self.reference, f, legal)
            anchor = kl_divergence(reference, self.categorical(self.actor, f, legal))
            anchor = (anchor * mask).sum() / mask.sum().clamp_min(1)
        else:
            anchor = self.actor.reference_kl(self.reference, f, self.tensor(batch['chain'], self.policy_dtype))
        loss = pg - self.cfg['entropy_coef'] * entropy + self.cfg['anchor_coef'] * anchor
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        ag = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg['max_grad_norm'])
        if not bool(torch.isfinite(ag)):
            raise FloatingPointError('Non-finite actor gradient')
        self.actor_opt.step()
        result.update(actor_grad_norm=float(ag), anchor_kl=float(anchor.detach()), actor_updated=1.)
        return finite_metrics(result)

    def save(self, path, iteration, env_steps):
        payload = {'format': 'macflow_ppo_v1', 'config': self.cfg, 'iteration': iteration,
                   'env_steps': env_steps, 'actor': self.actor.state_dict(),
                   'critic': self.critic.state_dict(), 'reference': self.reference.state_dict(),
                   'actor_optimizer': self.actor_opt.state_dict(), 'critic_optimizer': self.critic_opt.state_dict(),
                   'model': self.model.state_dict(), 'normalizer': self.normalizer,
                   'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
                   'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        tmp = str(path) + '.tmp'; torch.save(payload, tmp); os.replace(tmp, str(path))

    def restore(self, path):
        p = torch.load(path, map_location=self.device)
        if p.get('format') != 'macflow_ppo_v1':
            raise ValueError('Expected a PPO checkpoint')
        for name in ['variant', 'flow_steps', 'temperature', 'torch_history', 'checkpoint',
                     'source_checkpoint_sha256', 'actor_lr', 'critic_lr']:
            if p['config'].get(name) != self.cfg.get(name):
                raise ValueError('Resume configuration mismatch: ' + name)
        self.model.load_state_dict(p['model']); self.actor.load_state_dict(p['actor'])
        self.reference.load_state_dict(p['reference']); self.critic.load_state_dict(p['critic'])
        self.actor_opt.load_state_dict(p['actor_optimizer']); self.critic_opt.load_state_dict(p['critic_optimizer'])
        torch.set_rng_state(p['torch_rng'].cpu()); np.random.set_state(p['numpy_rng'])
        if p['cuda_rng'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in p['cuda_rng']])
        self.reset()
        return p['iteration'], p['env_steps']
