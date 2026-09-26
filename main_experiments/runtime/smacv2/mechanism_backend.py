"""Isolated latent-input / recurrent-encoder PPO ablations for the Torch student.

The latent arm optimizes the augmented policy p(z) pi(a|history,z); p(z) is fixed,
so its density cancels in the PPO ratio. Logged KL is conditional on stored z,
not the intractable marginal action KL. Deployment retains z=0 and greedy actions.

The encoder arm uses overlapping 16-step truncated histories with stored initial
hidden states, keeping the original shuffled target-transition minibatches. Loss
is applied only at each history's last transition. The reference encoder stays
frozen and receives the actual observation history, independently of the actor.
"""
import copy
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from anchor_ablation_backend import AnchorBackend
from common import finite_metrics
from torch_backend import masked_categorical, ppo_terms


def finite_categorical_kl(reference, current):
    """KL from finite log-softmax values, without treating exp underflow as zero support."""
    return (reference.probs * (reference.logits - current.logits)).sum(-1)


class MechanismBackend(AnchorBackend):
    def __init__(self, cfg, info):
        super().__init__(cfg, info)
        self.encoder_training = bool(cfg.get('train_encoder', False))
        self.latent_std = float(cfg.get('student_latent_std', 0.))
        if self.variant != 'torch' or self.latent_std < 0:
            raise ValueError('This ablation requires the Torch student and nonnegative noise')
        if self.encoder_training and self.latent_std:
            raise ValueError('The first mechanism experiment must isolate encoder and noise')
        self.reference_encoder = copy.deepcopy(self.model.encoder).requires_grad_(False).eval()
        self.reference_encoder.lstm.flatten_parameters()
        self.model.encoder.requires_grad_(self.encoder_training)
        self.encoder_opt = (torch.optim.Adam(self.model.encoder.parameters(), lr=cfg['encoder_lr'], eps=1e-5)
                            if self.encoder_training else None)
        self.reference_carry = None

    def reset(self):
        super().reset()
        self.reference_carry = None

    def extra_runtime_state(self):
        return self.reference_carry

    def restore_extra_runtime_state(self, state):
        self.reference_carry = state

    def _augmented_obs(self, obs):
        obs = self.tensor(obs, self.policy_dtype)
        ids = torch.eye(self.n_agents, device=self.device).expand(obs.shape[0], -1, -1)
        return torch.cat([obs, ids], -1)

    def _empty_carry(self, batch):
        enc = self.model.encoder
        shape = (enc.lstm.num_layers, batch * self.n_agents, enc.hidden_dim)
        return tuple(torch.zeros(shape, device=self.device, dtype=self.policy_dtype) for _ in range(2))

    @torch.no_grad()
    def act_batch(self, obs, legal, resets=None, deterministic=False, rngs=None):
        batch_size = len(obs)
        # Capture the unmasked initial state; masks are replayed at every sequence step.
        previous = self.carry if self.carry is not None else self._empty_carry(batch_size)
        features = self.features_batch(obs, legal, resets)
        masks = self.tensor(legal, torch.bool)
        z = torch.zeros(batch_size, self.n_agents, self.action_dim, device=self.device)
        if not deterministic and self.latent_std:
            if rngs is None:
                z = torch.randn_like(z) * self.latent_std
            else:
                z = torch.stack([torch.randn(self.n_agents, self.action_dim, device=self.device,
                                             generator=g) for g in rngs]) * self.latent_std
        dist = masked_categorical(self.actor(features, z), masks, self.cfg['temperature'])
        if deterministic:
            actions = dist.logits.argmax(-1)
        elif rngs is None:
            actions = dist.sample()
        else:
            actions = torch.stack([torch.multinomial(dist.probs[i], 1, replacement=True, generator=g)[:, 0]
                                   for i, g in enumerate(rngs)])
        cache = dict(features=features, legal=masks, actions=actions, old_lp=dist.log_prob(actions), latent=z)
        if self.encoder_training:
            augmented = self._augmented_obs(obs)
            active = ~masks[:, :, 0]
            if resets is not None:
                active = active & ~self.tensor(resets, torch.bool)[:, None]
            ref_initial = self.reference_carry if self.reference_carry is not None else self._empty_carry(batch_size)
            gate = active.reshape(1, -1, 1)
            ref_initial = tuple(x * gate for x in ref_initial)
            x = self.reference_encoder.pre(augmented.reshape(batch_size * self.n_agents, -1))[:, None]
            ref_x, self.reference_carry = self.reference_encoder.lstm(x, ref_initial)
            ref_features = self.reference_encoder.norm(ref_x[:, -1]).reshape(batch_size, self.n_agents, -1)
            h, c = [x.reshape(x.shape[0], batch_size, self.n_agents, -1).permute(1, 0, 2, 3) for x in previous]
            cache.update(raw_obs=augmented, seq_active=active, initial_h=h, initial_c=c,
                         reference_features=ref_features)
        return actions.cpu().numpy(), {k: v.cpu().numpy() for k, v in cache.items()}

    def prepare_batch(self, records):
        batch = {k: np.concatenate([r[k] for r in records], axis=0) for k in records[0]}
        if not self.encoder_training:
            return batch
        length = self.cfg['encoder_sequence_length']
        count, envs = len(records), len(records[0]['features'])
        obs = np.stack([r['raw_obs'] for r in records])
        active = np.stack([r['seq_active'] for r in records])
        seq_obs = np.zeros((count, envs, length, *obs.shape[2:]), dtype=obs.dtype)
        seq_active = np.zeros((count, envs, length, self.n_agents), dtype=bool)
        valid = np.zeros((count, envs, length), dtype=bool)
        hs, cs = [], []
        for t in range(count):
            start = max(0, t - length + 1)
            used = t - start + 1
            seq_obs[t, :, -used:] = obs[start:t+1].swapaxes(0, 1)
            seq_active[t, :, -used:] = active[start:t+1].swapaxes(0, 1)
            valid[t, :, -used:] = True
            hs.append(records[start]['initial_h']); cs.append(records[start]['initial_c'])
        batch.update(sequence_obs=seq_obs.reshape(count * envs, *seq_obs.shape[2:]),
                     sequence_active=seq_active.reshape(count * envs, length, self.n_agents),
                     sequence_valid=valid.reshape(count * envs, length),
                     sequence_h=np.concatenate(hs), sequence_c=np.concatenate(cs))
        return batch

    def replay_features(self, batch, return_carry=False):
        if not self.encoder_training:
            return self.tensor(batch['features'], self.policy_dtype)
        obs = self.tensor(batch['sequence_obs'], self.policy_dtype)
        valid = self.tensor(batch['sequence_valid'], torch.bool)
        active = self.tensor(batch['sequence_active'], torch.bool)
        b, length, n, _ = obs.shape
        carry = tuple(self.tensor(batch[k], self.policy_dtype).permute(1, 0, 2, 3).reshape(
            self.model.encoder.lstm.num_layers, b * n, -1) for k in ['sequence_h', 'sequence_c'])
        enc = self.model.encoder
        # Native LSTM supports backward while the policy stays in deterministic eval mode.
        with torch.backends.cudnn.flags(enabled=False):
            for t in range(length):
                gate = active[:, t].reshape(1, b*n, 1)
                real = valid[:, t, None].expand(b, n).reshape(1, b*n, 1)
                x = enc.pre(obs[:, t].reshape(b*n, -1))[:, None]
                _, advanced = enc.lstm(x, tuple(c * gate for c in carry))
                carry = tuple(torch.where(real, new, old) for new, old in zip(advanced, carry))
        features = enc.norm(carry[0][-1]).reshape(b, n, -1)
        return (features, carry) if return_carry else features

    def batch_distribution(self, batch, reference=False):
        if reference:
            f = self.tensor(batch['reference_features'] if self.encoder_training else batch['features'])
            actor = self.reference
        else:
            f, actor = self.replay_features(batch), self.actor
        return masked_categorical(actor(f, self.tensor(batch['latent'])),
                                  self.tensor(batch['legal'], torch.bool), self.cfg['temperature'])

    def likelihood(self, batch):
        dist = self.batch_distribution(batch)
        legal = self.tensor(batch['legal'], torch.bool)
        mask = legal.sum(-1) > 1
        lp = dist.log_prob(self.tensor(batch['actions'], torch.long))
        entropy = (dist.entropy() * mask).sum() / mask.sum().clamp_min(1)
        return lp, entropy, mask

    def trainable_parameters(self):
        return tuple(self.actor.parameters()) + (tuple(self.model.encoder.parameters()) if self.encoder_training else ())

    def objective_terms(self, batch):
        dist = self.batch_distribution(batch)
        mask = self.tensor(batch['legal'], torch.bool).sum(-1) > 1
        lp = dist.log_prob(self.tensor(batch['actions'], torch.long))
        entropy = (dist.entropy() * mask).sum() / mask.sum().clamp_min(1)
        pg, kl, cf = ppo_terms(lp, self.tensor(batch['old_lp']), self.tensor(batch['advantages'])[:, None],
                               mask, self.cfg['clip_coef'])
        with torch.no_grad():
            reference = self.batch_distribution(batch, reference=True)
        self.last_anchor_underflow_count = int(((dist.probs == 0) & (reference.probs > 0) &
                                                self.tensor(batch['legal'], torch.bool)).sum())
        anchor = (finite_categorical_kl(reference, dist) * mask).sum() / mask.sum().clamp_min(1)
        return pg, entropy, kl, cf, anchor

    def update(self, batch, actor_update=True):
        if self.actor_opt.param_groups[0]['lr'] != self.cfg['actor_lr']:
            raise AssertionError('Actor LR changed')
        values, targets = self.critic(self.tensor(batch['state'])), self.tensor(batch['returns'])
        v_loss = .5 * (values - targets).square().mean()
        self.critic_opt.zero_grad(set_to_none=True); v_loss.backward()
        cg = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg['max_grad_norm'])
        if not bool(torch.isfinite(cg)): raise FloatingPointError('Critic gradient')
        self.critic_opt.step()
        result = dict(value_loss=float(v_loss.detach()), critic_grad_norm=float(cg), policy_loss=0.,
            approx_kl=0., clip_fraction=0., actor_grad_norm=0., encoder_grad_norm=0., anchor_kl=0.,
            anchor_underflow_actions=0., entropy=0., actor_updated=0.)
        if not actor_update: return finite_metrics(result)
        pg, entropy, kl, cf, anchor = self.objective_terms(batch)
        result.update(policy_loss=float(pg.detach()), approx_kl=float(kl.detach()), clip_fraction=float(cf),
                      anchor_kl=float(anchor.detach()), entropy=float(entropy.detach()),
                      anchor_underflow_actions=float(self.last_anchor_underflow_count))
        if float(kl.detach()) > self.cfg['target_kl']: return finite_metrics(result)
        loss = pg - self.cfg['entropy_coef'] * entropy + self.cfg['anchor_coef'] * anchor
        self.actor_opt.zero_grad(set_to_none=True)
        if self.encoder_opt: self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.encoder_opt:
            result['encoder_grad_norm'] = float(torch.sqrt(sum(p.grad.square().sum() for p in self.model.encoder.parameters() if p.grad is not None)))
        ag = nn.utils.clip_grad_norm_(self.trainable_parameters(), self.cfg['max_grad_norm'])
        if not bool(torch.isfinite(ag)): raise FloatingPointError('Actor/encoder gradient')
        self.actor_opt.step()
        if self.encoder_opt: self.encoder_opt.step()
        result.update(actor_grad_norm=float(ag), actor_updated=1.)
        return finite_metrics(result)

    def objective_diagnostics(self, batch):
        item = {k: v[:self.cfg['batch_size']] for k, v in batch.items()}
        pg, _, _, _, anchor = self.objective_terms(item)
        params = self.trainable_parameters()
        def grad(loss, retain):
            gs = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
            return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, gs)])
        p, a = grad(pg, True), grad(anchor, False)
        pn, an = p.norm(), a.norm()
        target = batch['returns']; prediction = self.values_batch(batch['state'])
        return dict(anchor_coef_effective=self.cfg['anchor_coef'], diagnostic_pg_grad_norm=float(pn),
            diagnostic_anchor_grad_norm=float(an), diagnostic_weighted_anchor_grad_norm=float(an)*self.cfg['anchor_coef'],
            diagnostic_anchor_pg_cosine=float(torch.dot(p,a)/(pn*an).clamp_min(1e-20)),
            diagnostic_anchor_pg_ratio=float(an/pn.clamp_min(1e-20))*self.cfg['anchor_coef'],
            critic_explained_variance=1-float(np.var(target-prediction))/max(float(np.var(target)),1e-12),
            critic_target_variance=float(np.var(target)), latent_empirical_std=float(np.std(batch['latent'])))

    @torch.no_grad()
    def policy_snapshot(self, batch):
        size = self.cfg['batch_size']
        return torch.cat([self.batch_distribution({k:v[start:start+size] for k,v in batch.items()}).logits.cpu()
                          for start in range(0,len(batch['features']),size)])

    @torch.no_grad()
    def refresh_carry(self, batch, envs):
        if self.encoder_training:
            # Update the carried actor state over the latest truncated window after optimization.
            _, self.carry = self.replay_features({k:v[-envs:] for k,v in batch.items()}, return_carry=True)

    def save(self, path, iteration, env_steps):
        super().save(path, iteration, env_steps)
        payload = torch.load(path, map_location=self.device)
        payload.update(encoder_optimizer=self.encoder_opt.state_dict() if self.encoder_opt else None,
                       reference_encoder=self.reference_encoder.state_dict(), mechanism_format=1)
        tmp = str(path) + '.tmp'; torch.save(payload, tmp); os.replace(tmp, path)

    def restore(self, path):
        p = torch.load(path, map_location=self.device)
        if p.get('mechanism_format'):
            for key in ['train_encoder', 'student_latent_std', 'encoder_lr', 'encoder_sequence_length', 'anchor_schedule']:
                if p['config'].get(key) != self.cfg.get(key): raise ValueError('Mechanism resume mismatch: ' + key)
        counters = super().restore(path)
        if p.get('mechanism_format'):
            self.reference_encoder.load_state_dict(p['reference_encoder'])
            if self.encoder_opt: self.encoder_opt.load_state_dict(p['encoder_optimizer'])
        elif self.encoder_training:
            # New encoder optimizer: these parameters had no optimizer state in the frozen source.
            if Path(path).resolve() != Path(self.cfg['learning_rate_fork']['checkpoint']).resolve():
                raise ValueError('Unrecognized legacy source for encoder fork')
        return counters
