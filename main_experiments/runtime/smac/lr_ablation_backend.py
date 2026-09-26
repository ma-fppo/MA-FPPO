"""Explicit, audited learning-rate forks; the original PPO backend stays frozen."""
import copy
from pathlib import Path

import numpy as np
import torch

from common import atomic_json, sha256
from torch_backend import TorchBackend


def exact(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and set(left) == set(right) and all(exact(v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


class LearningRateBackend(TorchBackend):
    def restore(self, path):
        fork = self.cfg.get('learning_rate_fork')
        if not fork or Path(path).resolve() != Path(fork['checkpoint']).resolve():
            return super().restore(path)
        if self.variant != 'torch' or sha256(path) != fork['checkpoint_sha256']:
            raise ValueError('Only the fingerprinted Torch source can start this learning-rate fork')
        source = torch.load(path, map_location=self.device)
        requested = float(self.cfg['actor_lr'])
        previous = float(source['config']['actor_lr'])
        if previous != fork['source_actor_lr'] or requested != fork['requested_actor_lr']:
            raise ValueError('Learning-rate fork does not match its declared rates')
        if self.cfg['critic_lr'] != source['config']['critic_lr']:
            raise ValueError('This experiment cannot change the critic learning rate')
        self.cfg['actor_lr'] = previous
        try:
            counters = super().restore(path)
        finally:
            self.cfg['actor_lr'] = requested
        checks = {name: exact(getattr(self, name).state_dict(), source[name])
                  for name in ['model', 'actor', 'critic', 'reference']}
        checks.update(actor_optimizer_before_override=exact(self.actor_opt.state_dict(), source['actor_optimizer']),
                      critic_optimizer=exact(self.critic_opt.state_dict(), source['critic_optimizer']),
                      torch_rng=exact(torch.get_rng_state(), source['torch_rng']),
                      numpy_rng=exact(np.random.get_state(), source['numpy_rng']),
                      counters=counters == (source['iteration'], source['env_steps']))
        if self.device.type == 'cuda':
            checks['cuda_rng'] = exact(torch.cuda.get_rng_state_all(), source['cuda_rng'])
        if not all(checks.values()):
            raise AssertionError('Source state restoration failed: ' + str(checks))
        for group in self.actor_opt.param_groups:
            group['lr'] = requested
        expected_optimizer = copy.deepcopy(source['actor_optimizer'])
        for group in expected_optimizer['param_groups']:
            group['lr'] = requested
        checks['actor_optimizer_only_lr_changed'] = exact(self.actor_opt.state_dict(), expected_optimizer)
        checks['effective_actor_lr'] = all(g['lr'] == requested for g in self.actor_opt.param_groups)
        if not all(checks.values()):
            raise AssertionError('Learning-rate override changed unexpected state')
        audit = dict(status='verified', checkpoint=str(path), checkpoint_sha256=fork['checkpoint_sha256'],
                     source_actor_lr=previous, requested_actor_lr=requested,
                     effective_actor_lr=[g['lr'] for g in self.actor_opt.param_groups],
                     initial_env_steps=counters[1], initial_iteration=counters[0],
                     checks=checks, cuda_rng_checked=self.device.type == 'cuda')
        atomic_json(Path(self.cfg['output_dir']) / 'initial_resume_audit.json', audit)
        return counters

    def update(self, batch, actor_update=True):
        if any(g['lr'] != self.cfg['actor_lr'] for g in self.actor_opt.param_groups):
            raise AssertionError('Effective actor learning rate differs from the experiment config')
        return super().update(batch, actor_update)

    @torch.no_grad()
    def policy_snapshot(self, batch):
        """Log distributions on the collected features, without RNG or optimizer changes."""
        result = []
        size = self.cfg['batch_size']
        for start in range(0, len(batch['features']), size):
            features = self.tensor(batch['features'][start:start+size], self.policy_dtype)
            legal = self.tensor(batch['legal'][start:start+size], torch.bool)
            result.append(self.categorical(self.actor, features, legal).logits.cpu())
        return torch.cat(result)

    @torch.no_grad()
    def post_update_diagnostics(self, batch, old_logits):
        new_logits = self.policy_snapshot(batch)
        mask = torch.as_tensor(batch['legal']).sum(-1) > 1
        old_probs = old_logits.exp()
        exact_kl = (old_probs * (old_logits - new_logits)).sum(-1)
        actions = torch.as_tensor(batch['actions'], dtype=torch.long).unsqueeze(-1)
        new_lp = new_logits.gather(-1, actions).squeeze(-1)
        log_ratio = new_lp - torch.as_tensor(batch['old_lp'])
        ratio = log_ratio.exp()
        if not bool(mask.any()) or not bool(torch.isfinite(exact_kl).all()) or not bool(torch.isfinite(ratio).all()):
            raise FloatingPointError('Invalid post-update policy diagnostics')
        return dict(post_update_kl=float(exact_kl[mask].mean()),
                    post_update_kl_max=float(exact_kl[mask].max()),
                    post_update_sample_kl=float(((ratio-1)-log_ratio)[mask].mean()),
                    post_update_clip_fraction=float(((ratio-1).abs() > self.cfg['clip_coef'])[mask].float().mean()),
                    post_update_log_ratio_abs_max=float(log_ratio[mask].abs().max()),
                    effective_actor_lr=self.actor_opt.param_groups[0]['lr'])
