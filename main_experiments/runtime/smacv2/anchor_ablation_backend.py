"""Pretrained-reference ablations with an environment-step schedule and diagnostics."""
import numpy as np
import torch
from torch.distributions import kl_divergence

from lr_ablation_backend import LearningRateBackend
from torch_backend import ppo_terms


def anchor_at(schedule, steps):
    if schedule['kind'] == 'constant':
        return float(schedule['initial'])
    if schedule['kind'] != 'linear' or schedule['duration'] <= 0:
        raise ValueError('Invalid anchor schedule')
    progress = np.clip((steps - schedule['start_step']) / schedule['duration'], 0., 1.)
    return float(schedule['initial'] + progress * (schedule['final'] - schedule['initial']))


class AnchorBackend(LearningRateBackend):
    def set_progress(self, steps):
        self.cfg['anchor_coef'] = anchor_at(self.cfg['anchor_schedule'], steps)

    def objective_diagnostics(self, batch):
        """Read-only gradient attribution on a fixed minibatch, before PPO updates."""
        item = {k: v[:self.cfg['batch_size']] for k, v in batch.items()}
        lp, entropy, mask = self.likelihood(item)
        adv = self.tensor(item['advantages'])[:, None]
        pg, _, _ = ppo_terms(lp, self.tensor(item['old_lp']), adv, mask, self.cfg['clip_coef'])
        f = self.tensor(item['features'], self.policy_dtype)
        legal = self.tensor(item['legal'], torch.bool)
        with torch.no_grad():
            reference = self.categorical(self.reference, f, legal)
        anchor = kl_divergence(reference, self.categorical(self.actor, f, legal))
        anchor = (anchor * mask).sum() / mask.sum().clamp_min(1)
        parameters = tuple(self.actor.parameters())
        def gradient(loss):
            grads = torch.autograd.grad(loss, parameters, allow_unused=True)
            return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                              for p, g in zip(parameters, grads)])
        policy_g, anchor_g = gradient(pg), gradient(anchor)
        pn, an = policy_g.norm(), anchor_g.norm()
        cosine = torch.dot(policy_g, anchor_g) / (pn * an).clamp_min(1e-20)
        with torch.no_grad():
            prediction = self.values_batch(batch['state'])
        target = np.asarray(batch['returns'])
        variance = float(np.var(target))
        return dict(anchor_coef_effective=self.cfg['anchor_coef'],
                    diagnostic_pg_grad_norm=float(pn), diagnostic_anchor_grad_norm=float(an),
                    diagnostic_weighted_anchor_grad_norm=float(an) * self.cfg['anchor_coef'],
                    diagnostic_anchor_pg_cosine=float(cosine),
                    diagnostic_anchor_pg_ratio=float(an / pn.clamp_min(1e-20)) * self.cfg['anchor_coef'],
                    critic_explained_variance=1. - float(np.var(target - prediction)) / max(variance, 1e-12),
                    critic_target_variance=variance)
