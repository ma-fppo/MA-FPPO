"""Native JAX PPO for the existing MAC-Flow single-step student checkpoint.

The recurrent encoder is deliberately frozen. Its rollout features/carry are
therefore exact and do not become stale during PPO epochs. Only the deployed
student and a new centralized V(s) are optimized; offline Q/teacher stay intact.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ml_collections import ConfigDict

from common import finite_metrics

# A100/XLA's reduced-precision matmuls can change log probabilities when
# rollout batch=1 is replayed as a minibatch, despite identical parameters.
jax.config.update('jax_default_matmul_precision', 'highest')


class CentralValue(nn.Module):
    @nn.compact
    def __call__(self, state):
        x = nn.tanh(nn.Dense(128)(state))
        x = nn.tanh(nn.Dense(128)(x))
        return nn.Dense(1, kernel_init=nn.initializers.zeros)(x)[..., 0]


class JaxBackend:
    def __init__(self, cfg, info):
        self.cfg = cfg
        sys.path.insert(0, str(Path(cfg['project_root']) / 'mac_flow_offline_marl'))
        from agents.discrete_macflow import MACFlowDiscreteAgent
        saved = json.loads(Path(cfg['flags_json']).read_text())
        self.n_agents, self.action_dim = info['n_agents'], info['n_actions']
        c = ConfigDict(saved['agent_config'])
        self.agent = MACFlowDiscreteAgent.create(cfg['seed'],
            jnp.zeros((1, 1, self.n_agents, info['obs_shape'])),
            jnp.zeros((1, 1, self.n_agents, self.action_dim)),
            ['agent_' + str(i) for i in range(self.n_agents)], c)
        self.agent = self.agent.replace(network=self.agent.network.load(cfg['checkpoint']))
        self.key = jax.random.PRNGKey(cfg['seed'])
        self.actor_key = 'modules_actor_onestep_flow'
        self.actor = self.agent.network.params[self.actor_key]
        self.reference = self.actor
        self.critic_def = CentralValue()
        self.critic = self.critic_def.init(self.key, jnp.zeros((1, info['state_shape'])))['params']
        self.actor_tx = optax.chain(optax.clip_by_global_norm(cfg['max_grad_norm']), optax.adam(cfg['actor_lr'], eps=1e-5))
        self.critic_tx = optax.chain(optax.clip_by_global_norm(cfg['max_grad_norm']), optax.adam(cfg['critic_lr'], eps=1e-5))
        self.actor_opt = self.actor_tx.init(self.actor)
        self.critic_opt = self.critic_tx.init(self.critic)
        self.carry = None
        self._distribution = jax.jit(self._distribution_impl)
        self._actor_grad = jax.jit(jax.value_and_grad(self._actor_loss, has_aux=True))
        self._critic_grad = jax.jit(jax.value_and_grad(self._critic_loss))
        self._value = jax.jit(lambda p, s: self.critic_def.apply({'params': p}, s))
        self._encode = jax.jit(self._encode_impl)
        self._grad_norm = jax.jit(optax.global_norm)
        def apply_actor(params, state, grads):
            updates, state = self.actor_tx.update(grads, state, params)
            return optax.apply_updates(params, updates), state
        def apply_critic(params, state, grads):
            updates, state = self.critic_tx.update(grads, state, params)
            return optax.apply_updates(params, updates), state
        self._apply_actor = jax.jit(apply_actor)
        self._apply_critic = jax.jit(apply_critic)

    def reset(self):
        self.carry = None

    def actor_vector(self):
        return np.concatenate([np.asarray(p).ravel().copy() for p in jax.tree_util.tree_leaves(self.actor)])

    def rng_state(self):
        return self.key

    def restore_rng(self, state):
        self.key = state

    def seed(self, seed):
        self.key = jax.random.PRNGKey(seed)

    def make_rngs(self, seeds):
        return [jax.random.PRNGKey(int(seed)) for seed in seeds]

    def _encode_impl(self, obs, resets, carry):
        return self.agent.network.select('seq_encoder')(
            obs, resets, initial_carry=carry, return_carry=True)

    def _distribution_impl(self, actor, features, legal):
        params = dict(self.agent.network.params)
        params[self.actor_key] = actor
        logits = self.agent.network.select('actor_onestep_flow')(
            features, jnp.zeros((*features.shape[:-1], self.action_dim)),
            params=params, is_encoded=self.agent.config.get('use_lstm', False))
        logits = jnp.where(legal, logits / self.cfg['temperature'], -1e9)
        return jax.nn.log_softmax(logits, -1)

    def act(self, obs, legal, deterministic=False):
        actions, cache = self.act_batch(np.asarray(obs)[None], np.asarray(legal)[None],
                                        deterministic=deterministic)
        return actions[0], {k: v[0] for k, v in cache.items()}

    def act_batch(self, obs, legal, resets=None, deterministic=False, rngs=None):
        if not np.asarray(legal).any(-1).all():
            raise ValueError('Empty legal action mask')
        # The original JAX code PREPENDS IDs; the Torch branches APPEND them.
        b = len(obs)
        ids = jnp.broadcast_to(jnp.eye(self.n_agents), (b, self.n_agents, self.n_agents))
        features = jnp.concatenate([ids, jnp.asarray(obs)], -1)
        if self.agent.config.get('use_lstm', False):
            # Match the native evaluation carry: only reset on episode boundary.
            resets = np.full(b, self.carry is None) if resets is None else np.asarray(resets)
            reset_agents = jnp.broadcast_to(jnp.asarray(resets)[None, :, None], (1, b, self.n_agents))
            encoded, self.carry = self._encode(features[None], reset_agents, self.carry)
            features = encoded[0]
        lp = self._distribution(self.actor, features, jnp.asarray(legal))
        if rngs is None:
            self.key, key = jax.random.split(self.key)
            actions = jnp.argmax(lp, -1) if deterministic else jax.random.categorical(key, lp, axis=-1)
        else:
            split = jax.vmap(jax.random.split)(jnp.stack(rngs))
            rngs[:] = [key for key in split[:, 0]]
            actions = jnp.argmax(lp, -1) if deterministic else jax.vmap(
                lambda key, probs: jax.random.categorical(key, probs, axis=-1))(split[:, 1], lp)
        cache = {'features': np.asarray(features), 'legal': np.asarray(legal, bool),
                 'actions': np.asarray(actions), 'old_lp': np.asarray(jnp.take_along_axis(lp, actions[..., None], -1)[..., 0])}
        return np.asarray(actions), cache

    def value(self, state):
        return float(self.values_batch(np.asarray(state)[None])[0])

    def values_batch(self, states):
        return np.asarray(self._value(self.critic, jnp.asarray(states)))

    def replay_error(self, batch):
        return float(np.max(np.abs(self.log_probs_numpy(batch)-batch['old_lp'])))

    def log_probs_numpy(self, batch):
        size = self.cfg.get('batch_size', 128)
        result = []
        for start in range(0, len(batch['features']), size):
            f = jnp.asarray(batch['features'][start:start+size]); legal = jnp.asarray(batch['legal'][start:start+size])
            all_lp = self._distribution(self.actor, f, legal)
            lp = jnp.take_along_axis(all_lp, jnp.asarray(batch['actions'][start:start+size])[..., None], -1)[..., 0]
            result.append(np.asarray(lp))
        return np.concatenate(result)

    def _critic_loss(self, params, state, returns):
        values = self.critic_def.apply({'params': params}, state)
        return .5 * jnp.mean((values - returns)**2)

    def _actor_loss(self, actor, batch):
        lp_all = self._distribution_impl(actor, batch['features'], batch['legal'])
        lp = jnp.take_along_axis(lp_all, batch['actions'][..., None], -1)[..., 0]
        log_ratio = lp - batch['old_lp']
        ratio = jnp.exp(log_ratio)
        mask = (batch['legal'].sum(-1) > 1).astype(jnp.float32)
        den = jnp.maximum(mask.sum(), 1.)
        adv = batch['advantages'][:, None]
        surrogate = jnp.minimum(ratio*adv, jnp.clip(ratio, 1-self.cfg['clip_coef'], 1+self.cfg['clip_coef'])*adv)
        pg = -(surrogate * mask).sum() / den
        entropy = (-jnp.sum(jnp.exp(lp_all)*lp_all, -1) * mask).sum() / den
        ref_lp = self._distribution_impl(self.reference, batch['features'], batch['legal'])
        anchor = (jnp.sum(jnp.exp(ref_lp)*(ref_lp-lp_all), -1) * mask).sum() / den
        kl = (((ratio-1)-log_ratio)*mask).sum() / den
        cf = (((jnp.abs(ratio-1)>self.cfg['clip_coef']).astype(jnp.float32))*mask).sum()/den
        loss = pg - self.cfg['entropy_coef']*entropy + self.cfg['anchor_coef']*anchor
        return loss, {'policy_loss': pg, 'approx_kl': kl, 'clip_fraction': cf, 'entropy': entropy,
                      'anchor_kl': anchor, 'max_abs_log_ratio': jnp.max(jnp.abs(log_ratio))}

    def update(self, batch, actor_update=True):
        batch = {k: jnp.asarray(v) for k, v in batch.items()}
        vl, cg = self._critic_grad(self.critic, batch['state'], batch['returns'])
        result = {'value_loss': float(vl), 'critic_grad_norm': float(self._grad_norm(cg)),
                  'policy_loss': 0., 'approx_kl': 0., 'clip_fraction': 0., 'entropy': 0.,
                  'anchor_kl': 0., 'actor_grad_norm': 0., 'actor_updated': 0.}
        finite_metrics(result)
        self.critic, self.critic_opt = self._apply_critic(self.critic, self.critic_opt, cg)
        if actor_update:
            (_, metrics), grads = self._actor_grad(self.actor, batch)
            result.update({k: float(v) for k, v in metrics.items()})
            result['actor_grad_norm'] = float(self._grad_norm(grads))
            finite_metrics(result)
            if result['max_abs_log_ratio'] > 60:
                raise FloatingPointError('PPO ratio diverged')
            if result['approx_kl'] <= self.cfg['target_kl']:
                self.actor, self.actor_opt = self._apply_actor(self.actor, self.actor_opt, grads)
                result['actor_updated'] = 1.
        return finite_metrics(result)

    def save(self, path, iteration, env_steps):
        payload = {'format': 'macflow_ppo_jax_v1', 'config': self.cfg, 'iteration': iteration,
                   'env_steps': env_steps, 'actor': self.actor, 'critic': self.critic,
                   'reference': self.reference, 'actor_optimizer': self.actor_opt,
                   'critic_optimizer': self.critic_opt, 'key': self.key,
                   'numpy_rng': np.random.get_state()}
        tmp = str(path) + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(jax.device_get(payload), f)
        os.replace(tmp, str(path))

    def restore(self, path):
        with open(path, 'rb') as f:
            p = pickle.load(f)
        if p.get('format') != 'macflow_ppo_jax_v1':
            raise ValueError('Expected native JAX PPO checkpoint')
        for key in ['variant', 'checkpoint', 'flags_json', 'temperature',
                    'source_checkpoint_sha256', 'actor_lr', 'critic_lr']:
            if p['config'].get(key) != self.cfg.get(key):
                raise ValueError('Resume configuration mismatch: ' + key)
        self.actor, self.critic, self.reference = jax.tree_util.tree_map(jnp.asarray, (p['actor'], p['critic'], p['reference']))
        self.actor_opt = jax.tree_util.tree_map(jnp.asarray, p['actor_optimizer'])
        self.critic_opt = jax.tree_util.tree_map(jnp.asarray, p['critic_optimizer'])
        self.key = jnp.asarray(p['key']); np.random.set_state(p['numpy_rng']); self.reset()
        return p['iteration'], p['env_steps']
