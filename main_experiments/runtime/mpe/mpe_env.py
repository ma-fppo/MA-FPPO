"""Headless OMAR MPE, using the vendored environment's actual step/reset methods."""
import importlib
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
sys.path.insert(0, str(Path(__file__).parent / 'vendor'))
import numpy as np
import torch
from gym import spaces
from multiagent.environment import MultiAgentEnv
from ddpg_agent.networks import MLPNetwork
from data import with_ids


def headless(task):
    scenario = importlib.import_module('multiagent.scenarios.' + task).Scenario()
    # Bypass only pygame window/font setup. Dynamics, action scaling, reset,
    # reward and observations execute the unmodified original implementation.
    env = MultiAgentEnv.__new__(MultiAgentEnv)
    env.world = scenario.make_world(); env.n = len(env.world.agents)
    env.agents = env.world.policy_agents
    env.reset_callback = scenario.reset_world; env.reward_callback = scenario.reward
    env.observation_callback = scenario.observation
    env.done_callback = env.info_callback = env.post_step_callback = None
    env.discrete_action_input = env.discrete_action_space = env.force_discrete_action = env.shared_reward = False
    env.action_space = [spaces.Box(-1., 1., shape=(2,), dtype=np.float32) for _ in env.agents]
    env.max_timestep = 25; env.time = 0
    assert all(a.silent and a.movable for a in env.agents)
    return env


def prey_policy(root, task="simple_tag"):
    assert task in ("simple_tag", "simple_world")
    checkpoint = torch.load(Path(root) / task / 'pretrained_adv_model.pt', map_location='cpu')
    assert checkpoint['init_dict']['discrete_action'] is False
    # The original wrapper leaves discrete_action=True.
    # The original checkpoint metadata and recorded prey actions both require tanh.
    meta = checkpoint['init_dict']; agent = meta['agent_init_params'][-1]
    model = MLPNetwork(agent['num_in_pol'], agent['num_out_pol'], hidden_dim=meta['hidden_dim'], constrain_out=True, discrete_action=False)
    model.load_state_dict(checkpoint['agent_params'][-1]['policy'])
    return model.eval().requires_grad_(False)


class Batch:
    def __init__(self, task, root, seeds):
        self.task = task; self.envs = []; self.rngs = []; self.raw = []; self.elapsed = []
        saved = np.random.get_state()
        for seed in seeds:
            np.random.seed(int(seed)); env = headless(task); obs = env.reset()
            self.envs.append(env); self.raw.append(obs); self.rngs.append(np.random.get_state()); self.elapsed.append(0)
        np.random.set_state(saved)
        self.prey = prey_policy(root, task) if task in ('simple_tag', 'simple_world') else None

    def obs(self):
        return with_ids(np.array([o[:3] for o in self.raw], dtype=np.float32))

    def step(self, actions):
        active = len(actions)
        assert 0 < active <= len(self.envs)
        assert actions.shape == (active, 3, 2) and np.isfinite(actions).all()
        assert np.max(np.abs(actions)) <= 1.000001
        if self.prey is not None:
            with torch.no_grad():
                prey_actions = self.prey(torch.tensor(np.array([o[-1] for o in self.raw[:active]]), dtype=torch.float32)).clamp(-1, 1).numpy()
        final = []; rewards = []; done = []; saved = np.random.get_state()
        for i, env in enumerate(self.envs[:active]):
            np.random.set_state(self.rngs[i])
            action = list(actions[i]) + ([prey_actions[i]] if self.prey is not None else [])
            obs, reward, terms, _ = env.step(action)
            assert not any(terms), 'Unexpected true terminal: revise GAE bootstrapping'
            self.elapsed[i] += 1; ended = self.elapsed[i] == 25
            final.append(obs[:3]); rewards.append(reward[:3]); done.append(ended)
            if ended: obs = env.reset(); self.elapsed[i] = 0
            self.raw[i] = obs; self.rngs[i] = np.random.get_state()
        np.random.set_state(saved)
        return self.obs()[:active], with_ids(np.array(final, dtype=np.float32)), np.array(rewards, dtype=np.float32), np.array(done, dtype=np.float32)


def worker(conn, task, root, seeds):
    try:
        torch.set_num_threads(1)
        batch = Batch(task, root, seeds); conn.send(('ok', batch.obs()))
        while True:
            cmd, value = conn.recv()
            if cmd == 'close': break
            conn.send(('ok', batch.step(value)))
    except BaseException:
        try: conn.send(('error', traceback.format_exc()))
        except Exception: pass
    finally: conn.close()


class Pool:
    def __init__(self, task, root, n_envs, workers, seed):
        assert n_envs % workers == 0
        self.conns = []; self.processes = []; self.obs_chunks = []
        ctx = mp.get_context('spawn')
        for seeds in np.array_split(np.arange(n_envs) + seed, workers):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=worker, args=(child, task, root, seeds.tolist()), daemon=True)
            process.start(); child.close(); self.conns.append(parent); self.processes.append(process)
        self.current = np.concatenate([self.receive(c) for c in self.conns])

    def receive(self, conn):
        if not conn.poll(180): raise TimeoutError('MPE worker did not respond')
        status, data = conn.recv()
        if status != 'ok': raise RuntimeError(data)
        return data

    def step(self, actions):
        active = len(actions)
        assert 0 < active <= len(self.current)
        width = len(self.current) // len(self.conns)
        pending = []
        for start, c in zip(range(0, len(self.current), width), self.conns):
            chunk = actions[start:min(start + width, active)]
            if len(chunk):
                c.send(('step', chunk)); pending.append(c)
        values = [self.receive(c) for c in pending]
        result = tuple(np.concatenate(x) for x in zip(*values))
        if active == len(self.current): self.current = result[0]
        else: self.current[:active] = result[0]
        return result

    def close(self):
        for c in self.conns:
            try: c.send(('close', None))
            except (BrokenPipeError, EOFError): pass
        for p in self.processes:
            p.join(3)
            if p.is_alive(): p.terminate(); p.join(3)
        for c in self.conns: c.close()
