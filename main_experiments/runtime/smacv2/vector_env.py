"""Spawned SMAC workers with independent RNGs and batched synchronous stepping."""
import multiprocessing as mp
import os
from pathlib import Path
import random
import signal
import time
import traceback

import numpy as np

from common import load_environment_factory, observe, seed_environment, transition_flags


def _worker(conn, project_root, task, seed, log_path, mock=False, auto_reset=True):
    os.setsid()  # worker and its own SC2 descendants form an isolated group
    env = None
    if log_path:
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, 1); os.dup2(fd, 2); os.close(fd)
    try:
        if not mock:
            # SC2 chooses a port before binding it; ephemeral client connections
            # can steal that port. Coordinate non-ephemeral leases in new workers.
            from sc2_port_lease import install
            install()
        random.seed(seed); np.random.seed(seed)
        if mock:
            env = MockEnv(seed)
        else:
            env = load_environment_factory(project_root)(task, seed=seed)
        count = seed_environment(env, seed)
        if not mock and count == 0:
            raise RuntimeError('SMACv2 distribution RNGs were not found')
        env.reset()
        current = observe(env)
        conn.send(('ready', (env.get_env_info(), current)))
        while True:
            command, payload = conn.recv()
            if command == 'close':
                break
            if command != 'step':
                raise ValueError('Unknown worker command: ' + command)
            actions = np.asarray(payload)
            if not current[2][np.arange(len(actions)), actions].all():
                raise ValueError('Illegal action passed to worker')
            reward, done, info = env.step(actions.tolist())
            term, trunc = transition_flags(env, done, info)
            if done:
                final_state = np.zeros_like(current[1]) if term else np.asarray(env.get_state(), np.float32)
                if auto_reset:
                    env.reset()
                    current = observe(env)
            else:
                current = observe(env)
                final_state = current[1]
            conn.send(('step', dict(obs=current[0], state=current[1], legal=current[2],
                bootstrap_state=final_state, reward=float(reward), done=bool(done),
                terminated=term, truncated=trunc, win=float(info.get('battle_won', False)))))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try: conn.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError): pass
    finally:
        if env is not None:
            try: env.close()
            except Exception: pass
        conn.close()


class ParallelEnv:
    def __init__(self, project_root, task, n_envs, seed, log_dir=None, timeout=180, startup_batch=4, mock=False, auto_reset=True):
        self.timeout = timeout
        self.processes, self.parents = [], []
        self.n_envs = n_envs
        context = mp.get_context('spawn')
        if log_dir: Path(log_dir).mkdir(parents=True, exist_ok=True)
        ready = []
        try:
            for start in range(0, n_envs, startup_batch):
                for index in range(start, min(start+startup_batch, n_envs)):
                    parent, child = context.Pipe()
                    path = Path(log_dir)/('env_%03d.log' % index) if log_dir else None
                    process = context.Process(target=_worker,
                        args=(child, project_root, task, seed+index, path, mock, auto_reset))
                    process.start(); child.close()
                    self.processes.append(process); self.parents.append(parent)
                for index in range(start, len(self.parents)):
                    ready.append(self._receive(index, 'ready'))
            self.info = ready[0][0]
            if any(info != self.info for info, _ in ready):
                raise ValueError('Vector environment shapes differ')
            self.obs, self.state, self.legal = [np.stack([item[1][k] for item in ready]) for k in range(3)]
            self.resets = np.ones(n_envs, bool)
        except BaseException:
            self.close(); raise

    def _receive(self, index, expected):
        parent = self.parents[index]
        if not parent.poll(self.timeout):
            raise TimeoutError('Environment %d did not respond within %ss' % (index, self.timeout))
        kind, payload = parent.recv()
        if kind == 'error':
            raise RuntimeError('Environment %d failed:\n%s' % (index, payload))
        if kind != expected:
            raise RuntimeError('Unexpected worker reply: ' + kind)
        return payload

    def step(self, actions, active=None):
        if len(actions) != self.n_envs:
            raise ValueError('Action batch has the wrong number of environments')
        active = np.ones(self.n_envs, bool) if active is None else np.asarray(active, bool)
        for i, (parent, action) in enumerate(zip(self.parents, actions)):
            if active[i]: parent.send(('step', action))
        data = [self._receive(i, 'step') if active[i] else dict(obs=self.obs[i],state=self.state[i],
                legal=self.legal[i],bootstrap_state=self.state[i],reward=0.,done=False,
                terminated=False,truncated=False,win=0.) for i in range(self.n_envs)]
        result = {k: np.stack([item[k] for item in data]) for k in data[0]}
        self.obs, self.state, self.legal = result['obs'], result['state'], result['legal']
        self.resets = result['done']
        return result

    def close(self):
        for parent in self.parents:
            try: parent.send(('close', None))
            except (OSError, EOFError, BrokenPipeError): pass
        deadline = time.monotonic()+15
        for process in self.processes:
            process.join(max(0, deadline-time.monotonic()))
        for process in self.processes:
            if process.is_alive():
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        for process in self.processes:
            process.join(3)
            if process.is_alive():
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                process.join()
        for parent in self.parents:
            parent.close()


class MockEnv:
    """Small spawned-process contract fixture; never used in real training."""
    def __init__(self, seed): self.seed = seed; self.ep = -1
    def reset(self): self.t = 0; self.ep += 1
    def get_env_info(self): return dict(n_agents=1, n_actions=2, obs_shape=2, state_shape=2)
    def get_obs(self): return [[self.t, self.ep]]
    def get_state(self): return [self.t, self.ep]
    def get_avail_actions(self): return [[1, 1]]
    def step(self, action):
        self.t += 1
        done = self.t == 2
        return 1., done, dict(episode_limit=done and self.seed % 2 == 0, battle_won=False)
    def close(self): pass
