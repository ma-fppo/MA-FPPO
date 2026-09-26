"""Collect complete SMACv2 episodes with a frozen 15M MAC-Flow policy.

Workers reuse SC2 processes, but explicitly reset only after the final next
observation has been recorded. Only the model weights are loaded: no optimizer
is resumed and no model update is performed.
"""
import argparse
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np

from dataset_io import FIELDS, atomic_json, digest, read_metadata, summary, write_shard


def worker(conn, cfg, index, first_id, log_path):
    os.setsid()
    env = None
    fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1); os.dup2(fd, 2); os.close(fd)
    try:
        from common import load_environment_factory, observe, seed_environment, transition_flags
        from sc2_port_lease import install
        install()
        sc2_seed = cfg['sc2_seed_start'] + index
        env = load_environment_factory(cfg['project_root'])(cfg['task'], seed=sc2_seed)
        ordinal = -1

        def reset(episode_id):
            nonlocal ordinal
            ordinal += 1
            seed = cfg['seed_start'] + episode_id
            if not seed_environment(env, seed):
                raise RuntimeError('No SMACv2 capability RNG found')
            env.reset()
            obs = observe(env)
            # Preserve initial unit composition/position where the wrapper exposes it.
            units = {}
            base = env
            for _ in range(4):
                if hasattr(base, 'agents') and hasattr(base, 'enemies'):
                    for group in ('agents', 'enemies'):
                        units[group] = [dict(id=int(i), unit_type=int(u.unit_type),
                                            x=float(u.pos.x), y=float(u.pos.y))
                                        for i, u in sorted(getattr(base, group).items())]
                    break
                base = getattr(base, '_env', getattr(base, 'env', None))
                if base is None:
                    break
            return obs, dict(episode_id=int(episode_id), capability_seed=int(seed),
                             worker=index, sc2_seed=sc2_seed, worker_reset_ordinal=ordinal,
                             initial_units=units)

        current, meta = reset(first_id)
        conn.send(('ready', (env.get_env_info(), current, meta)))
        done = False
        while True:
            command, value = conn.recv()
            if command == 'close':
                break
            if command == 'reset':
                if not done:
                    raise ValueError('Refusing to reset an unfinished episode')
                current, meta = reset(value); done = False
                conn.send(('reset', (current, meta)))
            elif command == 'step':
                if done:
                    raise ValueError('Refusing to step a completed episode')
                actions = np.asarray(value)
                if not current[2][np.arange(len(actions)), actions].all():
                    raise ValueError('Illegal policy action')
                reward, done, info = env.step(actions.tolist())
                term, trunc = transition_flags(env, done, info)
                # Observe BEFORE reset, including the actual final observation.
                current = observe(env)
                if done and 'battle_won' not in info:
                    raise ValueError('Missing explicit battle_won label')
                conn.send(('step', (current, float(reward), bool(done), term, trunc,
                                    int(bool(info.get('battle_won', False))))))
            else:
                raise ValueError('Unknown command: ' + command)
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            conn.send(('error', traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        conn.close()


class Pool:
    def __init__(self, cfg, ids, log_dir):
        self.parents, self.processes, self.current, self.meta = [], [], [], []
        context = mp.get_context('spawn')
        log_dir.mkdir(parents=True, exist_ok=False)
        try:
            for start in range(0, len(ids), 4):
                for i in range(start, min(start + 4, len(ids))):
                    parent, child = context.Pipe()
                    p = context.Process(target=worker, args=(child, cfg, i, ids[i], log_dir / ('env_%03d.log' % i)))
                    p.start(); child.close()
                    self.parents.append(parent); self.processes.append(p)
                for i in range(start, len(self.parents)):
                    info, obs, meta = self.receive(i, 'ready')
                    if i and info != self.info:
                        raise ValueError('Inconsistent environment shapes')
                    self.info = info; self.current.append(obs); self.meta.append(meta)
        except BaseException:
            self.close(); raise

    def receive(self, i, expected):
        if not self.parents[i].poll(180):
            raise TimeoutError('SC2 worker timed out: ' + str(i))
        kind, payload = self.parents[i].recv()
        if kind == 'error':
            raise RuntimeError(payload)
        if kind != expected:
            raise RuntimeError('Unexpected worker reply: ' + kind)
        return payload

    def close(self):
        for c in self.parents:
            try:
                c.send(('close', None))
            except (OSError, EOFError):
                pass
        deadline = time.monotonic() + 15
        for p in self.processes:
            p.join(max(0, deadline - time.monotonic()))
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for p in self.processes:
                if p.is_alive():
                    try:
                        os.killpg(p.pid, sig)
                    except ProcessLookupError:
                        pass
            for p in self.processes:
                p.join(1)
        for c in self.parents:
            c.close()


def load_backend(cfg, info):
    import torch
    from torch_backend import TorchBackend
    for path, expected in cfg['source_hashes'].items():
        if digest(path) != expected:
            raise ValueError('Source changed: ' + path)
    if digest(cfg['ppo_checkpoint']) != cfg['ppo_sha256']:
        raise ValueError('Wrong 15M checkpoint')
    if digest(cfg['pretrained_checkpoint']) != cfg['pretrained_sha256']:
        raise ValueError('Wrong pretrained checkpoint')
    p = torch.load(cfg['ppo_checkpoint'], map_location='cpu')
    if p.get('format') != 'macflow_ppo_v1' or p['env_steps'] != 15000000:
        raise ValueError('Expected completed 15M PPO weights')
    policy_cfg = dict(p['config'], device='cuda', torch_threads=2, project_root=cfg['project_root'], checkpoint=cfg['pretrained_checkpoint'])
    expected = dict(task=cfg['task'], variant='torch', checkpoint=cfg['pretrained_checkpoint'],
                    project_root=cfg['project_root'], torch_history='recurrent',
                    temperature=1.0, student_latent_std=0.0, train_encoder=False)
    for key, value in expected.items():
        if policy_cfg.get(key) != value:
            raise ValueError('Unexpected checkpoint policy setting: ' + key)
    backend = TorchBackend(policy_cfg, info)
    backend.model.load_state_dict(p['model'], strict=True)
    # Actor appears both as a model submodule and as a separate saved state.
    for key, value in p['actor'].items():
        if not torch.equal(value, p['model']['actor_onestep_flow.' + key]):
            raise ValueError('Saved actor/model disagree')
    backend.actor.load_state_dict(p['actor'], strict=True)
    backend.model.requires_grad_(False).eval(); backend.actor.requires_grad_(False).eval()
    backend.reset()
    del p
    return backend


def collect(cfg, resume=False):
    for name, expected in cfg['collector_hashes'].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError('Collector source changed: ' + name)
    root = Path(cfg['output_dir']); root.mkdir(parents=True, exist_ok=True)
    # Lock is held until this function returns, including selection.
    lock = open(root / '.collection.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_path = root / 'collection_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != cfg or not resume:
            raise ValueError('Existing output requires identical config and explicit --resume')
    else:
        atomic_json(config_path, cfg)
    if cfg['target_episodes'] != 4 * cfg['keep_episodes'] or cfg['keep_episodes'] <= 0:
        raise ValueError('Invalid collection counts')
    original_lengths = np.load(cfg['original_path_lengths'], mmap_mode='r')
    if len(original_lengths) != cfg['original_episodes'] or int(original_lengths.sum()) != cfg['original_transitions']:
        raise ValueError('Original Replay dataset differs from the audited source')
    if cfg['mode'] == 'formal':
        if cfg['keep_episodes'] != cfg['original_episodes']:
            raise ValueError('Good dataset must keep the original number of trajectories')
        check = json.loads(Path(cfg['preflight_report']).read_text())
        if not (check['status'] == 'verified' and check['mode'] == 'preflight'
                and check['task'] == cfg['task'] and check['source_checkpoint_sha256'] == cfg['ppo_sha256']
                and check['raw']['episodes'] >= 32 and check['collector_hashes'] == cfg['collector_hashes']):
            raise ValueError('A successful real-environment preflight is required')
    # Each admission is checked again; no historical GPU usage is assumed.
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        fields = line.split(); mem[fields[0].rstrip(':')] = int(fields[1]) * 1024
    if mem['MemAvailable'] < (32 + 1.2 * cfg['n_envs']) * 2**30:
        raise RuntimeError('Insufficient available host RAM for SC2 workers')
    if shutil.disk_usage(root).free < cfg['min_free_disk_gib'] * 2**30:
        raise RuntimeError('Insufficient free disk for raw and selected datasets')
    if os.getloadavg()[0] > .8 * (os.cpu_count() or 1):
        raise RuntimeError('Host CPU load is too high to add SC2 workers')
    sys.path.insert(0, cfg['backend_dir'])
    for path, expected in cfg['source_hashes'].items():
        if digest(path) != expected:
            raise ValueError('Source changed: ' + path)
    raw = root / 'raw'; raw.mkdir(exist_ok=True)
    records, shard_manifest = [], []
    for path in sorted(raw.glob('shard_*.npz')):
        metas = read_metadata(path)
        if any(m['source_checkpoint_sha256'] != cfg['ppo_sha256'] for m in metas):
            raise ValueError('Foreign checkpoint in existing shard')
        records.extend(metas)
        shard_manifest.append(dict(file=path.name, sha256=digest(path), episodes=len(metas)))
    ids = [m['episode_id'] for m in records]
    if len(ids) != len(set(ids)) or not set(ids).issubset(set(range(cfg['target_episodes']))):
        raise ValueError('Duplicate or invalid existing episode IDs')
    existing_ids = set(ids)
    missing = iter(i for i in range(cfg['target_episodes']) if i not in existing_ids)
    initial = []
    for _ in range(min(cfg['n_envs'], cfg['target_episodes'] - len(ids))):
        initial.append(next(missing))
    session = uuid.uuid4().hex
    pending, pool, backend = [], None, None
    start, last_status = time.monotonic(), 0.0
    steps = 0

    def flush():
        if not pending:
            return
        path = raw / ('shard_%05d.npz' % len(shard_manifest))
        write_shard(path, pending)
        shard_manifest.append(dict(file=path.name, sha256=digest(path), episodes=len(pending)))
        records.extend(m for _, m in pending); pending.clear()

    try:
        if initial:
            atomic_json(root / 'status.json', dict(status='starting', pid=os.getpid(), committed_episodes=len(records)))
            pool = Pool(cfg, initial, root / 'worker_logs' / session)
            backend = load_backend(cfg, pool.info)
            before = backend.actor_vector()
            atomic_json(root / 'environment_info.json', dict(env_info=pool.info))
            n = len(initial); active = np.ones(n, bool); resets = np.ones(n, bool)
            buffers = [{k: [] for k in FIELDS} for _ in range(n)]
            identity = np.eye(pool.info['n_agents'], dtype=np.float32)
            while active.any():
                observations = np.stack([o[0] for o in pool.current])
                legals = np.stack([o[2] for o in pool.current])
                actions, _ = backend.act_batch(observations, legals, resets, deterministic=True)
                resets[:] = False
                for i in np.flatnonzero(active):
                    pool.parents[i].send(('step', actions[i]))
                reset_ids = []
                for i in np.flatnonzero(active):
                    previous = pool.current[i]
                    after, reward, done, term, trunc, win = pool.receive(i, 'step')
                    count = pool.info['n_agents']
                    row = dict(obs=np.concatenate([previous[0], identity], -1),
                               next_obs=np.concatenate([after[0], identity], -1),
                               states=previous[1], next_states=after[1], actions=actions[i].astype(np.int64),
                               rewards=np.full(count, reward, np.float32), legals=previous[2], next_legals=after[2],
                               terminals=np.full(count, term, bool), truncations=np.full(count, trunc, bool),
                               discounts=np.full(count, 0.0 if term else 1.0, np.float32))
                    for k in FIELDS:
                        buffers[i][k].append(row[k])
                    pool.current[i] = after; steps += 1
                    if done:
                        data = {k: np.stack(v) for k, v in buffers[i].items()}
                        meta = dict(pool.meta[i], session=session, source_checkpoint_sha256=cfg['ppo_sha256'],
                                    policy='deployment', win=win, length=len(data['actions']),
                                    terminated=term, truncated=trunc,
                                    **{'return': float(data['rewards'][:, 0].sum(dtype=np.float64))})
                        pending.append((data, meta)); buffers[i] = {k: [] for k in FIELDS}
                        episode_id = next(missing, None)
                        if episode_id is None:
                            active[i] = False
                        else:
                            pool.parents[i].send(('reset', episode_id)); reset_ids.append(i)
                for i in reset_ids:
                    pool.current[i], pool.meta[i] = pool.receive(i, 'reset'); resets[i] = True
                if len(pending) >= cfg['shard_episodes']:
                    flush()
                now = time.monotonic()
                if now - last_status >= 15:
                    all_meta = records + [m for _, m in pending]
                    state = dict(status='collecting', pid=os.getpid(), committed_episodes=len(records),
                                 completed_episodes=len(all_meta), target_episodes=cfg['target_episodes'],
                                 session_env_steps=steps, elapsed_seconds=now - start,
                                 steps_per_second=steps / max(now - start, 1),
                                 completed_win_rate=(sum(m['win'] for m in all_meta) / len(all_meta)) if all_meta else None)
                    atomic_json(root / 'status.json', state)
                    print(json.dumps(state), flush=True); last_status = now
            flush()
            if not np.array_equal(before, backend.actor_vector()):
                raise ValueError('Collector mutated actor weights')
        if len(records) != cfg['target_episodes']:
            raise ValueError('Collection count is not exact')
        if digest(cfg['ppo_checkpoint']) != cfg['ppo_sha256']:
            raise ValueError('Source checkpoint changed during rollout')
        atomic_json(root / 'raw_manifest.json', dict(shards=shard_manifest,
                    collection_config_sha256=digest(config_path), stats=summary(records)))
        atomic_json(root / 'status.json', dict(status='collected', committed_episodes=len(records),
                    target_episodes=cfg['target_episodes'], pid=os.getpid(), stats=summary(records)))
    except BaseException as exc:
        atomic_json(root / 'status.json', dict(status='failed', committed_episodes=len(records), error=repr(exc)))
        raise
    finally:
        if pool is not None:
            pool.close()
    # No SC2 processes remain while exporting/checking the selected dataset.
    from select_good import select
    if not (root / 'selection_report.json').exists():
        select(root)
    lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    # Never dispatch onto an occupied card based on an old utilization report.
    row = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                '--query-gpu=memory.used,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    used, free, utilization = [int(x.strip()) for x in row.strip().split(',')]
    pids = subprocess.check_output(['nvidia-smi', '-i', str(args.gpu),
                '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if used >= 1024 or free < 8192 or utilization > 10 or pids:
        raise RuntimeError('Selected GPU is occupied; choose a currently empty GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['OMP_NUM_THREADS'] = '1'; os.environ['MKL_NUM_THREADS'] = '1'
    collect(json.loads(Path(args.config).read_text()), resume=args.resume)
