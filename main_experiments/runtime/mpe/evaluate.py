"""Independent seed-fixed deployment/sampling return evaluation, no train RNG mutation."""
import argparse
from pathlib import Path
import numpy as np
import torch
from common import read, write, seed_all, sha
from model import ContinuousMACFlow, GaussianStudent
from mpe_env import Batch


def evaluate(checkpoint, cfg, episodes=20, seed=10000):
    seed_all(seed); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(checkpoint, map_location=device)
    model = ContinuousMACFlow(cfg['obs_dim']).to(device)
    if ck['phase'] == 'pretrain':
        model.load_state_dict(ck['model']); policy = GaussianStudent(model.student, cfg['initial_std']).to(device)
    else:
        policy = GaussianStudent(model.student, cfg['initial_std']).to(device); policy.load_state_dict(ck['actor'])
    policy.eval(); results = {}
    for mode in ('deployment', 'ppo_sampling'):
        returns = []; clipped = total = 0
        for start in range(0, episodes, 20):
            count = min(20, episodes-start)
            batch = Batch(cfg['task'], cfg['data_root'], range(seed+start, seed+start+count))
            # Independent torch generator per episode: stable across batch sizes.
            rngs = [torch.Generator(device=device).manual_seed(seed+start+i) for i in range(count)]
            sums = np.zeros(count)
            for _ in range(25):
                with torch.no_grad():
                    obs = torch.tensor(batch.obs(), device=device); dist = policy(obs)
                    noise = torch.stack([torch.randn((3, 2), generator=g, device=device) for g in rngs])
                    raw = dist.mean if mode == 'deployment' else dist.mean + dist.stddev * noise
                    clipped += int((raw.abs() > 1).sum()); total += raw.numel()
                    _, _, rewards, _ = batch.step(raw.clamp(-1, 1).cpu().numpy())
                sums += rewards.mean(-1)
            returns.extend(sums.tolist())
        results[mode] = dict(mean_return=float(np.mean(returns)), std_return=float(np.std(returns, ddof=1)),
            episodes=episodes, seed=seed, horizon=25, clipped_fraction=clipped/total, episode_returns=returns)
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), step=ck['step'], phase=ck['phase'],
                metric='episode return averaged over 3 controlled agents; raw OMAR rewards', policies=results)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--config', required=True)
    p.add_argument('--output', required=True); p.add_argument('--episodes', type=int, default=20); p.add_argument('--seed', type=int, default=10000)
    a = p.parse_args(); write(a.output, evaluate(a.checkpoint, read(a.config), a.episodes, a.seed))
