"""Seed-fixed independent episode returns for four-agent Ant-v2."""
import argparse
import numpy as np
import torch
from common import read,write,seed_all,sha
from model import ContinuousMACFlow,GaussianStudent
from mamujoco_env import Batch


def evaluate(checkpoint,cfg,episodes=20,seed=10000):
    seed_all(seed);device='cuda';ck=torch.load(checkpoint,map_location=device)
    model=ContinuousMACFlow(cfg['obs_dim'],cfg['action_dim']).to(device)
    if ck['phase']=='pretrain':
        model.load_state_dict(ck['model']);policy=GaussianStudent(model.student,cfg['initial_std'],cfg['action_dim']).to(device)
    else:
        policy=GaussianStudent(model.student,cfg['initial_std'],cfg['action_dim']).to(device);policy.load_state_dict(ck['actor'])
    policy.eval();results={}
    for mode in ('deployment','ppo_sampling'):
        returns=[];lengths=[];clipped=total=0
        for start in range(0,episodes,20):
            count=min(20,episodes-start);batch=Batch(cfg['task'],cfg['data_root'],range(seed+start,seed+start+count))
            rngs=[torch.Generator(device=device).manual_seed(seed+start+i) for i in range(count)]
            sums=np.zeros(count);steps=np.zeros(count,int);active=np.ones(count,bool)
            try:
                for _ in range(1000):
                    with torch.no_grad():
                        dist=policy(torch.tensor(batch.obs(),device=device))
                        noise=torch.stack([torch.randn((cfg['n_agents'],cfg['action_dim']),generator=g,device=device) for g in rngs])
                        raw=dist.mean if mode=='deployment' else dist.mean+dist.stddev*noise
                        mask=torch.tensor(active,device=device);clipped+=int((raw[mask].abs()>1).sum());total+=raw[mask].numel()
                        _,_,_,rewards,done,_=batch.step(raw.clamp(-1,1).cpu().numpy(),active)
                    sums+=rewards.mean(-1);steps+=active.astype(int);active&=~done.astype(bool)
                    if not active.any():break
                assert not active.any()
            finally:batch.close()
            returns.extend(sums.tolist());lengths.extend(steps.tolist())
        results[mode]=dict(mean_return=float(np.mean(returns)),std_return=float(np.std(returns,ddof=1)),
            mean_length=float(np.mean(lengths)),episodes=episodes,seed=seed,horizon=1000,clipped_fraction=clipped/total,
            episode_returns=returns,episode_lengths=lengths)
    return dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),step=ck['step'],phase=ck['phase'],
        metric='raw Ant-v2 team episode return (identical reward shared by four agents)',policies=results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed',type=int,default=10000)
    a=p.parse_args();write(a.output,evaluate(a.checkpoint,read(a.config),a.episodes,a.seed))
