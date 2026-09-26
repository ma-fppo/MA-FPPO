"""Verify the full original SMAC Poor arrays against the real SC2 environment."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from common import atomic_json, load_environment_factory, observe, sha256
from sc2_port_lease import install
install()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    cfg=json.loads(Path(p.parse_args().config).read_text())
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    folder=Path(cfg['data_root'])/cfg['task']/cfg['split']
    arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in ['obs','actions','rewards','legals','path_lengths','states','discounts']}
    n=arrays['obs'].shape[1];lengths=arrays['path_lengths'];total=len(arrays['obs'])
    checks=dict(length_sum=int(lengths.sum())==total,positive_lengths=bool((lengths>0).all()),
                all_rows_aligned=all(len(v)==total for k,v in arrays.items() if k!='path_lengths'))
    illegal=0;zero_legal=0;id_bad=0;finite=True
    for start in range(0,total,8192):
        a={k:v[start:start+8192] for k,v in arrays.items() if k!='path_lengths'}
        finite &= all(bool(np.isfinite(v).all()) for v in a.values())
        id_bad += int(np.any(a['obs'][...,:n]!=np.eye(n),axis=(1,2)).sum())
        zero_legal += int((a['legals'].sum(-1)==0).sum())
        assert a['actions'].min()>=0 and a['actions'].max()<a['legals'].shape[-1]
        illegal += int((np.take_along_axis(a['legals'],a['actions'][...,None],axis=-1)[...,0]==0).sum())
    checks.update(all_finite=finite,one_hot_ids_prefix=id_bad==0,recorded_actions_legal=illegal==0,
                  nonempty_legal_masks=zero_legal==0,shared_team_rewards=bool(np.all(arrays['rewards']==arrays['rewards'][:,:1])))
    env=load_environment_factory(cfg['project_root'])(cfg['task'],seed=90000)
    try:
        env.reset();obs,state,legal=observe(env);info=env.get_env_info()
        checks.update(real_smac_v1=getattr(env,'_ma_fppo_environment_kind',None)=='smac_v1',
            observation_dimensions=obs.shape==(n,arrays['obs'].shape[-1]-n),
            legal_dimensions=legal.shape==arrays['legals'].shape[1:],
            state_dimensions=state.shape==arrays['states'].shape[1:])
        for _ in range(10):
            actions=[int(np.flatnonzero(row)[-1]) for row in legal]
            _,done,_=env.step(actions)
            if done:env.reset()
            obs,state,legal=observe(env)
    finally:env.close()
    starts=np.r_[0,np.cumsum(lengths[:-1])]
    returns=np.add.reduceat(arrays['rewards'][:,0],starts)
    result=dict(status='verified' if all(checks.values()) else 'failed',checks=checks,task=cfg['task'],
        split=cfg['split'],data_path=str(folder),env_info=info,rows=total,episodes=len(lengths),
        episode_length_range=[int(lengths.min()),int(lengths.max())],mean_dataset_team_return=float(returns.mean()),
        illegal_recorded_actions=illegal,empty_masks=zero_legal,
        observation_transform='move existing leading one-hot agent IDs to the end; no normalization',
        files={k:dict(path=str(folder/(k+'.npy')),shape=list(v.shape),dtype=str(v.dtype),sha256=sha256(folder/(k+'.npy'))) for k,v in arrays.items()})
    atomic_json(out/'data_environment_audit.json',result);print(json.dumps(result,indent=2))
    if result['status']!='verified':raise AssertionError(checks)


if __name__=='__main__':main()
