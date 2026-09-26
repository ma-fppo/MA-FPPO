"""Validate converted SMAC data and record real environment dimensions."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--runtime',required=True)
    a=p.parse_args();c=json.loads(Path(a.config).read_text());rt=Path(a.runtime)
    sys.path.insert(0,str(rt))
    from common import load_environment_factory,observe,atomic_json
    from sc2_port_lease import install
    folder=Path(c['data_root'])/c['task']/c['split']
    keys=('obs','actions','rewards','legals','path_lengths','states')
    arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in keys}
    n=arrays['obs'].shape[1];lengths=arrays['path_lengths'];rows=len(arrays['obs'])
    assert (lengths>0).all() and int(lengths.sum())==rows
    assert all(len(v)==rows for k,v in arrays.items() if k!='path_lengths')
    leading=rt.name=='smac'
    for start in range(0,rows,8192):
        z={k:v[start:start+8192] for k,v in arrays.items() if k!='path_lengths'}
        assert all(np.isfinite(v).all() for v in z.values())
        ids=z['obs'][...,:n] if leading else z['obs'][...,-n:]
        assert np.all(ids==np.eye(n)), 'Agent ID placement differs from the supplied data adapter'
        actions=z['actions'];assert actions.dtype.kind in 'iu'
        assert actions.min()>=0 and actions.max()<z['legals'].shape[-1]
        assert np.take_along_axis(z['legals'],actions[...,None],-1).all()
        assert z['legals'].any(-1).all()
    install();env=load_environment_factory(c['project_root'])(c['task'],seed=90000)
    try:
        env.reset();obs,state,legal=observe(env);info=env.get_env_info()
        assert obs.shape==(n,arrays['obs'].shape[-1]-n)
        assert state.shape==arrays['states'].shape[1:]
        assert legal.shape==arrays['legals'].shape[1:]
    finally:env.close()
    out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'data_environment_audit.json',dict(status='verified',env_info=info,rows=rows,episodes=len(lengths),agent_ids='leading' if leading else 'trailing'))
if __name__=='__main__':main()
