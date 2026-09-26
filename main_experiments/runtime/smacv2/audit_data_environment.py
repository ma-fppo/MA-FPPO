"""Validate self-collected Good arrays against the original SMACv2 environment."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from common import atomic_json,load_environment_factory,observe,seed_environment,sha256
from sc2_port_lease import install

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True)
 cfg=json.loads(Path(p.parse_args().config).read_text());folder=Path(cfg['data_root'])/cfg['task']/cfg['split']
 manifest=json.loads((folder/'dataset_manifest.json').read_text())
 assert cfg['split']=='Good' and manifest['status']=='verified' and manifest['task']==cfg['task']
 keys=['obs','actions','rewards','legals','path_lengths','states','next_obs','next_states','next_legals','terminals','truncations','discounts']
 arrays={k:np.load(folder/(k+'.npy'),mmap_mode='r') for k in keys}
 lengths=arrays['path_lengths'];n=arrays['obs'].shape[1];total=len(arrays['obs'])
 assert n==5 and (lengths>0).all() and int(lengths.sum())==total
 assert all(len(v)==total for k,v in arrays.items() if k!='path_lengths')
 for start in range(0,total,8192):
  a={k:v[start:start+8192] for k,v in arrays.items() if k!='path_lengths'}
  assert all(np.isfinite(v).all() for v in a.values())
  assert np.all(a['rewards']==a['rewards'][:,:1])
  assert np.all(a['obs'][...,-n:]==np.eye(n)) and np.all(a['next_obs'][...,-n:]==np.eye(n))
  assert a['actions'].dtype.kind in 'iu' and a['actions'].min()>=0 and a['actions'].max()<a['legals'].shape[-1]
  assert np.take_along_axis(a['legals'],a['actions'][...,None],-1).all()
  assert a['legals'].any(-1).all() and a['next_legals'].any(-1).all()
 ends=np.cumsum(lengths)-1
 assert np.array_equal(np.flatnonzero((arrays['terminals']|arrays['truncations'])[:,0]),ends)
 # Selected winners have true terminal endings, consistent with the inherited sampler.
 assert arrays['terminals'][ends].all() and not arrays['truncations'].any()
 files={k:dict(path=str(folder/(k+'.npy')),shape=list(v.shape),dtype=str(v.dtype),sha256=sha256(folder/(k+'.npy'))) for k,v in arrays.items()}
 for k,v in files.items():assert v['sha256']==manifest['files'][k+'.npy'],k
 files['dataset_manifest']=dict(path=str(folder/'dataset_manifest.json'),sha256=sha256(folder/'dataset_manifest.json'))
 install();env=load_environment_factory(cfg['project_root'])(cfg['task'],seed=80000)
 try:
  assert seed_environment(env,80000)>0
  env.reset();obs,state,legal=observe(env);info=env.get_env_info()
  assert info['n_agents']==n and obs.shape==(n,arrays['obs'].shape[-1]-n)
  assert legal.shape==arrays['legals'].shape[1:] and state.shape==arrays['states'].shape[1:]
  for _ in range(20):
   _,done,_=env.step([int(np.flatnonzero(row)[-1]) for row in legal])
   if done:env.reset()
   obs,state,legal=observe(env)
 finally:env.close()
 sys.path.insert(0,str(Path(cfg['project_root'])/'mac_flow_pytorch_smacv2'))
 from macflow_torch.data import ConvertedSMACv2Sequences,SMACv2DatasetSpec
 data=ConvertedSMACv2Sequences(SMACv2DatasetSpec(task=cfg['task'],split=cfg['split'],data_root=cfg['data_root']),20)
 sample=data.sample_numpy(32)
 assert sample['obs'].shape==(32,20,n,arrays['obs'].shape[-1]) and np.all(sample['obs'][...,-n:]==np.eye(n))
 starts=np.r_[0,np.cumsum(lengths[:-1])];returns=np.add.reduceat(arrays['rewards'][:,0].astype(np.float64),starts)
 assert abs(float(returns.mean())-manifest['selected']['mean_return'])<1e-6
 result=dict(status='verified',task=cfg['task'],split=cfg['split'],data_path=str(folder),env_info=info,
  episodes=len(lengths),rows=total,excluded_short_episodes=int((lengths<20).sum()),mean_dataset_team_return=float(returns.mean()),recorded_success_rate=1.,files=files,
  observation_transform='unchanged raw observation with existing suffix agent IDs',
  checks=dict(full_data_hashes=True,real_environment_dimensions=True,legal_actions=True,no_duplicate_agent_ids=True,episode_boundaries=True,all_selected_endings_true_terminal=True))
 atomic_json(Path(cfg['output_dir'])/'data_environment_audit.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
