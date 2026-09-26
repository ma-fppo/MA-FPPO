"""Convert published OMIGA vaults without retaining reset delimiter records."""
import argparse,json,hashlib,zipfile
from pathlib import Path
import numpy as np
import tensorstore as ts
p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--task',choices=['3hopper','6halfcheetah','2ant']);args=p.parse_args()
ROOT=args.source_root.resolve()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
for task in ([args.task] if args.task else ['3hopper','6halfcheetah','2ant']):
 zip_path=ROOT/(task+'.zip');entry={'lfs':{'oid':sha(zip_path)}}
 if not (ROOT/(task+'.vlt')).exists():
  with zipfile.ZipFile(zip_path) as z:
   assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist());z.extractall(ROOT)
 for split in ['Expert','Medium','Medium-Replay','Medium-Expert']:
  source=ROOT/(task+'.vlt')/split;out=args.output_root.resolve()/task/split;out.mkdir(parents=True,exist_ok=True)
  if (out/'audit.json').exists():continue
  def read(name):return ts.open({'driver':'zarr','kvstore':{'driver':'ocdbt','base':'file://'+str(source)+'/','path':name}},open=True,read=True).result()
  n=int(read('vault_index').read().result()[0]);audit=dict(task=task,split=split,transitions=n,source_zip_sha256=entry['lfs']['oid'],arrays={})
  raw={}
  for src,dst in [('observations.','obs'),('actions.','actions'),('rewards.','rewards'),('terminals.','terminals'),('truncations.','truncations')]:
   a=np.asarray(read(src)[0,:n].read().result());assert np.isfinite(a).all();raw[dst]=a
  term=raw['terminals'];trunc=raw['truncations']
  assert np.array_equal(term,np.broadcast_to(term[:,:1],term.shape));assert np.array_equal(trunc,np.broadcast_to(trunc[:,:1],trunc.shape))
  # Published vault inserts an extra reset-observation/zero-action/zero-reward
  # delimiter after each trajectory. It is not an environment transition.
  markers=np.flatnonzero((term[:,0]>0)|(trunc[:,0]>0))
  assert np.all(raw['actions'][markers]==0) and np.all(raw['rewards'][markers]==0)
  keep=np.ones(n,bool);keep[markers]=False
  starts=np.r_[0,markers+1];stops=np.r_[markers,n];lengths=stops-starts;lengths=lengths[lengths>0]
  assert lengths.max()<=1000 and lengths.sum()==keep.sum()
  for key in ['obs','actions','rewards']:
   a=raw.pop(key)[keep];np.save(out/(key+'.npy'),a);audit['arrays'][key]=dict(shape=list(a.shape),min=float(a.min()),max=float(a.max()))
  clean_term=np.zeros((int(keep.sum()),term.shape[1]),np.float32);ends=lengths.cumsum()-1
  # As in the paper runtime, an episode ending before 1000 is a true termination.
  # No next observation is retained for timeouts: mask their final TD target.
  for i,L in enumerate(lengths):
   if L<1000 and (i<len(lengths)-1 or (len(markers)>0 and markers[-1]==n-1)):clean_term[ends[i]]=1
  np.save(out/'terminals.npy',clean_term);np.save(out/'truncations.npy',np.zeros_like(clean_term));np.save(out/'path_lengths.npy',lengths)
  obs=np.load(out/'obs.npy',mmap_mode='r')
  audit.update(episodes=len(lengths),valid_transitions=int(keep.sum()),removed_delimiter_rows=int(len(markers)),length_min=int(lengths.min()),length_max=int(lengths.max()),obs_abs_mean=float(np.abs(obs.mean(-1)).max()),obs_std_error=float(np.abs(obs.std(-1)-1).max()),status='passed',terminal_rule='known complete trajectories shorter than1000 terminal; missing timeout next observation masked')
  assert audit['obs_abs_mean']<1e-4 and audit['obs_std_error']<1e-4,audit
  (out/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
