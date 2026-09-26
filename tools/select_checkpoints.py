"""Select existing checkpoints by actual step count, including both endpoints."""
import argparse,json
from pathlib import Path
import numpy as np
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--count',type=int,default=20);a=p.parse_args()
 if a.count<2:p.error('--count must be at least 2')
 by_step={}
 for path in sorted(a.directory.glob('checkpoint*.pt')):
  if 'paused' in path.name or 'failure' in path.name:continue
  try:ck=torch.load(path,map_location='cpu',weights_only=False)
  except TypeError:ck=torch.load(path,map_location='cpu')
  step=ck.get('env_steps',ck.get('step',ck.get('steps')))
  if step is None:raise ValueError('No actual step count: '+str(path))
  by_step[int(step)]=path
 steps=np.array(sorted(by_step),dtype=np.int64)
 if len(steps)<a.count:raise ValueError(f'Need {a.count} distinct existing checkpoints; found {len(steps)}')
 targets=np.linspace(steps[0],steps[-1],a.count);chosen=[0]
 for i,target in enumerate(targets[1:-1],1):
  lo=chosen[-1]+1;hi=len(steps)-(a.count-i-1)
  chosen.append(lo+int(np.abs(steps[lo:hi]-target).argmin()))
 chosen.append(len(steps)-1)
 print(json.dumps([dict(step=int(steps[i]),checkpoint=str(by_step[int(steps[i])].resolve())) for i in chosen],indent=2))
if __name__=='__main__':main()
