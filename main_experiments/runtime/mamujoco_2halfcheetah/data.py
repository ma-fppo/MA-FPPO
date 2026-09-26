"""Original OG-MARL arrays; preserve existing leading agent IDs and all episodes."""
from pathlib import Path
import numpy as np
import torch


class Dataset:
    def __init__(self, root, task, split, seed=0):
        self.path = Path(root) / task / split
        self.arrays = {k: np.load(self.path/(k+'.npy'), mmap_mode='r') for k in ('obs','actions','rewards','discounts','path_lengths')}
        self.lengths = self.arrays['path_lengths']; self.ends = self.lengths.cumsum()-1
        self.starts = np.r_[0,self.ends[:-1]+1]; self.rng=np.random.RandomState(seed)
        self.obs_dim=self.arrays['obs'].shape[-1]
        assert self.obs_dim==15 and self.arrays['obs'].shape[1]==2 and self.ends[-1]+1==len(self.arrays['obs'])
        assert self.lengths.min()>0 and self.lengths.max()<=1000

    def sample_numpy(self, batch_size=32, sequence_length=20):
        # Uniform over all valid sequence starts, preserving within-episode next states.
        # Episodes shorter than a full sequence have no valid window. Preserve
        # the original arrays, but give these episodes zero sampling weight.
        choices=np.maximum(self.lengths-sequence_length+1,0); cumulative=choices.cumsum()
        if cumulative[-1]<=0: raise ValueError('No episodes long enough for sequence_length')
        pick=self.rng.randint(0,int(cumulative[-1]),batch_size)
        episodes=np.searchsorted(cumulative,pick,side='right')
        before=np.r_[0,cumulative[:-1]][episodes]
        starts=self.starts[episodes]+pick-before; rows=starts[:,None]+np.arange(sequence_length)
        at_end=rows==self.ends[episodes,None]
        terminal=at_end & (self.lengths[episodes,None]<1000)
        # Final observations are not stored. True-terminal TD needs no next state;
        # exclude only unknown time-limit TD targets, while keeping all FM data.
        critic_mask=~(at_end & (self.lengths[episodes,None]==1000))
        nxt=np.minimum(rows+1,self.ends[episodes,None])
        a=self.arrays
        result=dict(obs=a['obs'][rows],actions=a['actions'][rows],rewards=a['rewards'][rows],next_obs=a['obs'][nxt],
                    terminals=np.broadcast_to(terminal[...,None],(*terminal.shape,2)),critic_mask=critic_mask)
        return result, dict(rows=rows,next_rows=nxt,episodes=episodes,at_end=at_end)

    def sample(self, device, batch_size=32, sequence_length=20):
        result,_=self.sample_numpy(batch_size,sequence_length)
        return {k:torch.tensor(v,dtype=torch.float32,device=device) for k,v in result.items()}
