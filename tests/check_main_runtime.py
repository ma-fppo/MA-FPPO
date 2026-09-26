"""Finite losses, gradients, fixed-latent inheritance, raw density, and GAE."""
import argparse,inspect,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[1];rt=ROOT/'main_experiments/runtime'/a.runtime
sys.path.insert(0,str(rt));torch.set_num_threads(1);torch.manual_seed(7)
if a.runtime in ('smac','smacv2'):
 sys.path.insert(0,str(rt/'project/mac_flow_pytorch_smacv2'))
 from common import compute_gae,normalize_advantages
 from torch_backend import masked_categorical
 from macflow_torch.agent import MACFlowLearner
 from macflow_torch.models import MACFlowConfig
 cfg=MACFlowConfig(obs_dim=9,num_agents=3,action_dim=5)
 learner=MACFlowLearner(cfg,torch.device('cpu'))
 obs=torch.randn(2,4,3,9);obs[...,-3:]=torch.eye(3)
 actions=torch.randint(0,5,(2,4,3));batch=dict(obs=obs,actions=actions,rewards=torch.randn(2,4,3),legals=torch.ones(2,4,3,5),terminals=torch.zeros(2,4,3));batch['terminals'][:,-1]=1
 before=next(learner.model.actor_onestep_flow.parameters()).detach().clone();metrics=learner.train_step(batch)
 assert all(np.isfinite(v) for v in metrics.values());assert not torch.equal(before,next(learner.model.actor_onestep_flow.parameters()))
 logits=torch.randn(2,3,5,requires_grad=True);legal=torch.ones_like(logits,dtype=torch.bool);legal[...,2]=False
 dist=masked_categorical(logits,legal,1.);assert torch.all(dist.probs[...,2]==0)
 rewards=np.array([[1.,2.],[3.,4.]],np.float32);zeros=np.zeros_like(rewards);term=np.array([[False,True],[True,False]]);trunc=np.array([[False,False],[False,True]])
 adv,returns=compute_gae(rewards,zeros,zeros,term,trunc,gamma=.9,lam=.8)
 assert np.allclose(adv,[[3.16,2],[3,4]])
 assert np.isfinite(normalize_advantages(np.ones((4,2)))).all()
else:
 from model import ContinuousMACFlow,GaussianStudent,gae
 kw=dict(hidden=(16,16));sig=inspect.signature(ContinuousMACFlow)
 if 'action_dim' in sig.parameters:kw['action_dim']=2
 model=ContinuousMACFlow(9,**kw)
 obs=torch.randn(2,4,3,9);act=torch.randn(2,4,3,2).clamp(-1,1)
 batch=dict(obs=obs,next_obs=torch.randn_like(obs),actions=act,rewards=torch.randn(2,4,3),terminals=torch.zeros(2,4,3),critic_mask=torch.ones(2,4,dtype=torch.bool));batch['terminals'][:,-1]=1
 opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-3)
 before=next(model.student.parameters()).detach().clone();metrics=model.learn(batch,opt)
 assert not torch.equal(before,next(model.student.parameters()))
 assert all(p.grad is None for p in model.target_q.parameters())
 akw={'action_dim':2} if 'action_dim' in inspect.signature(GaussianStudent).parameters else {}
 actor=GaussianStudent(model.student,std=.2,**akw);o=obs[:,0];dist=actor(o)
 torch.testing.assert_close(dist.mean,model.student_action(o))
 raw=dist.sample();manual=-.5*((raw-dist.mean)/dist.stddev).square()-dist.stddev.log()-.5*np.log(2*np.pi)
 torch.testing.assert_close(dist.log_prob(raw).sum(-1),manual.sum(-1))
 before_mean=dist.mean.detach().clone();optimizer=torch.optim.Adam(actor.parameters(),lr=.001);loss=-actor(o).log_prob(raw.detach()).mean();optimizer.zero_grad();loss.backward();optimizer.step();assert not torch.equal(before_mean,actor(o).mean)
 rewards=torch.tensor([[1.,2.],[3.,4.]]);zeros=torch.zeros_like(rewards);done=torch.tensor([[0.,1.],[1.,1.]])
 adv,returns=gae(rewards,zeros,zeros,done,.9,.8);torch.testing.assert_close(adv,torch.tensor([[3.16,2.],[3.,4.]]))
print(a.runtime+': PASS')
