"""OMIGA/MAC-Flow fully observed MuJoCo; trailing ID, per-vector standardization.
Flat action concatenation matches the published OMIGA wrapper (not OG-MARL joints).
"""
import multiprocessing as mp
import traceback
import gym
import numpy as np
TASKS={'3hopper':('Hopper-v2',3,1,11),'6halfcheetah':('HalfCheetah-v2',6,1,17),'2ant':('Ant-v2',2,4,111)}

def observations(raw,n):
 x=np.concatenate([np.broadcast_to(raw,(n,len(raw))),np.eye(n,dtype=np.float32)],axis=-1).astype(np.float32)
 return (x-x.mean(-1,keepdims=True))/x.std(-1,keepdims=True)

class Batch:
 def __init__(self,task,root,seeds):
  name,self.n,self.a,self.s=TASKS[task];self.envs=[];raw=[]
  for seed in seeds:
   env=gym.make(name);env.seed(int(seed));o=env.reset();assert o.shape==(self.s,);self.envs.append(env);raw.append(o)
  self.states=np.asarray(raw,dtype=np.float32);self.current=np.asarray([observations(o,self.n) for o in raw]);self.elapsed=np.zeros(len(raw),int)
 def obs(self):return self.current
 def step(self,actions,active=None):
  count=len(actions);assert actions.shape==(count,self.n,self.a) and np.isfinite(actions).all() and np.abs(actions).max()<=1.000001
  active=np.ones(count,bool) if active is None else active;final=self.states.copy();rewards=np.zeros((len(self.envs),self.n),np.float32);dones=np.zeros(len(self.envs),np.float32);terms=dones.copy()
  for i,env in enumerate(self.envs[:count]):
   if not active[i]:continue
   # Published NormalizedActions maps [-1,1] into the base actuator range.
   flat=actions[i].reshape(-1);actual=(flat+1)*.5*(env.action_space.high-env.action_space.low)+env.action_space.low
   # Bypass Gym TimeLimit so simultaneous true termination at horizon is retained.
   o,r,terminal,info=env.unwrapped.step(actual);self.elapsed[i]+=1;done=bool(terminal or self.elapsed[i]>=1000)
   final[i]=o;rewards[i]=r;dones[i]=done;terms[i]=bool(terminal)
   if done:o=env.reset();self.elapsed[i]=0
   self.states[i]=o;self.current[i]=observations(o,self.n)
  assert np.isfinite(self.current).all() and np.isfinite(rewards).all()
  return self.current[:count].copy(),self.states[:count].copy(),final[:count],rewards[:count],dones[:count],terms[:count]
 def close(self):
  for e in self.envs:e.close()

def worker(conn,task,root,seeds):
 batch=None
 try:
  batch=Batch(task,root,seeds);conn.send(('ok',(batch.current,batch.states)))
  while True:
   cmd,value=conn.recv()
   if cmd=='close':break
   conn.send(('ok',batch.step(value)))
 except BaseException:
  try:conn.send(('error',traceback.format_exc()))
  except Exception:pass
 finally:
  if batch:batch.close()
  conn.close()

class Pool:
 def __init__(self,task,root,n_envs,workers,seed):
  assert n_envs%workers==0
  self.conns=[];self.processes=[];ctx=mp.get_context('spawn')
  for seeds in np.array_split(np.arange(n_envs)+seed,workers):
   parent,child=ctx.Pipe();p=ctx.Process(target=worker,args=(child,task,root,seeds.tolist()),daemon=True);p.start();child.close();self.conns.append(parent);self.processes.append(p)
  initial=[self.receive(c) for c in self.conns];self.current,self.states=[np.concatenate(v) for v in zip(*initial)]
 def receive(self,c):
  if not c.poll(180):raise TimeoutError('MuJoCo worker timeout')
  status,result=c.recv()
  if status!='ok':raise RuntimeError(result)
  return result
 def step(self,actions):
  count=len(actions);assert 0<count<=len(self.current);width=len(self.current)//len(self.conns);selected=[]
  for i,c in enumerate(self.conns):
   start=i*width
   if start>=count:break
   c.send(('step',actions[start:min(start+width,count)]));selected.append(c)
  result=tuple(np.concatenate(v) for v in zip(*[self.receive(c) for c in selected]));self.current[:count],self.states[:count]=result[:2];return result
 def close(self):
  for c in self.conns:
   try:c.send(('close',None))
   except (BrokenPipeError,EOFError):pass
  for p in self.processes:
   p.join(3)
   if p.is_alive():p.terminate();p.join(3)
  for c in self.conns:c.close()
