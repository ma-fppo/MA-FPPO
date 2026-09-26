"""Original two-agent Ant-v2 with proper seeding and terminal/time-limit flags."""
import multiprocessing as mp
import os
from pathlib import Path
import sys
import traceback
sys.path.insert(0,str(Path(__file__).parent/'vendor'))
import numpy as np
import gym
from multiagent_mujoco.mujoco_multi import MujocoMulti

ENV_ARGS=dict(scenario='Ant-v2',agent_conf='2x4',agent_obsk=1,episode_limit=1000,global_categories='qvel,qpos')

class TerminalProbe(gym.Wrapper):
    """Capture Ant's true done before legacy Gym TimeLimit overwrites it."""
    def reset(self,**kwargs):
        self.last_terminated=False
        return self.env.reset(**kwargs)
    def step(self,action):
        obs,reward,done,info=self.env.step(action);self.last_terminated=bool(done)
        return obs,reward,done,info

def make_env(seed):
    env=MujocoMulti(env_args=ENV_ARGS,add_agent_ids_to_obs=True)
    env.terminal_probe=TerminalProbe(env.env);env.timelimit_env.env=env.terminal_probe
    # MujocoMulti.seed is a no-op: seed the underlying Gym RNG directly.
    env.wrapped_env.seed(int(seed));env.reset()
    assert env.get_obs().shape==(2,56) and env.get_state().shape==(111,)
    return env


class Batch:
    def __init__(self, task, root, seeds):
        assert task=='2ant';self.envs=[make_env(seed) for seed in seeds]
        self.current=np.array([e.get_obs() for e in self.envs],dtype=np.float32)
        self.states=np.array([e.get_state() for e in self.envs],dtype=np.float32)

    def obs(self):return self.current

    def step(self, actions, active=None):
        count=len(actions)
        assert 0<count<=len(self.envs) and actions.shape==(count,2,4) and np.isfinite(actions).all() and abs(actions).max()<=1.000001
        active=np.ones(count,bool) if active is None else active
        assert active.shape==(count,)
        final_states=self.states.copy();rewards=np.zeros((len(self.envs),2),np.float32)
        dones=np.zeros(len(self.envs),np.float32);terminals=np.zeros(len(self.envs),np.float32)
        for i,e in enumerate(self.envs[:count]):
            if not active[i]:continue
            obs,rew,done,info=e.step(actions[i])
            assert bool(done[0])==bool(done[1])
            ended=bool(done[0]);terminal=bool(e.terminal_probe.last_terminated)
            assert not terminal or ended
            final_states[i]=e.get_state();rewards[i]=rew;dones[i]=ended;terminals[i]=terminal
            if ended:obs=e.reset()
            self.current[i]=obs;self.states[i]=e.get_state()
        assert np.isfinite(self.current).all() and np.isfinite(rewards).all()
        return self.current[:count].copy(),self.states[:count].copy(),final_states[:count],rewards[:count],dones[:count],terminals[:count]

    def close(self):
        # MujocoMulti.close is unimplemented; close its real Gym environment.
        for e in self.envs:e.wrapped_env.close()


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
            parent,child=ctx.Pipe();p=ctx.Process(target=worker,args=(child,task,root,seeds.tolist()),daemon=True)
            p.start();child.close();self.conns.append(parent);self.processes.append(p)
        initial=[self.receive(c) for c in self.conns];self.current,self.states=[np.concatenate(v) for v in zip(*initial)]

    def receive(self,c):
        if not c.poll(180):raise TimeoutError('MA-MuJoCo worker timeout')
        status,result=c.recv()
        if status!='ok':raise RuntimeError(result)
        return result

    def step(self,actions):
        count=len(actions);assert 0<count<=len(self.current)
        if count==len(self.current):
            for c,a in zip(self.conns,np.array_split(actions,len(self.conns))):c.send(('step',a))
            selected=self.conns
        else:
            width=len(self.current)//len(self.conns);selected=[]
            for i,c in enumerate(self.conns):
                start=i*width
                if start>=count:break
                c.send(('step',actions[start:min(start+width,count)]));selected.append(c)
        result=tuple(np.concatenate(v) for v in zip(*[self.receive(c) for c in selected]))
        if count==len(self.current):self.current,self.states=result[:2]
        else:self.current[:count],self.states[:count]=result[:2]
        return result

    def close(self):
        for c in self.conns:
            try:c.send(('close',None))
            except (BrokenPipeError,EOFError):pass
        for p in self.processes:
            p.join(3)
            if p.is_alive():p.terminate();p.join(3)
        for c in self.conns:c.close()
