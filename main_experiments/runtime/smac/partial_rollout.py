"""Exact partial final rollout while preserving inactive recurrent/environment state."""
import numpy as np


def act_prefix(backend,env,count):
    if count==env.n_envs:
        return backend.act_batch(env.obs,env.legal,env.resets)
    assert 0<count<env.n_envs and not backend.encoder_training
    saved={}
    for name in ['carry','reference_carry']:
        full=getattr(backend,name)
        saved[name]=full
        if full is not None:
            assert all(x.shape[1]==env.n_envs*backend.n_agents for x in full)
            setattr(backend,name,tuple(x[:,:count*backend.n_agents].clone() for x in full))
    try:
        actions,cache=backend.act_batch(env.obs[:count],env.legal[:count],env.resets[:count])
        for name,full in saved.items():
            partial=getattr(backend,name)
            if full is not None:
                merged=tuple(x.clone() for x in full)
                for target,value in zip(merged,partial):target[:,:count*backend.n_agents]=value
                setattr(backend,name,merged)
            else:
                # Only the last rollout is partial, after full recurrent state exists.
                assert partial is None,name
        return actions,cache
    except BaseException:
        for name,value in saved.items():setattr(backend,name,value)
        raise


def step_prefix(env,actions):
    count=len(actions)
    if count==env.n_envs:return env.step(actions)
    assert 0<count<env.n_envs
    active=np.arange(env.n_envs)<count
    previous_resets=env.resets.copy()
    padded=np.zeros((env.n_envs,*actions.shape[1:]),dtype=actions.dtype)
    padded[:count]=actions
    result=env.step(padded,active)
    env.resets[count:]=previous_resets[count:]
    return {key:value[:count] for key,value in result.items()}
