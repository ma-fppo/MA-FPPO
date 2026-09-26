"""Original Torch MAC-Flow learner, 1M offline updates, with saved zero checkpoint."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from common import atomic_json, sha256


def group_norm(model, prefix, gradients=False):
    return math.sqrt(sum(float((p.grad if gradients else p).detach().float().square().sum())
                         for n,p in model.named_parameters() if n.startswith(prefix) and (not gradients or p.grad is not None)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    cfg=json.loads(Path(parser.parse_args().config).read_text())
    sys.path.insert(0,str(Path(cfg['project_root'])/'mac_flow_pytorch_smacv2'))
    from macflow_torch.agent import MACFlowLearner
    from macflow_torch.models import MACFlowConfig
    from smac_data import PoorSequences
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=False)
    assert cfg['steps'] in (500,2000,1000000)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    assert torch.cuda.is_available()
    dataset=PoorSequences(cfg['data_root'],cfg['task'],cfg['sequence_length'],split=cfg['split'])
    model_cfg=MACFlowConfig(obs_dim=dataset.obs_dim,num_agents=dataset.num_agents,action_dim=dataset.action_dim)
    learner=MACFlowLearner(model_cfg,torch.device('cuda'))
    initial={k:v.detach().cpu().clone() for k,v in learner.model.state_dict().items()}
    config=dict(cfg,model_config=vars(model_cfg),dataset_audit=cfg['data_audit'],
                source_learner_sha256=sha256(Path(cfg['project_root'])/'mac_flow_pytorch_smacv2/macflow_torch/agent.py'))
    atomic_json(out/'run_config.json',config)
    started=time.monotonic();log=(out/'metrics.jsonl').open('w',buffering=1)
    def emit(kind,**fields):
        r=dict(kind=kind,task=cfg['task'],elapsed_s=time.monotonic()-started,**fields)
        line=json.dumps(r,allow_nan=False);print(line,flush=True);log.write(line+'\n')
    def save(step):
        path=out/('checkpoint_final.pt' if step==cfg['steps'] else 'checkpoint_step_%d.pt'%step)
        temp=path.with_suffix('.tmp');learner.save(str(temp),extra=dict(step=step,task=cfg['task'],args=cfg,
            numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate()))
        temp.replace(path);return path
    def evaluate(step,path):
        ev=dict(cfg['ppo_config'],checkpoint=str(path),output_dir=str(out/('eval_step_%d'%step)))
        cf=out/('eval_config_%d.json'%step);atomic_json(cf,ev)
        with (out/('eval_step_%d.log'%step)).open('w') as handle:
            subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_mechanism_checkpoint.py')),
                '--config',str(cf),'--env-info',cfg['data_audit']],stdout=handle,stderr=subprocess.STDOUT,check=True)
        status=json.loads((Path(ev['output_dir'])/'status.json').read_text());assert status['status']=='complete'
        for line in (Path(ev['output_dir'])/'evaluation.jsonl').read_text().splitlines():
            r=json.loads(line);r.pop('checkpoint',None)
            emit('eval',step=step,checkpoint=str(path),checkpoint_sha256=sha256(path),eval_seed=ev['eval_seed'],**{k:v for k,v in r.items() if k!='task'})
    try:
        zero=save(0)
        if cfg.get('evaluate',True):evaluate(0,zero)
        for step in range(1,cfg['steps']+1):
            metrics=learner.train_step(dataset.sample_torch(cfg['batch_size'],torch.device('cuda')))
            if not all(math.isfinite(v) for v in metrics.values()):raise FloatingPointError(metrics)
            if step==1 or step%cfg['log_interval']==0 or step==cfg['steps']:
                grads={part:group_norm(learner.model,part,True) for part in ['encoder.','actor_bc_flow.','actor_onestep_flow.','q.']}
                if not all(math.isfinite(v) and v>0 for v in grads.values()):raise FloatingPointError(grads)
                emit('train',step=step,gradient_norms=grads,**metrics)
                atomic_json(out/'progress.json',dict(status='running',step=step,elapsed_s=time.monotonic()-started,metrics=metrics,gradient_norms=grads))
            if step%cfg['save_interval']==0 or step==cfg['steps']:
                path=save(step)
                if cfg.get('evaluate',True):evaluate(step,path)
                if (out/'stop_after_checkpoint').exists():
                    atomic_json(out/'status.json',dict(status='paused',step=step,checkpoint=str(path)));return
        final=learner.model.state_dict()
        delta={part:math.sqrt(sum(float((v.detach().cpu()-initial[k]).square().sum()) for k,v in final.items() if k.startswith(part))) for part in ['encoder.','actor_bc_flow.','actor_onestep_flow.','q.']}
        assert all(v>0 and math.isfinite(v) for v in delta.values())
        assert all(bool(torch.isfinite(v).all()) for v in final.values())
        restored=torch.load(path,map_location='cpu');assert restored['step']==cfg['steps']
        assert all(torch.equal(v.detach().cpu(),restored['model'][k]) for k,v in final.items())
        optimizer_steps=sorted({int(s['step']) for s in learner.optim.state.values() if 'step' in s})
        assert optimizer_steps==[cfg['steps']]
        result=dict(status='complete',step=cfg['steps'],checkpoint=str(path),checkpoint_sha256=sha256(path),
            parameter_deltas=delta,optimizer_steps=optimizer_steps,all_parameters_finite=True,roundtrip_exact=True,elapsed_s=time.monotonic()-started)
        atomic_json(out/'status.json',result);emit('complete',**{k:v for k,v in result.items() if k!='elapsed_s'})
    except BaseException as e:
        atomic_json(out/'status.json',dict(status='failed',error=repr(e)));raise
    finally:log.close()


if __name__=='__main__':main()
