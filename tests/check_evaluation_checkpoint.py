from pathlib import Path
import importlib.util,tempfile,torch
from types import SimpleNamespace
R=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
for family in ['smac','smacv2']:
 spec=importlib.util.spec_from_file_location('evaluation_checkpoint',R/'main_experiments/runtime'/family/'evaluation_checkpoint.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 cfg=dict(variant='torch',flow_steps=8,temperature=1.,torch_history='recurrent',train_encoder=False,student_latent_std=0.)
 target=SimpleNamespace(device='cpu',cfg=dict(cfg,checkpoint='/new/location'))
 for key in ['model','actor','reference','critic','reference_encoder']:setattr(target,key,torch.nn.Linear(3,2))
 original={k:torch.nn.Linear(3,2).state_dict() for k in ['model','actor','reference','critic','reference_encoder']};payload=dict(original,format='macflow_ppo_v1',config=dict(cfg,checkpoint='/old/location'),mechanism_format=1)
 called=[];target.reset=lambda:called.append(True)
 with tempfile.TemporaryDirectory() as d:
  p=Path(d)/'model.pt';torch.save(payload,p);m.restore_evaluation_policy(target,p)
  for key in original:
   for k,v in original[key].items():assert torch.equal(v,getattr(target,key).state_dict()[k])
  assert called
  target.cfg['temperature']=2.
  try:m.restore_evaluation_policy(target,p)
  except ValueError as e:assert 'temperature' in str(e)
  else:raise AssertionError('mismatched policy semantics accepted')
 print(family,'portable restore and mismatch rejection PASS')
