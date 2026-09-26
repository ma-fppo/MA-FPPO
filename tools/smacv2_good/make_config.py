"""Create portable settings for the unchanged top-quarter collection procedure."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--task',required=True,choices=['terran_5_vs_5','zerg_5_vs_5']);p.add_argument('--replay-dir',type=Path,required=True);p.add_argument('--pretrained-checkpoint',type=Path,required=True);p.add_argument('--online-checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--config-out',type=Path,required=True);p.add_argument('--preflight-report',type=Path);p.add_argument('--preflight',action='store_true');p.add_argument('--n-envs',type=int,default=8);p.add_argument('--min-free-disk-gib',type=float,default=20);a=p.parse_args()
root=Path(__file__).resolve().parents[2];rt=root/'main_experiments/runtime/smacv2';lengths_path=a.replay_dir.resolve()/'path_lengths.npy';lengths=np.load(lengths_path);keep=8 if a.preflight else len(lengths)
if not a.preflight and not a.preflight_report:p.error('Run a preflight first and pass its selection_report.json')
c=dict(task=a.task,mode='preflight' if a.preflight else 'formal',original_episodes=len(lengths),original_transitions=int(lengths.sum()),original_path_lengths=str(lengths_path),target_episodes=4*keep,keep_episodes=keep,seed_start=1000000,sc2_seed_start=5000000,n_envs=a.n_envs,shard_episodes=256,min_free_disk_gib=a.min_free_disk_gib,backend_dir=str(rt),project_root=str(rt/'project'),ppo_checkpoint=str(a.online_checkpoint.resolve()),ppo_sha256=sha(a.online_checkpoint),pretrained_checkpoint=str(a.pretrained_checkpoint.resolve()),pretrained_sha256=sha(a.pretrained_checkpoint),source_hashes={str(x):sha(x) for x in [rt/'common.py',rt/'torch_backend.py',rt/'sc2_port_lease.py']},output_dir=str(a.output_dir.resolve()),preflight_report=str(a.preflight_report.resolve()) if a.preflight_report else None,collector_hashes={name:sha(Path(__file__).parent/name) for name in ['collect.py','select_good.py','dataset_io.py']})
a.config_out.parent.mkdir(parents=True,exist_ok=True);a.config_out.write_text(json.dumps(c,indent=2)+'\n')
