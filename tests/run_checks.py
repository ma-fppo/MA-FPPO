"""Run environment-free mathematical tests and validate every release recipe."""
import ast,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in ROOT.rglob('*.py'):ast.parse(p.read_text(),filename=str(p.relative_to(ROOT)))
for p in ROOT.glob('*_experiments/configs/*.json'):json.loads(p.read_text())
for rt in ('mpe','mamujoco_2ant','mamujoco_4ant','mamujoco_2halfcheetah','omiga','smac','smacv2'):
 subprocess.run([sys.executable,str(ROOT/'tests/check_main_runtime.py'),'--runtime',rt],check=True,cwd=ROOT)
subprocess.run([sys.executable,str(ROOT/'tests/check_evaluation_checkpoint.py')],check=True,cwd=ROOT)
subprocess.run([sys.executable,str(ROOT/'tests/check_smac_factory.py')],check=True,cwd=ROOT)
print('All source and CPU semantic checks passed.')
