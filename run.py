#!/usr/bin/env python3
"""Portable three-training-seed entrypoint; never launches a scheduler."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def expand(value, roots):
    if isinstance(value, dict): return {k: expand(v, roots) for k, v in value.items()}
    if isinstance(value, list): return [expand(v, roots) for v in value]
    if isinstance(value, str):
        for key, path in roots.items(): value = value.replace('${' + key + '}', str(path))
        if '${' in value: raise ValueError('Unresolved configuration variable: ' + value)
    return value

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('method', choices=['main'])
    p.add_argument('--config', required=True, help='Path to a supplied JSON configuration')
    p.add_argument('--phase', choices=['audit', 'pretrain', 'online', 'all', 'evaluate'], default='all')
    p.add_argument('--data-root', type=Path, default=ROOT/'data')
    p.add_argument('--output-root', type=Path, default=ROOT/'runs')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--seed', type=int, default=10000, help='Evaluation seed, separate from training seeds')
    p.add_argument('--training-seeds', type=int, nargs='+', default=[0, 1, 2], help='Independent training seeds, run sequentially (default: 0 1 2)')
    p.add_argument('--training-seed', type=int, default=0, help='Training seed whose configuration is used for evaluation')
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--sc2path', type=Path, help='Installed StarCraft II root for the selected benchmark')
    p.add_argument('--dry-run', action='store_true', help='Validate and show commands without executing them')
    args = p.parse_args()
    if args.episodes < 1: p.error('--episodes must be positive')
    seeds = [args.training_seed] if args.phase == 'evaluate' else args.training_seeds
    if len(set(seeds)) != len(seeds) or any(seed < 0 or seed >= 2**32 for seed in seeds):
        p.error('Training seeds must be unique integers in [0, 2**32)')
    for training_seed in seeds:
        run_one(args, p, training_seed)

def set_training_seed(value, seed):
    if isinstance(value, dict):
        return {k: seed if k == 'seed' else set_training_seed(v, seed) for k, v in value.items()}
    if isinstance(value, list):
        return [set_training_seed(v, seed) for v in value]
    return value

def run_one(args, p, training_seed):
    output_root = args.output_root.resolve()/('seed_' + str(training_seed))
    roots = dict(PACKAGE_ROOT=ROOT, DATA_ROOT=args.data_root.resolve(), OUTPUT_ROOT=output_root)
    src = Path(args.config).resolve()
    c = set_training_seed(expand(json.loads(src.read_text()), roots), training_seed)
    work = output_root/'resolved_configs'/args.method/src.stem
    env = dict(os.environ, WANDB_MODE='disabled')
    if args.sc2path: env['SC2PATH'] = str(args.sc2path.resolve())
    def config_file(name, obj):
        path = work/(name + '.json')
        if not args.dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj, indent=2) + '\n')
        return str(path)
    commands = []
    rt = ROOT/'main_experiments/runtime'/c['runtime']
    if not rt.is_dir(): raise FileNotFoundError(rt)
    discrete = c['runtime'] in ('smac','smacv2')
    phases = (['audit','pretrain','online'] if discrete else ['pretrain','online']) if args.phase=='all' else [args.phase]
    if discrete:
        online = c['online']; pre = c['pretrain']; audit = c['audit']
        if args.phase=='evaluate':
            if not args.checkpoint: p.error('--checkpoint is required')
            online['eval_seed'] = args.seed; online['eval_episodes'] = args.episodes
            online['output_dir'] = str(output_root/'evaluations'/src.stem/args.checkpoint.stem/str(args.seed))
        paths = {k:config_file(k,v) for k,v in [('online',online),('pretrain',pre),('audit',audit)]}
        for phase in phases:
            if phase=='audit':
                commands.append([sys.executable,str(ROOT/'tools/check_discrete_data.py'),'--config',paths['audit'],'--runtime',str(rt)])
            elif phase=='pretrain':commands.append([sys.executable,str(rt/'pretrain_smac.py'),'--config',paths['pretrain']])
            elif phase=='online':commands.append([sys.executable,str(rt/'train_mechanism_ablation.py'),'--config',paths['online']])
            else:
                commands.append([sys.executable,str(rt/'evaluate_mechanism_checkpoint.py'),'--config',paths['online'],'--env-info',pre['data_audit'],'--checkpoint',str(args.checkpoint.resolve())])
    else:
        path = config_file('train', c['train'])
        for phase in phases:
            if phase=='audit': p.error('Continuous data contracts are checked when the dataset is loaded')
            elif phase in ('pretrain','online'):commands.append([sys.executable,str(rt/'train.py'),'--config',path,'--phase','ppo' if phase=='online' else phase])
            else:
                if not args.checkpoint: p.error('--checkpoint is required')
                out = output_root/'evaluations'/src.stem/(args.checkpoint.stem+'_'+str(args.seed)+'.json')
                if not args.dry_run:out.parent.mkdir(parents=True,exist_ok=True)
                commands.append([sys.executable,str(rt/'evaluate.py'),'--config',path,'--checkpoint',str(args.checkpoint.resolve()),'--output',str(out),'--episodes',str(args.episodes),'--seed',str(args.seed)])
    print(json.dumps({'configuration':src.name,'training_seed':training_seed,'commands':commands,'dry_run':args.dry_run},indent=2),flush=True)
    if not args.dry_run:
        for command in commands: subprocess.run(command,cwd=ROOT,env=env,check=True)

if __name__=='__main__':main()
