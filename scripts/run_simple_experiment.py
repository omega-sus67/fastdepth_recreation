#!/usr/bin/env python3
"""Run a matched baseline, pruning and recovery comparison on tuning data."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/experiment_2026-11-16_simple.json')
    parser.add_argument('--output', type=Path, required=True, help='New directory for this run, inside the project')
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        output.relative_to(ROOT)
    except ValueError:
        parser.error('output must be inside the project so comparison paths are portable')
    if output.exists():
        parser.error('output already exists; choose a new run directory')
    config = json.loads(args.config.read_text())
    if config['pruning']['stage'] != 'first' or config['training']['epochs'] != 1:
        parser.error('only first-decoder pruning and one-epoch recovery are supported')
    seed = config['training']['seeds'][0]
    baseline = ROOT / config['baseline']
    candidate = output / 'pruned/candidate.pth'
    supervised = output / 'supervised/checkpoint.pth'
    distilled = output / 'distilled/checkpoint.pth'
    output.mkdir(parents=True)
    env = os.environ.copy()
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    def run(script, *arguments):
        command = [sys.executable, str(ROOT / 'scripts' / script), *map(str, arguments)]
        print('Running:', ' '.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    run('prune_candidate.py', '--checkpoint', baseline, '--stage', 'first', '--keep',
        config['pruning']['keep'], '--output', output / 'pruned')
    for name, weight in [('supervised', 0.0), ('distilled', config['distill_weight'])]:
        run('train_distill.py', '--config', args.config.resolve(), '--checkpoint', candidate,
            '--teacher', baseline, '--distill-weight', weight, '--seed', seed,
            '--deterministic', '--output', output / name)
    models = [('baseline', baseline), ('pruned_before_recovery', candidate),
              ('supervised_control', supervised), ('distilled_student', distilled)]
    comparison = {'manifest': config['manifest'], 'overlap_report': config['overlap_report'], 'group': 'tune',
                  'models': [{'name': name, 'checkpoint': str(path.relative_to(ROOT)), 'description': name}
                             for name, path in models], 'limitations': config['limitations']}
    comparison_path = output / 'comparison_config.json'
    comparison_path.write_text(json.dumps(comparison, indent=2) + '\n')
    run('compare_tuning_models.py', '--config', comparison_path, '--output', output / 'comparison')
    print('Completed tuning comparison:', output / 'comparison' / 'summary.json')
    print('Select a student using the declared protocol; final evaluation and Orin testing remain separate.')


if __name__ == '__main__':
    main()
