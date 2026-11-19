#!/usr/bin/env python3
"""Build a fixed, scene-diverse INT8 calibration tensor from training data only."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fastdepth_project.data import load_h5, preprocess


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''): h.update(part)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('output already exists')
    config = json.loads(a.config.read_text())
    source = ROOT / config['manifest']; overlap_path = ROOT / config['overlap_report']
    manifest = json.loads(source.read_text()); overlap = json.loads(overlap_path.read_text())
    if overlap['manifest_sha256'] != sha(source) or not overlap['train_tune_scene_disjoint'] or overlap['exact_rgb_overlap_with_evaluation'] or overlap['exact_rgb_overlap_train_tune_count']:
        raise ValueError('Training/tuning overlap audit failed')
    by_scene = {}
    for row in manifest['groups']['train']['samples']:
        by_scene.setdefault(Path(row['source_member']).parts[-2], []).append(row)
    eligible = sorted(name for name, rows in by_scene.items() if len(rows) >= 4)
    random.Random(42).shuffle(eligible)
    if len(eligible) < 32: raise ValueError('Need at least 32 eligible training scenes')
    chosen = []
    for scene in eligible[:32]:
        rows = sorted(by_scene[scene], key=lambda r: r['source_member'])
        for j in range(4):
            chosen.append((scene, rows[round(j * (len(rows) - 1) / 3)]))
    a.output.mkdir(parents=True)
    x = np.lib.format.open_memmap(a.output / 'inputs.npy', mode='w+', dtype=np.float32, shape=(len(chosen), 3, 224, 224))
    records = []
    for i, (scene, row) in enumerate(chosen):
        path = ROOT / row['path']
        if sha(path) != row['sha256']: raise ValueError(f'Sample hash mismatch: {path}')
        rgb, _ = load_h5(path); x[i] = preprocess(rgb).numpy()
        records.append({'scene': scene, 'path': row['path'], 'sha256': row['sha256']})
    x.flush(); del x
    report = {'scope': 'training-only INT8 calibration; no tuning or evaluation inputs',
              'config_sha256': sha(a.config), 'source_manifest_sha256': sha(source),
              'overlap_report_sha256': sha(overlap_path), 'selection': '32 eligible named scenes shuffled with seed 42; four evenly spaced samples per scene',
              'inputs_sha256': sha(a.output / 'inputs.npy'), 'shape': [len(chosen), 3, 224, 224], 'samples': records,
              'limitations': 'Nonuniform scene windows; exact checks cannot prove independence from initial checkpoint pretraining.'}
    (a.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Prepared {len(chosen)} training-only calibration inputs from {len(eligible[:32])} scenes.')


if __name__ == '__main__': main()
