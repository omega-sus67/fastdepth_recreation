#!/usr/bin/env python3
"""Prepare only the scene-grouped tuning split for target-side model selection."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fastdepth_project.data import load_h5, preprocess

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): p.error('output exists')
    cfg = json.loads(a.config.read_text()); source = ROOT / cfg['manifest']; audit = ROOT / cfg['overlap_report']
    manifest = json.loads(source.read_text()); overlap = json.loads(audit.read_text())
    if overlap['manifest_sha256'] != sha(source) or not overlap['train_tune_scene_disjoint'] or overlap['exact_rgb_overlap_with_evaluation'] or overlap['exact_rgb_overlap_train_tune_count']:
        raise ValueError('Data overlap audit failed')
    rows = manifest['groups']['tune']['samples']; n = len(rows)
    if not n: raise ValueError('Empty tuning set')
    a.output.mkdir(parents=True)
    x = np.lib.format.open_memmap(a.output/'inputs.npy', mode='w+', dtype=np.float32, shape=(n,3,224,224))
    y = np.lib.format.open_memmap(a.output/'targets.npy', mode='w+', dtype=np.float32, shape=(n,1,224,224))
    sources = []
    for i, row in enumerate(rows):
        path = ROOT / row['path']
        if sha(path) != row['sha256']: raise ValueError(f'Sample hash mismatch: {path}')
        rgb, depth = load_h5(path); image, target = preprocess(rgb, depth)
        x[i], y[i] = image.numpy(), target.numpy()
        sources.append({'name': row['path'], 'sha256': row['sha256'], 'scene': Path(row['source_member']).parts[-2]})
    x.flush(); y.flush(); del x, y
    report = {'samples': n, 'sources': sources, 'inputs_sha256': sha(a.output/'inputs.npy'),
              'targets_sha256': sha(a.output/'targets.npy'), 'source_manifest_sha256': sha(source),
              'config_sha256': sha(a.config), 'split': 'already-inspected scene-grouped tuning only',
              'preprocessing': 'project paired nearest-neighbor preprocessing; RGB divided by 255',
              'units': 'native, unverified', 'limitations': 'Scene identity and checkpoint pretraining independence are not proven.'}
    (a.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Prepared {n} tuning pairs.')

if __name__ == '__main__': main()
