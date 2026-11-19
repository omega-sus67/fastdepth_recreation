#!/usr/bin/env python3
"""Prepare fixed real evaluation tensors for offline Jetson testing."""
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
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data/nyu_depth_v2/extracted/val/official')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    paths = sorted(a.data.glob('*.h5'))
    if a.limit is not None:
        if a.limit <= 0: p.error('limit must be positive')
        paths = paths[:a.limit]
    if not paths or a.output.exists(): p.error('no samples or output already exists')
    a.output.mkdir(parents=True)
    x = np.lib.format.open_memmap(a.output / 'inputs.npy', mode='w+', dtype=np.float32, shape=(len(paths), 3, 224, 224))
    y = np.lib.format.open_memmap(a.output / 'targets.npy', mode='w+', dtype=np.float32, shape=(len(paths), 1, 224, 224))
    source = []
    for i, path in enumerate(paths):
        rgb, depth = load_h5(path)
        image, target = preprocess(rgb, depth)
        x[i] = image.numpy(); y[i] = target.numpy()
        source.append({'name': path.name, 'sha256': sha(path)})
        if (i + 1) % 100 == 0: print(f'{i+1}/{len(paths)}', flush=True)
    x.flush(); y.flush(); del x, y
    report = {'samples': len(paths), 'sources': source, 'dtype': 'float32',
              'inputs_shape': [len(paths), 3, 224, 224], 'targets_shape': [len(paths), 1, 224, 224],
              'inputs_sha256': sha(a.output / 'inputs.npy'), 'targets_sha256': sha(a.output / 'targets.npy'),
              'preprocessing': 'project paired nearest-neighbor preprocessing; RGB divided by 255',
              'units': 'native, unverified', 'split': '654 local evaluation files; official identity not fully verified'}
    (a.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Prepared {len(paths)} samples in {a.output}')

if __name__ == '__main__': main()
