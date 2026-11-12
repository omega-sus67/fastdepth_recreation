#!/usr/bin/env python3
"""Inspect NYU validation archives without extracting them."""
import argparse
import io
from pathlib import Path
import tarfile

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data', type=Path, default=root / 'data/nyu_depth_v2/raw')
parser.add_argument('--output', type=Path, default=root / 'results/validation_sample.png')
args = parser.parse_args()
archives = sorted(args.data.glob('val-*.tar'))
if not archives:
    parser.error(f'No val-*.tar archives found in {args.data}')
names = set()
first = None
for archive in archives:
    with tarfile.open(archive) as tar:
        members = [m for m in tar if m.isfile() and m.name.endswith('.h5')]
        print(f'{archive.name}: {len(members)} HDF5 samples')
        for member in members:
            if member.name in names:
                raise ValueError(f'Duplicate archive member: {member.name}')
            names.add(member.name)
        if first is None and members:
            first = (members[0].name, tar.extractfile(members[0]).read())
print(f'Total samples: {len(names)}')
if first is None:
    raise SystemExit('No HDF5 samples found')
name, raw = first
with h5py.File(io.BytesIO(raw), 'r') as f:
    print(f'Example {name}:')
    for key in f:
        print(f'  {key}: shape={f[key].shape}, dtype={f[key].dtype}')
    rgb, depth = np.array(f['rgb']), np.array(f['depth'])
if rgb.ndim == 3 and rgb.shape[0] == 3:
    rgb = rgb.transpose(1, 2, 0)
valid = depth[np.isfinite(depth) & (depth > 0)]
print(f'Positive finite depth: {valid.min():.3f} to {valid.max():.3f}; units require verification')
args.output.parent.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(rgb); axes[0].set_title('RGB')
im = axes[1].imshow(depth, cmap='magma'); axes[1].set_title('Depth (raw units)')
for ax in axes: ax.axis('off')
fig.colorbar(im, ax=axes[1]); fig.tight_layout(); fig.savefig(args.output, dpi=130)
print(f'Saved {args.output}')
