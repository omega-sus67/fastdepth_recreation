#!/usr/bin/env python3
"""Create paired PyTorch depth visualizations for the board-selected student."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fastdepth_project.data import preprocess
from fastdepth_project.models import load_converted

CHECKPOINTS = {
    'width192': 'results/training/2026-11-16-simple-distilled-alpha01/checkpoint.pth',
    'width176': 'results/training/2026-11-16-latency-176/checkpoint.pth',
    'width160': 'results/training/2026-11-16-latency-160/checkpoint.pth',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True, help='Board sweep selection.json')
    p.add_argument('--images', type=Path, default=ROOT / 'visual_photos')
    p.add_argument('--output', type=Path, required=True, help='New output directory')
    p.add_argument('--device', default='cuda')
    p.add_argument('--limit', type=int, help='Optional small smoke test')
    a = p.parse_args()
    if a.output.exists() or (a.limit is not None and a.limit < 1):
        p.error('Output must not exist and limit must be positive')
    selection = json.loads(a.selection.read_text())
    selected = selection.get('selected_student')
    if selected not in CHECKPOINTS:
        p.error('No recognized student selected; inspect the sweep selection.json')
    manifest_path = a.images / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    records = manifest['samples'][:a.limit] if a.limit else manifest['samples']
    if not records: p.error('No photos in manifest')
    for entry in records:
        name = entry['photo']
        if Path(name).name != name or sha(a.images / name) != entry['photo_sha256']:
            raise ValueError('Photo path or hash does not match manifest: ' + name)
    baseline_path = ROOT / 'checkpoints/converted/community-skipadd-pruned.pth'
    student_path = ROOT / CHECKPOINTS[selected]
    baseline, _ = load_converted(baseline_path, a.device)
    student, _ = load_converted(student_path, a.device)
    baseline.eval(); student.eval()
    a.output.mkdir(parents=True)
    entries = []
    for item in records:
        path = a.images / item['photo']
        with Image.open(path) as im: rgb = np.asarray(im.convert('RGB'))
        x = preprocess(rgb).unsqueeze(0).to(a.device)
        with torch.inference_mode():
            base = baseline(x).cpu().numpy()[0, 0]
            child = student(x).cpu().numpy()[0, 0]
        if not np.isfinite(base).all() or not np.isfinite(child).all():
            raise ValueError('Nonfinite depth prediction: ' + item['photo'])
        stem = path.stem
        np.save(a.output / f'{stem}_baseline.npy', base)
        np.save(a.output / f'{stem}_{selected}.npy', child)
        low = float(min(base.min(), child.min())); high = float(max(base.max(), child.max()))
        if low == high: high = low + 1e-6
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        axes[0].imshow(rgb); axes[0].set_title('Original RGB (model uses center crop)')
        axes[1].imshow(base, cmap='magma', vmin=low, vmax=high); axes[1].set_title('Baseline')
        im = axes[2].imshow(child, cmap='magma', vmin=low, vmax=high); axes[2].set_title(selected)
        for ax in axes: ax.axis('off')
        fig.colorbar(im, ax=axes[1:].tolist(), label='Predicted depth (units unverified)', shrink=.8)
        fig.savefig(a.output / f'{stem}_comparison.png', dpi=160); plt.close(fig)
        entries.append({'photo': item['photo'], 'source_h5': item['original_h5'],
                        'comparison': f'{stem}_comparison.png', 'shared_color_min': low, 'shared_color_max': high})
        print(f'Created {stem}_comparison.png', flush=True)
    (a.output / 'manifest.json').write_text(json.dumps({
        'selected_student': selected, 'selection_sha256': sha(a.selection), 'photo_manifest_sha256': sha(manifest_path),
        'baseline_checkpoint_sha256': sha(baseline_path), 'student_checkpoint_sha256': sha(student_path),
        'runtime': 'PyTorch', 'device': a.device, 'samples': entries,
        'interpretation': 'Same color scale for both predictions within each photo; scale differs between photos. Qualitative comparison only; RGB view is uncropped while model input is center-cropped. Not TensorRT or a Jetson latency measurement.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
