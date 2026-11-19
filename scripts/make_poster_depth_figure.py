#!/usr/bin/env python3
"""Render a truthful, shared-scale poster figure from saved depth predictions."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from fastdepth_project.data import preprocess


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample', default='photo_0001_00001', help='Photo basename in the saved manifest')
    p.add_argument('--photos', type=Path, default=ROOT / 'jetson_handoff/visual_photos')
    p.add_argument('--predictions', type=Path, default=ROOT / 'results/jetson_followup_2026-11-16/visual-comparison')
    p.add_argument('--selection', type=Path, default=ROOT / 'results/jetson_orin_2026-11-16/results/orin-sweep-visit/selection.json')
    p.add_argument('--output', type=Path, required=True, help='New output directory')
    a = p.parse_args()
    if a.output.exists(): p.error('output directory already exists')
    photos_manifest = json.loads((a.photos / 'manifest.json').read_text())
    predictions_manifest = json.loads((a.predictions / 'manifest.json').read_text())
    selection = json.loads(a.selection.read_text())
    if selection.get('selected_student') != 'width176' or predictions_manifest.get('selected_student') != 'width176':
        raise ValueError('Expected the recorded width176 selection')
    if sha(a.selection) != predictions_manifest['selection_sha256'] or sha(a.photos / 'manifest.json') != predictions_manifest['photo_manifest_sha256']:
        raise ValueError('Selection or photo-manifest fingerprint mismatch')
    photo = next((row for row in photos_manifest['samples'] if Path(row['photo']).stem == a.sample), None)
    pred = next((row for row in predictions_manifest['samples'] if Path(row['photo']).stem == a.sample), None)
    if photo is None or pred is None or Path(a.sample).name != a.sample: p.error('sample not found')
    photo_path = a.photos / photo['photo']
    if sha(photo_path) != photo['photo_sha256']: raise ValueError('Photo fingerprint mismatch')
    with Image.open(photo_path) as im:
        rgb = np.asarray(im.convert('RGB'))
    network_rgb = preprocess(rgb).permute(1, 2, 0).numpy()
    baseline_path = a.predictions / f'{a.sample}_baseline.npy'
    student_path = a.predictions / f'{a.sample}_width176.npy'
    baseline = np.load(baseline_path, allow_pickle=False)
    student = np.load(student_path, allow_pickle=False)
    if baseline.shape != (224, 224) or student.shape != baseline.shape or not np.isfinite(baseline).all() or not np.isfinite(student).all():
        raise ValueError('Invalid prediction arrays')
    low = float(min(baseline.min(), student.min())); high = float(max(baseline.max(), student.max()))
    if high <= low: high = low + 1e-6
    a.output.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0), gridspec_kw={'wspace': 0.07})
    fig.patch.set_facecolor('white')
    axes[0].imshow(network_rgb, interpolation='nearest')
    axes[1].imshow(baseline, cmap='magma', vmin=low, vmax=high, interpolation='nearest')
    depth_img = axes[2].imshow(student, cmap='magma', vmin=low, vmax=high, interpolation='nearest')
    for ax, label, subtitle in zip(axes, ('Model input', 'Baseline', 'Selected student'),
                                    ('RGB after model crop', 'Already-pruned baseline', '176-channel decoder')):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_title(label, fontsize=16, fontweight='bold', color='#193751', pad=10)
        ax.text(.5, -0.035, subtitle, transform=ax.transAxes, ha='center', va='top', fontsize=10, color='#42596c')
    fig.subplots_adjust(left=.035, right=.975, top=.89, bottom=.31)
    colorbar_axis = fig.add_axes([.40, .17, .50, .028])
    cb = fig.colorbar(depth_img, cax=colorbar_axis, orientation='horizontal')
    cb.set_label('Predicted depth (native units unverified)', fontsize=10, labelpad=4)
    cb.ax.tick_params(labelsize=9, length=2)
    fig.text(.035, .025, 'Same colour scale for both maps · PyTorch predictions · one previously inspected indoor image', fontsize=9, color='#42596c')
    fig.savefig(a.output / 'depth_comparison.png', dpi=300, facecolor='white')
    fig.savefig(a.output / 'depth_comparison.pdf', facecolor='white')
    plt.close(fig)
    record = {
        'scope': 'qualitative saved PyTorch predictions, not a new inference or accuracy comparison',
        'sample': a.sample, 'source_h5': photo['original_h5'], 'photo_sha256': sha(photo_path),
        'baseline_prediction_sha256': sha(baseline_path), 'student_prediction_sha256': sha(student_path),
        'baseline_checkpoint_sha256': predictions_manifest['baseline_checkpoint_sha256'],
        'student_checkpoint_sha256': predictions_manifest['student_checkpoint_sha256'],
        'selection_sha256': sha(a.selection), 'photo_manifest_sha256': sha(a.photos / 'manifest.json'),
        'runtime_recorded_by_source': predictions_manifest['runtime'], 'device_recorded_by_source': predictions_manifest['device'],
        'runtime_host': 'not identified by source manifest', 'model_input_preprocessing': 'project preprocess: 640x480 RGB to 224x224 crop',
        'scale_min': low, 'scale_max': high, 'scale': 'common minimum and maximum from both arrays',
        'limits': 'Previously inspected NYU-derived image; physical depth units and pretraining overlap not verified; no TensorRT or speed claim.'}
    (a.output / 'metadata.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Created', a.output / 'depth_comparison.png')
    print('Created', a.output / 'depth_comparison.pdf')


if __name__ == '__main__': main()
