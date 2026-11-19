#!/usr/bin/env python3
"""Convert a recorded RGB video to a colorized depth video using PyTorch."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import cv2
import numpy as np
import torch
from fastdepth_project.data import preprocess
from fastdepth_project.models import load_converted, sha256
from fastdepth_project.video import adapt_frame

CHECKPOINTS = {
    'baseline': ROOT / 'checkpoints/converted/community-skipadd-pruned.pth',
    'student': ROOT / 'results/training/width_176/checkpoint.pth',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path, help='Recorded input video')
    p.add_argument('--output', type=Path, required=True, help='New output directory')
    choice = p.add_mutually_exclusive_group()
    choice.add_argument('--model', choices=tuple(CHECKPOINTS), help='Baseline or board-selected width176 student')
    choice.add_argument('--checkpoint', type=Path, help='Use a specific converted PyTorch checkpoint')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resize-mode', choices=('stretch', 'letterbox', 'center-crop'), default='stretch',
                   help='Adapt input frames to 4:3 before the model preprocessing (default: legacy stretch)')
    p.add_argument('--side-by-side', action='store_true', help='Show preprocessed RGB beside depth')
    p.add_argument('--depth-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
                   help='Fixed color range in model native units; otherwise normalize every frame')
    p.add_argument('--save-depth', action='store_true', help='Save each numeric prediction as a .npy file')
    p.add_argument('--max-frames', type=int, help='Optional frame limit for a quick test')
    a = p.parse_args()
    if a.output.exists(): p.error('output exists; choose a new output directory')
    if a.max_frames is not None and a.max_frames < 1: p.error('max-frames must be positive')
    if a.depth_range is not None and (not np.isfinite(a.depth_range).all() or a.depth_range[0] >= a.depth_range[1]):
        p.error('depth-range must contain two finite numbers with MIN < MAX')
    checkpoint = a.checkpoint or CHECKPOINTS[a.model or 'baseline']
    if not checkpoint.is_file(): p.error(f'checkpoint not found: {checkpoint}')
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened(): p.error(f'cannot open input video: {a.video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release(); p.error('input video has no valid FPS')
    writer = None
    try:
        model, record = load_converted(checkpoint, a.device)
        model.eval()
        a.output.mkdir(parents=True)
        if a.save_depth: (a.output / 'depth_arrays').mkdir()
        width = 448 if a.side_by_side else 224
        writer = cv2.VideoWriter(str(a.output / 'depth.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, 224))
        if not writer.isOpened(): raise RuntimeError('cannot open MP4 video writer; check OpenCV codecs')
        count = 0
        first_geometry = None
        source_sizes = set()
        start = time.perf_counter()
        while a.max_frames is None or count < a.max_frames:
            ok, frame = cap.read()
            if not ok: break
            prepared, geometry = adapt_frame(frame, a.resize_mode)
            if first_geometry is None: first_geometry = geometry
            source_sizes.add((geometry['source_width'], geometry['source_height']))
            rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
            x = preprocess(rgb)
            with torch.inference_mode():
                prediction = model(x.unsqueeze(0).to(a.device)).cpu().numpy()[0, 0]
            if prediction.shape != (224, 224) or not np.isfinite(prediction).all():
                raise ValueError(f'invalid depth prediction at frame {count}')
            if a.save_depth: np.save(a.output / 'depth_arrays' / f'{count:06d}.npy', prediction)
            if a.depth_range is None:
                preview = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            else:
                low, high = a.depth_range
                preview = np.rint(np.clip((prediction - low) / (high - low), 0, 1) * 255).astype(np.uint8)
            colored = cv2.applyColorMap(preview, cv2.COLORMAP_MAGMA)
            if a.side_by_side:
                cropped_rgb = np.rint(x.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                colored = np.concatenate((cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR), colored), axis=1)
            writer.write(colored)
            count += 1
            if count % 100 == 0: print(f'Processed {count} frames', flush=True)
        writer.release(); writer = None
        elapsed = time.perf_counter() - start
        if not count: raise ValueError('input video contained no readable frames')
        report = {
            'input_video': str(a.video), 'output_video': 'depth.mp4', 'frames': count,
            'elapsed_seconds': elapsed, 'complete_pipeline_fps': count / elapsed, 'input_fps': fps,
            'output_resolution': [width, 224], 'side_by_side': a.side_by_side,
            'input_resize': a.resize_mode, 'first_frame_geometry': first_geometry,
            'source_resolutions': [list(size) for size in sorted(source_sizes)],
            'model_preprocessing': 'legacy 304x228 center crop resized to 224x224 (nonuniform); side-by-side shows actual model input',
            'preview': 'fixed shared range with clipping' if a.depth_range else 'per-frame normalized color; may flicker',
            'depth_range_native_units': a.depth_range, 'numeric_depth_saved': a.save_depth,
            'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
            'model_source': record['provenance']['source_status'], 'runtime': 'PyTorch', 'device': a.device,
            'timing_scope': 'decode, resize, preprocess, transfer, inference, colorize and encode; model load excluded; optional depth saving included',
            'limitations': 'video colors are visualization, not calibrated metric depth; units unverified; input audio not retained; offline throughput is not live camera latency',
        }
        (a.output / 'metadata.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        cap.release()
        if writer is not None: writer.release()


if __name__ == '__main__':
    main()
