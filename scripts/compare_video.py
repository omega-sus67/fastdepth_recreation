#!/usr/bin/env python3
"""Compare baseline and selected student on the same recorded video frames."""
import argparse
import csv
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

BASELINE = ROOT / 'checkpoints/converted/community-skipadd-pruned.pth'
STUDENT = ROOT / 'results/training/width_176/checkpoint.pth'
HEADER = 52


def percentile_scale(pending):
    """Choose a fixed shared color range from the initial frames of both models."""
    values = np.concatenate([d.reshape(-1) for item in pending for d in (item['baseline'], item['student'])])
    low, high = (float(v) for v in np.percentile(values, [1, 99]))
    if high <= low:
        high = low + max(abs(low) * 1e-6, 1e-6)
    return low, high


def colorize(depth, low, high):
    values = np.rint(np.clip((depth - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(values, cv2.COLORMAP_MAGMA)


def panel(image, label, low, high):
    output = np.zeros((224 + HEADER, 224, 3), dtype=np.uint8)
    output[HEADER:] = image
    cv2.putText(output, label, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1, cv2.LINE_AA)
    if low is not None:
        cv2.putText(output, f'{low:.2f} to {high:.2f} (units unknown)', (5, 32), cv2.FONT_HERSHEY_SIMPLEX, .32, (220, 220, 220), 1, cv2.LINE_AA)
        ramp = np.linspace(0, 255, 214, dtype=np.uint8).reshape(1, 214)
        output[42:48, 5:219] = np.repeat(cv2.applyColorMap(ramp, cv2.COLORMAP_MAGMA), 6, axis=0)
    return output


def summarize(rows, key):
    v = np.asarray([row[key] for row in rows], dtype=np.float64)
    return {'mean_ms': float(v.mean()), 'median_ms': float(np.median(v)), 'p95_ms': float(np.percentile(v, 95))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path, help='Input video (audio is not retained)')
    p.add_argument('--output', type=Path, required=True, help='New output directory')
    p.add_argument('--baseline', type=Path, default=BASELINE)
    p.add_argument('--student', type=Path, default=STUDENT, help='Default: selected width176 checkpoint')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resize-mode', choices=('stretch', 'letterbox', 'center-crop'), default='stretch')
    p.add_argument('--depth-range', nargs=2, type=float, metavar=('MIN', 'MAX'), help='Fixed common scale in native units')
    p.add_argument('--calibration-frames', type=int, default=30, help='Initial frames used to choose a fixed scale (default: 30)')
    p.add_argument('--warmup', type=int, default=3, help='Warmup predictions per model (default: 3)')
    p.add_argument('--max-frames', type=int, help='Optional maximum input frames')
    p.add_argument('--save-depth', action='store_true', help='Save both raw float predictions per frame')
    a = p.parse_args()
    if a.output.exists(): p.error('output exists; choose a new directory')
    if a.calibration_frames < 1 or a.warmup < 0 or (a.max_frames is not None and a.max_frames < 1):
        p.error('calibration frames and max frames must be positive; warmup must be nonnegative')
    if a.depth_range and (not np.isfinite(a.depth_range).all() or a.depth_range[0] >= a.depth_range[1]):
        p.error('depth-range requires finite MIN < MAX')
    for file in (a.video, a.baseline, a.student):
        if not file.is_file(): p.error(f'file not found: {file}')
    if a.device.startswith('cuda') and not torch.cuda.is_available(): p.error('CUDA unavailable in this Python environment')
    capture = cv2.VideoCapture(str(a.video))
    if not capture.isOpened(): p.error('cannot open video with OpenCV')
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release(); p.error('input has no valid FPS')
    reported_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    try:
        base, base_record = load_converted(a.baseline, a.device)
        student, student_record = load_converted(a.student, a.device)
        base.eval(); student.eval()
        a.output.mkdir(parents=True)
        if a.save_depth:
            (a.output / 'baseline_depth').mkdir(); (a.output / 'student_depth').mkdir()
        writer = cv2.VideoWriter(str(a.output / 'comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (672, 224 + HEADER))
        if not writer.isOpened(): raise RuntimeError('MP4 writer unavailable; verify OpenCV codec support')
        pending = []; rows = []; scale = tuple(a.depth_range) if a.depth_range else None
        auto_frames = 0; clip = {'baseline_low': 0, 'baseline_high': 0, 'student_low': 0, 'student_high': 0}
        geometry = None; source_sizes = set(); first = True; warmup_seconds = 0.0
        start = time.perf_counter()
        csv_path = a.output / 'frame_timings.csv'
        def render(item):
            low, high = scale
            t = time.perf_counter_ns()
            b, s = item['baseline'], item['student']
            for name, d in [('baseline', b), ('student', s)]:
                clip[name + '_low'] += int(np.count_nonzero(d < low))
                clip[name + '_high'] += int(np.count_nonzero(d > high))
            rgb_bgr = cv2.cvtColor(item['input_rgb'], cv2.COLOR_RGB2BGR)
            image = np.concatenate((panel(rgb_bgr, 'Model input (RGB)', None, None),
                                    panel(colorize(b, low, high), 'Baseline', low, high),
                                    panel(colorize(s, low, high), 'Student', low, high)), axis=1)
            writer.write(image)
            if a.save_depth:
                np.save(a.output / 'baseline_depth' / f"{item['index']:06d}.npy", b)
                np.save(a.output / 'student_depth' / f"{item['index']:06d}.npy", s)
            item['row']['render_encode_save_ms'] = (time.perf_counter_ns() - t) / 1e6
        while a.max_frames is None or len(rows) < a.max_frames:
            t = time.perf_counter_ns(); ok, frame = capture.read(); decode_ms = (time.perf_counter_ns() - t) / 1e6
            if not ok: break
            t = time.perf_counter_ns(); adapted, frame_geometry = adapt_frame(frame, a.resize_mode)
            if geometry is None: geometry = frame_geometry
            source_sizes.add((frame_geometry['source_width'], frame_geometry['source_height']))
            rgb = cv2.cvtColor(adapted, cv2.COLOR_BGR2RGB); x_cpu = preprocess(rgb)
            input_rgb = np.rint(x_cpu.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            x = x_cpu.unsqueeze(0).to(a.device)
            if a.device.startswith('cuda'): torch.cuda.synchronize()
            preprocessing_ms = (time.perf_counter_ns() - t) / 1e6
            if first:
                t = time.perf_counter()
                with torch.inference_mode():
                    for _ in range(a.warmup): base(x); student(x)
                if a.device.startswith('cuda'): torch.cuda.synchronize()
                warmup_seconds = time.perf_counter() - t; first = False
            outputs = []
            for model in (base, student):
                t = time.perf_counter_ns()
                with torch.inference_mode(): depth = model(x).cpu().numpy()[0, 0]
                duration = (time.perf_counter_ns() - t) / 1e6
                if depth.shape != (224, 224) or not np.isfinite(depth).all(): raise ValueError('invalid depth prediction')
                outputs.append((depth, duration))
            row = {'frame': len(rows), 'decode_ms': decode_ms, 'preprocess_h2d_ms': preprocessing_ms,
                   'baseline_infer_d2h_ms': outputs[0][1], 'student_infer_d2h_ms': outputs[1][1], 'render_encode_save_ms': 0.0}
            rows.append(row)
            item = {'index': row['frame'], 'input_rgb': input_rgb, 'baseline': outputs[0][0], 'student': outputs[1][0], 'row': row}
            if scale is None:
                pending.append(item)
                if len(pending) >= a.calibration_frames:
                    scale = percentile_scale(pending); auto_frames = len(pending)
                    for previous in pending: render(previous)
                    pending.clear()
            else: render(item)
            if len(rows) % 100 == 0: print(f'Processed {len(rows)} frames', flush=True)
        if not rows: raise ValueError('video contained no readable frames')
        if pending:
            scale = percentile_scale(pending); auto_frames = len(pending)
            for item in pending: render(item)
        writer.release(); writer = None
        elapsed = time.perf_counter() - start
        with csv_path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        validation = cv2.VideoCapture(str(a.output / 'comparison.mp4')); decoded = 0
        if not validation.isOpened(): raise RuntimeError('output video cannot be reopened')
        try:
            while True:
                ok, image = validation.read()
                if not ok: break
                if image.shape[:2] != (224 + HEADER, 672): raise ValueError('unexpected output frame dimensions')
                decoded += 1
        finally: validation.release()
        if decoded != len(rows): raise ValueError(f'output frame count mismatch: wrote {len(rows)}, decoded {decoded}')
        total_pixels = len(rows) * 224 * 224
        report = {
            'input_video': str(a.video), 'input_video_sha256': sha256(a.video),
            'frames_processed': len(rows), 'input_reported_frame_count': reported_count if np.isfinite(reported_count) else None,
            'input_may_have_ended_early': bool(np.isfinite(reported_count) and reported_count > len(rows) + 1 and (a.max_frames is None or len(rows) < a.max_frames)),
            'input_fps_reported': fps, 'output_fps': fps, 'output_size': [672, 224 + HEADER],
            'source_resolutions': [list(v) for v in sorted(source_sizes)], 'resize_mode': a.resize_mode,
            'first_frame_geometry': geometry, 'model_input_shape': [1, 3, 224, 224],
            'model_preprocessing': 'existing 640x480 adapter, legacy resize/crop 304x228 then nonuniform resize to 224x224',
            'baseline_sha256': sha256(a.baseline), 'student_sha256': sha256(a.student),
            'baseline_provenance': base_record['provenance']['source_status'], 'student_provenance': student_record['provenance']['source_status'],
            'runtime': 'PyTorch', 'torch': str(torch.__version__), 'opencv': cv2.__version__, 'device': a.device,
            'gpu': torch.cuda.get_device_name(torch.device(a.device)) if a.device.startswith('cuda') else None,
            'common_depth_range': list(scale), 'scale_method': 'user-specified' if a.depth_range else 'initial-frame sample, pooled 1st and 99th percentiles across both models',
            'scale_calibration_frames': auto_frames, 'clipped_pixel_fraction': {k: v / total_pixels for k, v in clip.items()},
            'warmup_inferences_per_model': a.warmup, 'warmup_seconds': warmup_seconds,
            'elapsed_seconds': elapsed, 'paired_pipeline_fps_including_warmup': len(rows) / elapsed,
            'per_stage': {key: summarize(rows, key) for key in ('decode_ms', 'preprocess_h2d_ms', 'baseline_infer_d2h_ms', 'student_infer_d2h_ms', 'render_encode_save_ms')},
            'timing_scope': 'offline paired processing: decode, both model inferences, preprocessing, transfers, visual rendering and encode; includes warmup and scale calibration; excludes model loading, input hashing and output validation; optional array writes included',
            'output_decoded_frames': decoded, 'numeric_depth_saved': a.save_depth,
            'limitations': 'fixed frame rate assumes constant-rate source; source frame timestamps/audio not retained; decoder frame count is only a reported estimate and cannot prove whether input decoding was complete; legacy resize can distort; depth units unverified; paired runtime and timing differ from single model/TensorRT/live camera; initial-frame scale can clip later scenes; inference order is fixed',
        }
        (a.output / 'metadata.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        capture.release()
        if writer is not None: writer.release()


if __name__ == '__main__': main()
