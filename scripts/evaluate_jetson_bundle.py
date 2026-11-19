#!/usr/bin/env python3
"""Measure a local TensorRT engine on preprocessed RGB/depth evaluation tensors."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fastdepth_project.metrics import DepthMetrics

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', required=True, type=Path)
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--runs', type=int, default=300)
    p.add_argument('--timing-rounds', type=int, default=1, help='Repeated timing rounds after accuracy evaluation')
    a = p.parse_args()
    if a.output.exists() or a.warmup < 0 or a.runs < 1 or a.timing_rounds < 1: p.error('output exists or invalid timing settings')
    manifest = json.loads((a.data / 'manifest.json').read_text())
    for name, key in [('inputs.npy', 'inputs_sha256'), ('targets.npy', 'targets_sha256')]:
        if sha(a.data / name) != manifest[key]: raise ValueError('data hash mismatch: ' + name)
    inputs = np.load(a.data / 'inputs.npy', mmap_mode='r', allow_pickle=False)
    targets = np.load(a.data / 'targets.npy', mmap_mode='r', allow_pickle=False)
    n = manifest['samples']
    if inputs.shape != (n, 3, 224, 224) or targets.shape != (n, 1, 224, 224): raise ValueError('incorrect data shape')
    ref_manifest = json.loads((a.reference.parent / 'manifest.json').read_text())
    if ref_manifest['data_manifest_sha256'] != sha(a.data / 'manifest.json') or ref_manifest['samples'] != n:
        raise ValueError('reference data manifest mismatch')
    if sha(a.reference) != ref_manifest['models'][a.reference.stem]['reference_sha256']:
        raise ValueError('reference hash mismatch')
    reference = np.load(a.reference, mmap_mode='r', allow_pickle=False)
    if reference.shape != (n, 1, 224, 224): raise ValueError('incorrect reference shape')
    logger = trt.Logger(trt.Logger.WARNING); runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(a.engine.read_bytes())
    if engine is None: raise RuntimeError('engine deserialization failed')
    context = engine.create_execution_context()
    x = torch.empty((1, 3, 224, 224), device='cuda'); y = torch.empty((1, 1, 224, 224), device='cuda')
    if not context.set_tensor_address('rgb', x.data_ptr()) or not context.set_tensor_address('depth', y.data_ptr()): raise RuntimeError('binding failed')
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    def execute():
        if not context.execute_async_v3(stream.cuda_stream): raise RuntimeError('execution failed')
    a.output.mkdir(parents=True)
    total = DepthMetrics(); rows = []
    for i in range(n):
        start = time.perf_counter_ns()
        x.copy_(torch.from_numpy(np.array(inputs[i:i+1], copy=True)))
        stream.wait_stream(torch.cuda.current_stream()); execute(); stream.synchronize()
        prediction = y.cpu().numpy()[0, 0]
        transfer_inference_output_ms = (time.perf_counter_ns() - start) / 1e6
        target = targets[i, 0]
        diff = np.abs(prediction.astype(np.float64) - reference[i, 0].astype(np.float64))
        if not np.isfinite(diff).all(): raise ValueError('nonfinite runtime difference')
        one = DepthMetrics(); one.update(prediction, target); total.update(prediction, target)
        rows.append({'sample': manifest['sources'][i]['name'], 'prepared_input_to_cpu_output_ms': transfer_inference_output_ms,
                     'runtime_max_abs_difference': float(diff.max()), 'runtime_mean_abs_difference': float(diff.mean()), **one.result()})
        if (i + 1) % 100 == 0: print(f'{i+1}/{n}', flush=True)
    with (a.output / 'per_image.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    timings = []; timing_rows = []; round_medians = []; timing_windows = []
    for round_id in range(a.timing_rounds):
        for _ in range(a.warmup): execute()
        stream.synchronize(); round_times = []; window_start = time.time()
        for iteration in range(a.runs):
            start = time.perf_counter_ns(); execute(); stream.synchronize()
            elapsed = (time.perf_counter_ns() - start) / 1e6
            timings.append(elapsed); round_times.append(elapsed); timing_rows.append([round_id, iteration, elapsed])
        timing_windows.append({'start_unix_seconds': window_start, 'end_unix_seconds': time.time(), 'inferences': len(round_times)})
        round_medians.append(float(np.median(round_times)))
    with (a.output / 'latency_ms.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['round', 'iteration', 'latency_ms']); w.writerows(timing_rows)
    report = {'engine_sha256': sha(a.engine), 'manifest_sha256': sha(a.data / 'manifest.json'),
              'reference_sha256': sha(a.reference), 'max_runtime_abs_difference': max(r['runtime_max_abs_difference'] for r in rows),
              'mean_runtime_abs_difference': float(np.mean([r['runtime_mean_abs_difference'] for r in rows])),
              'samples': n, 'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__), 'tensorrt': trt.__version__,
              'pooled': total.result(), 'mean_image_rmse': float(np.mean([r['rmse'] for r in rows])),
              'mean_image_delta1': float(np.mean([r['delta1'] for r in rows])), 'warmup': a.warmup, 'runs': a.runs,
              'timing_rounds': a.timing_rounds, 'round_medians_ms': round_medians, 'timing_windows': timing_windows,
              'model_only_median_ms': float(np.median(timings)), 'model_only_p95_ms': float(np.percentile(timings, 95)),
              'prepared_input_to_cpu_output_median_ms': float(np.median([r['prepared_input_to_cpu_output_ms'] for r in rows])),
              'prepared_input_to_cpu_output_p95_ms': float(np.percentile([r['prepared_input_to_cpu_output_ms'] for r in rows], 95)),
              'timing_scope': 'host elapsed enqueue plus non-default stream synchronization; preallocated input/output, excludes transfers and preprocessing',
              'prepared_input_scope': 'host array copy, host-to-device, inference, device-to-host; excludes image decode, resize and visualization; includes memory-map page faults',
              'limitations': 'preprocessed fixed evaluation set; units and complete split identity unverified; not end-to-end latency'}
    (a.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
