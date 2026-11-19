#!/usr/bin/env python3
"""Compare ONNX models on the recorded tuning split, never the evaluation set."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fastdepth_project.data import load_h5, preprocess
from fastdepth_project.metrics import DepthMetrics

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--model', action='append', required=True, metavar='NAME=MODEL.onnx')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, help='Diagnostic subset, not full tuning comparison')
    a = p.parse_args()
    if a.output.exists(): p.error('output exists')
    cfg = json.loads(a.config.read_text()); manifest_path = ROOT / cfg['manifest']; overlap_path = ROOT / cfg['overlap_report']
    manifest = json.loads(manifest_path.read_text()); overlap = json.loads(overlap_path.read_text())
    if overlap['manifest_sha256'] != sha(manifest_path) or not overlap['train_tune_scene_disjoint'] or overlap['exact_rgb_overlap_with_evaluation'] or overlap['exact_rgb_overlap_train_tune_count']:
        raise ValueError('Data overlap audit failed')
    samples = manifest['groups']['tune']['samples']
    if a.limit is not None:
        if a.limit < 1: p.error('limit must be positive')
        samples = samples[:a.limit]
    if not samples: p.error('empty tuning set')
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    models = {}
    for spec in a.model:
        name, sep, path = spec.partition('=')
        if not sep or not name or name in models: p.error('expected unique NAME=PATH')
        path = Path(path); session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        models[name] = {'session': session, 'path': str(path), 'sha256': sha(path), 'total': DepthMetrics(), 'rmse': []}
    rows = []
    for i, row in enumerate(samples):
        path = ROOT / row['path']
        if sha(path) != row['sha256']: raise ValueError(f'Sample hash mismatch: {path}')
        rgb, depth = load_h5(path); x, target = preprocess(rgb, depth); target = target.numpy()[0]
        for name, entry in models.items():
            prediction = entry['session'].run(None, {'rgb': x.numpy()[None]})[0][0,0]
            one = DepthMetrics(); one.update(prediction, target); entry['total'].update(prediction, target)
            metrics = one.result(); entry['rmse'].append(metrics['rmse'])
            rows.append({'model': name, 'sample': row['path'], **metrics})
        if (i + 1) % 50 == 0 or i + 1 == len(samples): print(f'{i+1}/{len(samples)} tuning images', flush=True)
    a.output.mkdir(parents=True)
    with (a.output/'per_image.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    result = {name: {'path': e['path'], 'sha256': e['sha256'], 'pooled': e['total'].result(),
                     'mean_image_rmse': float(np.mean(e['rmse']))} for name,e in models.items()}
    report = {'scope': 'Diagnostic subset of tuning data' if a.limit is not None else 'Previously inspected tuning split; not a final evaluation',
              'samples': len(samples), 'config_sha256': sha(a.config), 'manifest_sha256': sha(manifest_path),
              'runtime': 'ONNX Runtime CPU', 'onnxruntime': ort.__version__, 'models': result,
              'limitations': 'Tuning set previously inspected; unknown pretraining overlap and physical scene identity. No TensorRT or Orin performance claim.'}
    (a.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: {'rmse': x['pooled']['rmse'], 'delta1': x['pooled']['delta1']} for name,x in result.items()}, indent=2))

if __name__ == '__main__': main()
