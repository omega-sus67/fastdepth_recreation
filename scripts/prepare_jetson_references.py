#!/usr/bin/env python3
"""Generate desktop reference predictions for the fixed board comparison."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fastdepth_project.models import load_converted

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--model', action='append', metavar='NAME=CHECKPOINT', help='Choose a reference model; repeat for each model')
    a = p.parse_args()
    if a.output.exists(): p.error('output exists')
    manifest = json.loads((a.data / 'manifest.json').read_text())
    if sha(a.data / 'inputs.npy') != manifest['inputs_sha256']: raise ValueError('input hash mismatch')
    inputs = np.load(a.data / 'inputs.npy', mmap_mode='r', allow_pickle=False); n = len(inputs)
    models = {
        'baseline': 'checkpoints/converted/community-skipadd-pruned.pth',
        'student': 'results/training/width_176/checkpoint.pth',
    }
    if a.model:
        models = {}
        for item in a.model:
            name, separator, source = item.partition('=')
            if not separator or not name or not source or Path(name).name != name or name in models:
                p.error('each model must have a unique simple NAME=CHECKPOINT')
            models[name] = source
    a.output.mkdir(parents=True); reports = {}
    for name, source in models.items():
        model, _ = load_converted(ROOT / source, a.device)
        path = a.output / f'{name}.npy'
        out = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(n, 1, 224, 224))
        with torch.inference_mode():
            for i in range(n):
                x = torch.from_numpy(np.array(inputs[i:i+1], copy=True)).to(a.device)
                out[i] = model(x).cpu().numpy()[0]
                if (i + 1) % 200 == 0: print(f'{name}: {i+1}/{n}', flush=True)
        out.flush(); del out
        reports[name] = {'reference_sha256': sha(path), 'checkpoint_sha256': sha(ROOT / source), 'checkpoint': source}
    report = {'data_manifest_sha256': sha(a.data / 'manifest.json'), 'samples': n, 'device': a.device,
              'gpu': torch.cuda.get_device_name(0) if a.device.startswith('cuda') else None,
              'torch': str(torch.__version__), 'models': reports,
              'limitations': 'desktop PyTorch predictions for runtime parity, not ground truth or Jetson results'}
    (a.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')

if __name__ == '__main__': main()
