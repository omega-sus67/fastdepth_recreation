#!/usr/bin/env python3
"""Run a fixed float32 tensor through a locally built TensorRT engine."""
import argparse
import json
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--engine', type=Path, required=True)
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    parser.error('output directory already exists')
values = np.fromfile(args.input, dtype='<f4')
if values.size != 3 * 224 * 224 or not np.isfinite(values).all():
    parser.error('input must contain 1x3x224x224 finite little-endian float32 values')
reference = np.load(args.reference, allow_pickle=False)
if reference.shape != (1, 1, 224, 224) or not np.isfinite(reference).all():
    parser.error('invalid reference output')
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
if engine is None:
    raise RuntimeError('engine deserialization failed')
context = engine.create_execution_context()
if context is None:
    raise RuntimeError('execution context creation failed')
x = torch.from_numpy(values.reshape(1, 3, 224, 224)).to('cuda')
y = torch.empty((1, 1, 224, 224), device='cuda', dtype=torch.float32)
if not context.set_tensor_address('rgb', x.data_ptr()) or not context.set_tensor_address('depth', y.data_ptr()):
    raise RuntimeError('tensor binding failed')
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
if not context.execute_async_v3(stream.cuda_stream):
    raise RuntimeError('TensorRT inference failed')
stream.synchronize()
output = y.cpu().numpy()
if not np.isfinite(output).all():
    raise ValueError('nonfinite output')
diff = np.abs(output.astype(np.float64) - reference.astype(np.float64))
args.output.mkdir(parents=True)
np.save(args.output / 'depth.npy', output)
report = {'engine': str(args.engine), 'input': str(args.input), 'reference': str(args.reference),
          'output_shape': list(output.shape), 'max_absolute_difference': float(diff.max()),
          'mean_absolute_difference': float(diff.mean()), 'tensorrt': trt.__version__,
          'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0),
          'scope': 'single synthetic tensor; not an accuracy or latency measurement'}
(args.output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
