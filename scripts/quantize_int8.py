#!/usr/bin/env python3
"""Create an explicit symmetric INT8 Q/DQ ONNX model using training-only inputs."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


class Reader(CalibrationDataReader):
    def __init__(self, inputs, limit):
        self.inputs, self.limit, self.index = inputs, limit, 0

    def get_next(self):
        if self.index >= self.limit: return None
        value = {'rgb': np.array(self.inputs[self.index:self.index+1], dtype=np.float32, copy=True)}
        self.index += 1
        return value

    def rewind(self): self.index = 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--onnx', type=Path, required=True)
    p.add_argument('--calibration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, help='Diagnostic partial calibration only, not the declared experiment')
    p.add_argument('--policy', choices=('all-conv', 'pointwise', 'encoder', 'decoder'), default='all-conv',
                   help='Convolution subset to quantize; other operations remain floating-point')
    a = p.parse_args()
    if a.output.exists(): p.error('output already exists')
    manifest = json.loads((a.calibration / 'manifest.json').read_text())
    if manifest['inputs_sha256'] != sha(a.calibration / 'inputs.npy'): raise ValueError('Calibration tensor hash mismatch')
    inputs = np.load(a.calibration / 'inputs.npy', mmap_mode='r', allow_pickle=False)
    if list(inputs.shape) != manifest['shape'] or inputs.dtype != np.float32: raise ValueError('Invalid calibration tensor')
    limit = len(inputs) if a.limit is None else a.limit
    if not 1 <= limit <= len(inputs): p.error('invalid calibration limit')
    a.output.mkdir(parents=True)
    target = a.output / 'model.onnx'
    source = onnx.load(a.onnx)
    convs = [node for node in source.graph.node if node.op_type == 'Conv']
    selected = None
    if a.policy == 'pointwise':
        selected = [node.name for node in convs
                    if any(attr.name == 'kernel_shape' and list(attr.ints) == [1, 1] for attr in node.attribute)
                    and not any('decode_conv6' in name for name in node.input)]
        if not selected: raise ValueError('No eligible pointwise convolutions')
    elif a.policy in ('encoder', 'decoder'):
        selected = [node.name for node in convs if any(
            name.startswith('decode_conv') if a.policy == 'decoder' else name.startswith('conv')
            for name in node.input[1:2])]
        if not selected: raise ValueError('No eligible convolutions for chosen policy')
    # ORT creates a sibling *-inferred.onnx file; isolate concurrent attempts.
    # These project exports embed weights, so copying the ONNX file is sufficient.
    with tempfile.TemporaryDirectory(prefix='fastdepth-quant-') as temp:
        source_copy = Path(temp) / 'source.onnx'
        shutil.copy2(a.onnx, source_copy)
        quantize_static(source_copy, target, Reader(inputs, limit), quant_format=QuantFormat.QDQ,
                        op_types_to_quantize=['Conv'], nodes_to_quantize=selected, per_channel=True, activation_type=QuantType.QInt8,
                        weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax,
                        calibration_providers=['CPUExecutionProvider'],
                        extra_options={'ActivationSymmetric': True, 'WeightSymmetric': True, 'QuantizeBias': False})
    model = onnx.load(target); onnx.checker.check_model(model)
    initializers = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
    qnodes = [v for v in model.graph.node if v.op_type in ('QuantizeLinear', 'DequantizeLinear')]
    if not qnodes: raise ValueError('Quantizer did not add Q/DQ operations')
    nonzero = [node.name for node in qnodes if len(node.input) >= 3 and node.input[2] in initializers and np.any(initializers[node.input[2]] != 0)]
    if nonzero: raise ValueError(f'Non-zero quantization zero points: {nonzero[:5]}')
    ref = ort.InferenceSession(str(a.onnx), providers=['CPUExecutionProvider'])
    quant = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
    sample = np.array(inputs[0:1], copy=True)
    expected = ref.run(None, {'rgb': sample})[0]; actual = quant.run(None, {'rgb': sample})[0]
    if actual.shape != expected.shape or not np.isfinite(actual).all(): raise ValueError('Invalid quantized output')
    diff = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    report = {'scope': 'desktop ONNX Runtime CPU post-training quantization; not Jetson validation',
              'source_onnx_sha256': sha(a.onnx), 'quantized_onnx_sha256': sha(target),
              'calibration_manifest_sha256': sha(a.calibration / 'manifest.json'), 'calibration_samples': limit,
              'calibration_scope': 'training-only', 'method': 'Conv Q/DQ signed symmetric INT8; per-channel weights, MinMax activations',
              'policy': a.policy, 'selected_conv_nodes': selected if selected is not None else [n.name for n in convs],
              'bias_policy': 'floating-point bias to avoid unsupported INT32 DequantizeLinear in desktop TensorRT 10.3',
              'qdq_nodes': len(qnodes), 'nonzero_initializer_zero_points': nonzero,
              'output_shape': list(actual.shape), 'sample_max_abs_difference': float(diff.max()),
              'sample_mean_abs_difference': float(diff.mean()), 'onnxruntime': ort.__version__, 'onnx': onnx.__version__,
              'limits': 'One calibration input compared; no acceptance threshold, full accuracy, TensorRT compatibility, target speed or fully INT8 execution established.'}
    (a.output/'quantization.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
