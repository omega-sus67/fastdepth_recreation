#!/usr/bin/env python3
"""Exercise desktop dependencies and runtimes; no pretrained accuracy is implied."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {
    'torch': 'torch', 'torchvision': 'torchvision', 'numpy': 'numpy',
    'h5py': 'h5py', 'matplotlib': 'matplotlib', 'imageio': 'imageio',
    'skimage': 'scikit-image', 'cv2': 'opencv-python-headless',
    'onnx': 'onnx', 'onnxruntime': 'onnxruntime', 'onnxscript': 'onnxscript',
    'scipy': 'scipy', 'yaml': 'PyYAML', 'tqdm': 'tqdm',
    'pytest': 'pytest', 'huggingface_hub': 'huggingface-hub',
}


def check_environment(require_cuda=False, require_tensorrt=False, require_independent=False):
    report = {
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Desktop dependency and synthetic runtime checks; not model accuracy or Orin validation.',
        'python': {'version': platform.python_version(), 'executable': sys.executable,
                   'prefix': sys.prefix, 'base_prefix': sys.base_prefix},
        'platform': platform.platform(), 'packages': {}, 'checks': {},
        'shared_package_paths': [p for p in sys.path if 'droidrun/venv' in p],
        'external_site_packages': [p for p in sys.path if 'site-packages' in p
                                   and not Path(p).resolve().is_relative_to(Path(sys.prefix).resolve())],
        'pythonpath_environment': os.environ.get('PYTHONPATH'),
    }
    failures = []
    if require_independent and (report['external_site_packages'] or sys.prefix == sys.base_prefix):
        failures.append('Independent virtual environment required; inherited site-packages or system interpreter detected')
    for module, package in PACKAGES.items():
        try:
            obj = importlib.import_module(module)
            report['packages'][package] = {'version': getattr(obj, '__version__', 'unknown'),
                                          'loaded_from': obj.__file__}
        except Exception as exc:
            report['packages'][package] = {'error': str(exc)}
            failures.append(f'{package}: {exc}')

    def exercise(name, function):
        try:
            report['checks'][name] = {'status': 'passed', **function()}
        except Exception as exc:
            report['checks'][name] = {'status': 'failed', 'error': str(exc)}
            failures.append(f'{name}: {exc}')

    if 'error' not in report['packages'].get('torch', {}):
        import torch

        def cpu_check():
            x = torch.ones(16, 16)
            torch.testing.assert_close(x @ x, torch.full_like(x, 16))
            model = torch.nn.Conv2d(3, 1, 3, padding=1).eval()
            with torch.inference_mode():
                y = model(torch.zeros(1, 3, 8, 8))
            assert y.shape == (1, 1, 8, 8) and torch.isfinite(y).all()
            return {'matrix_value': 16, 'synthetic_conv_shape': list(y.shape)}

        exercise('pytorch_cpu', cpu_check)
        report['cuda'] = {'available': torch.cuda.is_available(), 'pytorch_build': torch.version.cuda}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report['cuda'].update(device=props.name, total_memory_bytes=props.total_memory,
                                  compute_capability=f'{props.major}.{props.minor}')

            def cuda_check():
                x = torch.ones(128, 128, device='cuda')
                y = x @ x
                torch.cuda.synchronize()
                torch.testing.assert_close(y, torch.full_like(y, 128))
                return {'matrix_value': float(y[0, 0]), 'expected': 128}

            exercise('pytorch_cuda', cuda_check)
        elif require_cuda:
            failures.append('CUDA is required but unavailable')

    if all('error' not in report['packages'][p] for p in ('onnx', 'onnxruntime', 'numpy')):
        def ort_check():
            import numpy as np
            import onnx
            import onnxruntime as ort
            from onnx import TensorProto, helper
            graph = helper.make_graph(
                [helper.make_node('Add', ['x', 'x'], ['y'])], 'synthetic_runtime_check',
                [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 8, 8])],
                [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 3, 8, 8])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
            model.ir_version = 8
            onnx.checker.check_model(model)
            session = ort.InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])
            x = np.arange(192, dtype=np.float32).reshape(1, 3, 8, 8)
            y = session.run(None, {'x': x})[0]
            np.testing.assert_allclose(y, 2*x)
            return {'providers': ort.get_available_providers(), 'tested_provider': 'CPUExecutionProvider',
                    'output_shape': list(y.shape), 'max_abs_error': float(np.abs(y-2*x).max())}

        exercise('onnxruntime_cpu', ort_check)

    if all('error' not in report['packages'][p] for p in ('torch', 'onnx', 'onnxruntime', 'onnxscript')):
        def exporter_check():
            import numpy as np
            import onnxruntime as ort
            import torch
            model = torch.nn.Conv2d(3, 1, 3, padding=1).eval()
            x = torch.ones(1, 3, 8, 8)
            with tempfile.TemporaryDirectory(prefix='edgeai-export-') as directory:
                path = Path(directory)/'synthetic.onnx'
                torch.onnx.export(model, (x,), str(path), dynamo=True, opset_version=18,
                                  input_names=['x'], output_names=['y'])
                session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
                output = session.run(None, {'x': x.numpy()})[0]
                with torch.inference_mode():
                    expected = model(x).numpy()
                np.testing.assert_allclose(output, expected, atol=1e-5, rtol=1e-5)
                return {'exporter': 'torch.onnx dynamo', 'opset': 18,
                        'max_abs_error': float(np.abs(output-expected).max())}

        exercise('pytorch_onnx_export', exporter_check)

    try:
        import tensorrt as trt
        report['packages']['tensorrt'] = {'version': trt.__version__, 'loaded_from': trt.__file__}

        def tensorrt_check():
            import numpy as np
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('TensorRT execution requires CUDA')
            logger = trt.Logger(trt.Logger.ERROR)
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 64 * 1024 * 1024)
            input_tensor = network.add_input('x', trt.float32, (1, 3, 8, 8))
            offset = np.full((1, 1, 1, 1), 1.25, dtype=np.float32)
            constant = network.add_constant(offset.shape, offset)
            layer = network.add_elementwise(input_tensor, constant.get_output(0), trt.ElementWiseOperation.SUM)
            layer.get_output(0).name = 'y'
            network.mark_output(layer.get_output(0))
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError('TensorRT engine build returned None')
            runtime = trt.Runtime(logger)
            engine = runtime.deserialize_cuda_engine(serialized)
            context = engine.create_execution_context()
            x = torch.arange(192, dtype=torch.float32, device='cuda').reshape(1, 3, 8, 8).contiguous()
            y = torch.empty_like(x)
            if not context.set_tensor_address('x', x.data_ptr()) or not context.set_tensor_address('y', y.data_ptr()):
                raise RuntimeError('Failed to bind TensorRT buffers')
            if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
                raise RuntimeError('TensorRT execution failed')
            torch.cuda.synchronize()
            expected = x + 1.25
            torch.testing.assert_close(y, expected)
            return {'precision': 'FP32', 'synthetic_output_shape': list(y.shape),
                    'engine_bytes': len(bytes(serialized)), 'max_abs_error': float((y-expected).abs().max())}

        exercise('tensorrt_cuda', tensorrt_check)
    except ImportError as exc:
        report['packages']['tensorrt'] = {'available': False, 'error': str(exc)}
        report['checks']['tensorrt_cuda'] = {'status': 'not_installed', 'required': require_tensorrt}
        if require_tensorrt:
            failures.append(f'TensorRT required: {exc}')
    except Exception as exc:
        report['checks']['tensorrt_cuda'] = {'status': 'failed', 'error': str(exc)}
        failures.append(f'TensorRT: {exc}')

    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'],
                                capture_output=True, text=True, timeout=15, check=True)
        report['nvidia_smi'] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        report['nvidia_smi_error'] = str(exc)
    report['project'] = {
        'vendored_source_present': (ROOT/'third_party/fast-depth/models.py').is_file(),
        'checkpoint_files': [str(p.relative_to(ROOT)) for p in (ROOT/'checkpoints').rglob('*.pth*')],
        'validation_archives': len(list((ROOT/'data/nyu_depth_v2/raw').glob('val-*.tar'))),
    }
    report['failures'] = failures
    report['passed'] = not failures
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--require-tensorrt', action='store_true')
    parser.add_argument('--require-independent', action='store_true')
    args = parser.parse_args()
    report = check_environment(args.require_cuda, args.require_tensorrt, args.require_independent)
    for name, package in report['packages'].items():
        print(f"{name}: {package.get('version', package.get('error', 'not installed'))}")
    print(f"Python: {report['python']['version']} ({sys.executable})")
    for name, result in report['checks'].items():
        print(f"{name}: {result['status']}")
    for failure in report['failures']:
        print(f'ERROR: {failure}', file=sys.stderr)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print(f'Saved {args.output}')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
