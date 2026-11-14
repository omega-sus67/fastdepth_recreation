"""Explicit architecture/weight conversion, including pruned channel dimensions."""
import argparse
from collections import OrderedDict, defaultdict
from functools import lru_cache
import hashlib
import importlib.util
import pickle
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ('mobilenet-nnconv5', 'mobilenet-nnconv5dw',
            'mobilenet-nnconv5dw-skipadd', 'mobilenet-nnconv5dw-skipadd-pruned')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def upstream():
    directory = ROOT/'third_party/fast-depth'
    spec = importlib.util.spec_from_file_location('_fastdepth_vendor_models', directory/'models.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(directory))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(directory))
    return module


def build_random_model(variant):
    """For explicitly labeled structural checks only; never pretrained inference."""
    vendor = upstream()
    if variant == VARIANTS[0]:
        return vendor.MobileNet('nnconv5', (224, 224), pretrained=False)
    if variant == VARIANTS[1]:
        return vendor.MobileNet('nnconv5dw', (224, 224), pretrained=False)
    if variant == VARIANTS[2]:
        return vendor.MobileNetSkipAdd((224, 224), pretrained=False)
    if variant == VARIANTS[3]:
        raise ValueError('Pruned topology must come from its checkpoint/configuration; no unpruned substitute is allowed')
    raise ValueError(f'Unknown variant: {variant}')


def module_spec(module):
    """Encode behavior class plus every structural child and channel parameter."""
    vendor = upstream()
    params = {}
    if isinstance(module, nn.Conv2d):
        kind = 'Conv2d'
        params = {name: getattr(module, name) for name in
                  ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'dilation', 'groups')}
        params.update(bias=module.bias is not None, padding_mode=getattr(module, 'padding_mode', 'zeros'))
    elif isinstance(module, nn.BatchNorm2d):
        kind = 'BatchNorm2d'
        params = {name: getattr(module, name) for name in ('num_features', 'eps', 'momentum', 'affine')}
        params['track_running_stats'] = getattr(module, 'track_running_stats', True)
    elif isinstance(module, nn.ReLU6):
        kind, params = 'ReLU6', {'inplace': module.inplace}
    elif isinstance(module, nn.ReLU):
        kind, params = 'ReLU', {'inplace': module.inplace}
    elif isinstance(module, nn.Sequential):
        kind = 'Sequential'
    elif type(module) in (vendor.MobileNet, vendor.MobileNetSkipAdd, vendor.NNConv, vendor.Identity):
        kind = type(module).__name__
        if hasattr(module, 'output_size'):
            params['output_size'] = list(module.output_size)
    else:
        raise ValueError(f'Unsupported module in architecture conversion: {type(module)}')
    return {'type': kind, 'parameters': params,
            'children': [[name, module_spec(child)] for name, child in module.named_children()]}


def from_spec(spec):
    vendor = upstream()
    kind, params = spec['type'], dict(spec['parameters'])
    if kind == 'Sequential':
        module = nn.Sequential()
    elif kind in ('Conv2d', 'BatchNorm2d', 'ReLU', 'ReLU6'):
        module = getattr(nn, kind)(**params)
    elif kind in ('MobileNet', 'MobileNetSkipAdd', 'NNConv', 'Identity'):
        cls = getattr(vendor, kind)
        module = cls.__new__(cls)
        nn.Module.__init__(module)
        if 'output_size' in params:
            module.output_size = tuple(params['output_size'])
    else:
        raise ValueError(f'Unsupported architecture type: {kind}')
    for name, child in spec['children']:
        module.add_module(name, from_spec(child))
    return module


class LegacyResult:
    """Metadata holder; no legacy result methods are executed during conversion."""


class LegacyOptimizerState:
    """Metadata holder; conversion does not resume the old optimizer."""


def legacy_backend():
    # PyTorch 0.4 modules serialized this obsolete backend; modern forward ignores it.
    return None


def load_legacy(path, expected_sha256):
    """Read legacy files with an explicit known-class allowlist, never unrestricted pickle."""
    if sha256(path) != expected_sha256.lower():
        raise ValueError('Checkpoint SHA-256 mismatch')
    vendor = upstream()
    classes = (vendor.MobileNet, vendor.MobileNetSkipAdd, vendor.NNConv, vendor.Identity,
               nn.Sequential, nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.ReLU6,
               nn.Parameter, nn.DataParallel, argparse.Namespace, OrderedDict, defaultdict, dict, set)
    allowed = list(classes)
    allowed += [(getattr(vendor, name), f'models.{name}') for name in
                ('MobileNet', 'MobileNetSkipAdd', 'NNConv', 'Identity')]
    allowed += [(LegacyResult, 'metrics.Result'), (LegacyOptimizerState, 'torch.optim.sgd.SGD'),
                (legacy_backend, 'torch.nn.backends.thnn._get_thnn_function_backend')]
    # PyTorch 0.4 uses persistent IDs for module source, which weights_only=True
    # cannot read. Supply a restricted pickle implementation instead: every
    # global must be one of these reviewed classes/functions, including aliases.
    permitted = {}
    for item in allowed:
        obj, name = item if isinstance(item, tuple) else (item, f'{item.__module__}.{item.__name__}')
        permitted[name] = obj
    for name in ('FloatStorage', 'LongStorage', 'IntStorage', 'DoubleStorage', 'HalfStorage', 'ByteStorage'):
        permitted[f'torch.{name}'] = getattr(torch, name)
    for name in ('_rebuild_tensor', '_rebuild_tensor_v2', '_rebuild_parameter'):
        permitted[f'torch._utils.{name}'] = getattr(torch._utils, name)
    permitted.update({'__builtin__.dict': dict, '__builtin__.set': set})

    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            key = f'{module}.{name}'
            if key not in permitted:
                raise pickle.UnpicklingError(f'Unreviewed checkpoint global is forbidden: {key}')
            return permitted[key]

    restricted_pickle = SimpleNamespace(
        __name__='fastdepth_restricted_pickle', Unpickler=RestrictedUnpickler,
        load=lambda stream, **kwargs: RestrictedUnpickler(stream, **kwargs).load())
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, pickle_module=restricted_pickle)
    model = checkpoint.get('model') if isinstance(checkpoint, dict) else checkpoint
    if isinstance(model, nn.DataParallel):
        model = model.module
    if not isinstance(model, nn.Module):
        raise ValueError('Expected a serialized model with its actual topology; state_dict-only files need explicit model configuration')
    return model, checkpoint


def load_converted(path, device='cpu'):
    record = torch.load(path, map_location='cpu', weights_only=True)
    if record.get('schema_version') != 1:
        raise ValueError('Unsupported converted checkpoint schema')
    model = from_spec(record['model_config'])
    model.load_state_dict(record['state_dict'], strict=True)
    return model.to(device).eval(), record
