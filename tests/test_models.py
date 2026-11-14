from pathlib import Path
import pickle

import pytest
import torch
from torch import nn

from fastdepth_project.models import (VARIANTS, build_random_model, from_spec,
                                     load_converted, load_legacy, module_spec, sha256)


@pytest.mark.parametrize('variant', VARIANTS[:3])
def test_original_architecture_forward(variant):
    model = build_random_model(variant).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 32, 32))
    assert output.shape == (1, 1, 32, 32)
    assert torch.isfinite(output).all()


def test_pruned_model_cannot_silently_use_unpruned_channels():
    with pytest.raises(ValueError, match='Pruned topology'):
        build_random_model(VARIANTS[3])


def test_explicit_channel_topology_and_weights_roundtrip(tmp_path):
    model = nn.Sequential(nn.Conv2d(3, 5, 3, padding=1, groups=1, bias=False),
                          nn.BatchNorm2d(5), nn.ReLU6(), nn.Conv2d(5, 1, 1)).eval()
    spec = module_spec(model)
    assert spec['children'][0][1]['parameters']['out_channels'] == 5
    path = tmp_path/'converted.pth'
    torch.save({'schema_version': 1, 'variant': 'structural-test',
                'model_config': spec, 'state_dict': model.state_dict()}, path)
    restored, _ = load_converted(path)
    x = torch.rand(1,3,8,8)
    with torch.inference_mode():
        torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


def test_unknown_module_type_is_not_reconstructed():
    with pytest.raises(ValueError, match='Unsupported architecture type'):
        from_spec({'type': 'ArbitraryClass', 'parameters': {}, 'children': []})


def test_hash_checked_before_legacy_deserialization(tmp_path):
    path = tmp_path/'bad.pth'
    path.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        load_legacy(path, '0'*64)


class UnreviewedObject:
    def __reduce__(self):
        return eval, ('40 + 2',)


def test_unreviewed_pickle_global_rejected(tmp_path):
    path = tmp_path/'unreviewed.pth'
    torch.save(UnreviewedObject(), path, _use_new_zipfile_serialization=False)
    with pytest.raises(pickle.UnpicklingError, match='forbidden:.*eval'):
        load_legacy(path, sha256(path))
