import numpy as np
import pytest
from types import SimpleNamespace
from PIL import Image
from fastdepth_project import data
from fastdepth_project.data import preprocess

def test_shapes_alignment_and_normalization():
    rgb=np.zeros((480,640,3),dtype=np.uint8)
    rgb[:,:,0]=128
    depth=np.full((480,640),3.5,dtype=np.float32)
    x,y=preprocess(rgb,depth)
    assert tuple(x.shape)==(3,224,224) and tuple(y.shape)==(1,224,224)
    assert np.allclose(x[0].numpy(),128/255)
    assert np.allclose(y.numpy(),3.5)

def test_rejects_bad_shape():
    with pytest.raises(ValueError,match='shape'):
        preprocess(np.zeros((4,4,3),dtype=np.uint8))

def test_legacy_pillow_nearest_matches_current(monkeypatch):
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    depth = rng.random((480, 640), dtype=np.float32)
    expected_x, expected_y = preprocess(rgb, depth)
    legacy_image = SimpleNamespace(fromarray=Image.fromarray, NEAREST=Image.Resampling.NEAREST)
    monkeypatch.setattr(data, 'Image', legacy_image)
    actual_x, actual_y = preprocess(rgb, depth)
    assert np.array_equal(expected_x.numpy(), actual_x.numpy())
    assert np.array_equal(expected_y.numpy(), actual_y.numpy())
