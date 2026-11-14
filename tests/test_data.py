import numpy as np
import pytest
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
