"""FastDepth validation transforms adapted to modern Pillow and NumPy."""
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
import torch


def preprocess(rgb, depth=None):
    rgb = np.asarray(rgb)
    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB image with shape (480, 640, 3)')
    if depth is not None and np.asarray(depth).shape != (480, 640):
        raise ValueError('Expected depth image with shape (480, 640)')
    # Pillow < 9.1 exposes NEAREST on Image instead of Image.Resampling.
    nearest = getattr(Image, 'Resampling', Image).NEAREST
    def transform(array):
        image = Image.fromarray(array)
        # scipy.misc.imresize's float scale truncates the scaled dimensions.
        image = image.resize((int(640 * 250.0 / 480), 250), nearest)
        left = int(round((image.width - 304) / 2.0))
        top = int(round((image.height - 228) / 2.0))
        image = image.crop((left, top, left + 304, top + 228))
        return np.asarray(image.resize((224, 224), nearest))
    rgb_out = transform(rgb).astype(np.float32) / 255.0
    x = torch.from_numpy(np.ascontiguousarray(rgb_out.transpose(2, 0, 1)))
    if depth is None:
        return x
    y = torch.from_numpy(transform(np.asarray(depth, dtype=np.float32)).copy()).unsqueeze(0)
    return x, y


def load_h5(path):
    with h5py.File(Path(path), 'r') as f:
        rgb = np.asarray(f['rgb']).transpose(1, 2, 0)
        depth = np.asarray(f['depth'])
    return rgb, depth
