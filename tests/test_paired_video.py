import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('compare_video', Path(__file__).parents[1] / 'scripts/compare_video.py')
video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video)


def test_common_scale_uses_both_models_and_clips():
    frames = [{'baseline': np.ones((2, 2), dtype=np.float32),
               'student': np.full((2, 2), 3, dtype=np.float32)}]
    low, high = video.percentile_scale(frames)
    assert (low, high) == (1, 3)
    assert np.array_equal(video.colorize(np.array([[0]], dtype=np.float32), low, high),
                          video.colorize(np.array([[1]], dtype=np.float32), low, high))
    assert np.array_equal(video.colorize(np.array([[4]], dtype=np.float32), low, high),
                          video.colorize(np.array([[3]], dtype=np.float32), low, high))


def test_constant_predictions_have_finite_nonzero_range():
    array = np.full((2, 2), 2, dtype=np.float32)
    low, high = video.percentile_scale([{'baseline': array, 'student': array}])
    assert low == 2 and np.isfinite(high) and high > low


def test_panel_shows_actual_shared_color_legend():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    output = video.panel(image, 'Depth', 1.0, 3.0)
    assert output.shape == (276, 224, 3)
    assert np.array_equal(output[42, 5], video.colorize(np.array([[1]], dtype=np.float32), 1, 3)[0, 0])
    assert np.array_equal(output[42, 218], video.colorize(np.array([[3]], dtype=np.float32), 1, 3)[0, 0])
