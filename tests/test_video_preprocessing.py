import cv2
import numpy as np
import pytest
from fastdepth_project.video import adapt_frame


def test_stretch_matches_previous_video_pipeline():
    frame = np.random.default_rng(0).integers(0, 256, (270, 480, 3), dtype=np.uint8)
    image, info = adapt_frame(frame, 'stretch')
    assert np.array_equal(image, cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR))
    assert info['scale_x'] != info['scale_y']


def test_letterbox_retains_16_9_and_records_padding():
    image, info = adapt_frame(np.full((720, 1280, 3), 200, dtype=np.uint8), 'letterbox')
    assert image.shape == (480, 640, 3)
    assert (info['resized_width'], info['resized_height']) == (640, 360)
    assert (info['padding_top'], info['padding_bottom']) == (60, 60)
    assert np.all(image[:60] == 0) and np.all(image[60:420] == 200)


def test_center_crop_removes_sides_without_padding():
    image, info = adapt_frame(np.full((720, 1280, 3), 200, dtype=np.uint8), 'center-crop')
    assert image.shape == (480, 640, 3) and np.all(image == 200)
    assert (info['crop_left'], info['crop_top'], info['crop_width'], info['crop_height']) == (160, 0, 960, 720)


def test_frame_and_mode_validation():
    with pytest.raises(ValueError): adapt_frame(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError): adapt_frame(np.zeros((10, 10, 3), dtype=np.uint8), 'bad')
