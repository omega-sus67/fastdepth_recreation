"""Prepare arbitrary video frame shapes for the existing 640x480 model adapter."""
import cv2
import numpy as np


def adapt_frame(frame, mode='stretch'):
    """Return a 640x480 BGR frame and the geometry applied to the source.

    The model's own crop and resize are performed separately by preprocess().
    """
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError('Expected an 8-bit three-channel BGR video frame')
    height, width = frame.shape[:2]
    if height < 1 or width < 1:
        raise ValueError('Empty video frame')
    target_width, target_height = 640, 480
    info = {'source_width': width, 'source_height': height, 'mode': mode}
    if mode == 'stretch':
        image = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        info.update({'scale_x': target_width / width, 'scale_y': target_height / height})
    elif mode == 'letterbox':
        scale = min(target_width / width, target_height / height)
        scaled_width = max(1, min(target_width, round(width * scale)))
        scaled_height = max(1, min(target_height, round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        scaled = cv2.resize(frame, (scaled_width, scaled_height), interpolation=interpolation)
        left = (target_width - scaled_width) // 2
        top = (target_height - scaled_height) // 2
        image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        image[top:top + scaled_height, left:left + scaled_width] = scaled
        info.update({'resized_width': scaled_width, 'resized_height': scaled_height,
                     'padding_left': left, 'padding_top': top,
                     'padding_right': target_width - scaled_width - left,
                     'padding_bottom': target_height - scaled_height - top})
    elif mode == 'center-crop':
        if width * target_height > height * target_width:
            crop_width = max(1, height * target_width // target_height)
            crop_height = height
        else:
            crop_width = width
            crop_height = max(1, width * target_height // target_width)
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        crop = frame[top:top + crop_height, left:left + crop_width]
        interpolation = cv2.INTER_AREA if crop_width > target_width or crop_height > target_height else cv2.INTER_LINEAR
        image = cv2.resize(crop, (target_width, target_height), interpolation=interpolation)
        info.update({'crop_left': left, 'crop_top': top, 'crop_width': crop_width, 'crop_height': crop_height})
    else:
        raise ValueError(f'Unknown resize mode: {mode}')
    return image, info
