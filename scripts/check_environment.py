#!/usr/bin/env python3
"""Report desktop setup and exercise a small operation on the selected device."""
import importlib
import platform
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
print(f"Python: {sys.version.split()[0]} ({sys.executable})")
print(f"Platform: {platform.platform()}")
missing = []
for name in ("torch", "torchvision", "numpy", "h5py", "matplotlib", "imageio", "skimage", "cv2", "onnx", "onnxruntime", "pytest", "huggingface_hub"):
    try:
        mod = importlib.import_module(name)
        print(f"{name}: {getattr(mod, '__version__', 'installed')}")
        if name in ("torch", "torchvision"):
            print(f"  loaded from: {mod.__file__}")
    except Exception as exc:
        missing.append(name)
        print(f"{name}: ERROR: {exc}")
if "torch" not in missing:
    import torch
    print(f"CUDA available: {torch.cuda.is_available()} (build: {torch.version.cuda})")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        x = torch.ones(128, 128, device="cuda")
        torch.cuda.synchronize()
        print(f"GPU matrix multiply: {(x @ x)[0, 0].item():.0f} (expected 128)")
print(f"Upstream source present: {(root / 'third_party/fast-depth/models.py').exists()}")
print(f"Checkpoint files: {len(list((root / 'checkpoints').glob('**/*.pth*')))}")
print(f"Validation archives: {len(list((root / 'data/nyu_depth_v2/raw').glob('val-*.tar')))}")
if missing:
    sys.exit(1)
