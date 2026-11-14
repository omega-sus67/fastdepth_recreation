# FastDepth edge AI project

This project runs and studies monocular depth estimation using a recovered FastDepth checkpoint. It includes PyTorch inference, NYU-derived data tools, structured pruning experiments, ONNX exports, and TensorRT preparation for Jetson Orin Nano.

The desktop pipeline has been tested on an RTX 3060 Laptop GPU. The Jetson has not yet been tested, and the provenance and depth units of the community checkpoint remain unverified. Current pruning experiments do not establish a reliable speedup or accuracy improvement.

## Desktop setup

The tested environment uses Python 3.12 and an NVIDIA GPU. From the repository root:

```bash
bash scripts/setup_desktop.sh --locked --with-tensorrt
bash scripts/python.sh scripts/check_environment.py --require-cuda --require-tensorrt --require-independent
bash scripts/python.sh -m pytest -q
```

The wrapper isolates project Python from inherited user and ROS paths. The desktop environment and its wheels are not a Jetson installation procedure.

## Image and video inference

```bash
bash scripts/python.sh scripts/infer_image.py path/to/image.png --output results/inference/my-image
bash scripts/python.sh scripts/infer_video.py path/to/video.mp4 --output results/inference/my-video
```

Image input must be 640 pixels wide and 480 high. Choose a new output directory for each run. The image command writes numerical `depth.npy`, `preview.png` and metadata. Preview colors are normalized for display; the depth units are unverified. The video command resizes frames and does not retain audio.

## Project layout

- `src/fastdepth_project/`: model loading, preprocessing, metrics and pruning.
- `scripts/`: setup, inference, data acquisition, training, evaluation and deployment tools; see [script guide](scripts/README.md).
- `configs/`: recorded dataset and experiment settings.
- `checkpoints/`: recovered checkpoint and model candidates.
- `exports/`: ONNX models for deployment experiments.
- `results/`: recorded desktop results and evidence; see [results guide](results/README.md).
- `tests/`: automated checks.
- `third_party/fast-depth/`: vendored upstream FastDepth code and license.
- `data/`: dataset navigation and acquisition context; downloaded datasets are kept locally and excluded from the current Git tree.

Project documentation and papers are also kept locally and are excluded from the current Git tree. Older Git history may still contain files previously committed. Dataset-dependent commands require acquiring the data separately; see [dataset guide](data/README.md).

## Jetson deployment

Verify the actual JetPack, CUDA, TensorRT and Python versions on the Orin Nano. Build TensorRT engines on the board from ONNX, then compare outputs and evaluate real inputs before reporting accuracy or speed. Desktop engine and latency measurements do not establish Jetson performance.
