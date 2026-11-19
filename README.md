# FastDepth edge AI project

This project runs and studies monocular depth estimation using a recovered FastDepth checkpoint. It includes PyTorch inference, NYU-derived data tools, structured pruning experiments, ONNX exports, and TensorRT preparation for Jetson Orin Nano.

The desktop pipeline has been tested on an RTX 3060 Laptop GPU, and a four-architecture, three-precision sweep was run on a Jetson Orin Nano on 16 November 2026. The provenance and depth units of the community checkpoint remain unverified. The observed pruning timing differences are not a reliable speedup; INT8 loses accuracy.

The active experiment compares the community-mirrored, already-pruned baseline with three further-pruned students (192, 176 and 160 channels), each in FP32-configured, FP16-enabled and explicit INT8 modes. The Jetson run built all 12 engines and selected the 176-channel student; the observed pruning timing difference was not reliable, and INT8 accuracy was substantially degraded. A supervised-only recovery is retained as an earlier control.

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
bash scripts/python.sh scripts/infer_video.py path/to/video.mp4 --model student --side-by-side --output results/inference/my-video
bash scripts/python.sh scripts/demo_simple.py path/to/image.png --model baseline --output results/inference/my-baseline
bash scripts/python.sh scripts/demo_simple.py path/to/image.png --model student --output results/inference/my-student
```

For one input video with matched baseline and student predictions, run `bash scripts/python.sh scripts/compare_video.py path/to/video.mp4 --resize-mode center-crop --output results/inference/comparison`. It records per-stage timings and uses one shared color scale; see the locally retained `docs/video_pipeline.md`. The paired pipeline is desktop-tested and uses PyTorch.

Image input must be 640 pixels wide and 480 high. Choose a new output directory for each run. The image command writes numerical `depth.npy`, `preview.png` and metadata. Preview colors are normalized for display; the depth units are unverified. The video command does not retain audio. It supports `--resize-mode stretch` (original default), `letterbox` (padding) and `center-crop` (cropping); the model preprocessing afterward still resizes a 304×228 crop nonuniformly to 224×224. The selected student is width176; use `--model baseline` to compare. `--depth-range MIN MAX` sets a fixed visualization scale in unverified native units, and `--save-depth` saves numerical frame predictions. The pipeline uses PyTorch, not TensorRT; board video execution is unverified.

To reproduce the exploratory pruning and matched recovery comparison when the local training/tuning data are available:

```bash
bash scripts/python.sh scripts/run_simple_experiment.py --config configs/experiment_2026-11-16_simple_alpha01.json --output results/training/my-simple-experiment
```

This repeats the post-hoc 0.1 teacher-weight protocol, so it runs training and tuning evaluation and can take several minutes. Omitting `--config` uses the earlier 0.5 protocol. The script preserves a supervised-only control and does not perform final evaluation or claim target-device speedup. The orchestration sequence was checked with mocked subprocesses; the current full comparison was launched using the constituent scripts.

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
- `jetson_handoff/`: the prepared Jetson package, including its own start guide and evaluation inputs.
- `submissions/`: locally retained submission drafts and poster assets.

The current model checkpoints are the baseline in `checkpoints/converted/` and the 192-, 176- and 160-channel students in `results/training/2026-11-16-simple-distilled-alpha01/`, `2026-11-16-latency-176/` and `2026-11-16-latency-160/`. The selected demonstration student is width176. The supervised and alternative distillation controls are also retained. Earlier run summaries and settings remain as historical evidence, but older model binaries and desktop engines were retired; those old configurations cannot run again until their binaries are restored or regenerated. See [cleanup manifest](results/history/2026-11-19-cleanup/manifest.json) for the list and hashes. The old handoff archives were removed after their contents were checked against the current files; unique older small files were preserved with the manifest.

Project documentation and papers are also kept locally and are excluded from the current Git tree. Older Git history may still contain files previously committed. Dataset-dependent commands require acquiring the data separately; see [dataset guide](data/README.md).

## Jetson deployment

Verify the actual JetPack, CUDA, TensorRT and Python versions on the Orin Nano. Build TensorRT engines on the board from ONNX, then compare outputs and evaluate real inputs before reporting accuracy or speed. Desktop engine and latency measurements do not establish Jetson performance.
