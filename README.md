# FastDepth edge AI project

This project recreates and experiments with monocular depth estimation using a recovered FastDepth model. Given an RGB image, it predicts a depth map. It includes a desktop inference pipeline, NYU-derived datasets, structured-pruning experiments, ONNX exports, and documentation of both successful and unsuccessful results.

**Status:** Desktop experiments run on an RTX 3060 Laptop GPU. The target is Jetson Orin Nano, but the model and performance have not yet been verified on the board. The recovered community checkpoint's original-author identity and the depth units remain unverified; our experiments have not established a reliable hardware speedup.

The deployment target is **Jetson Orin Nano**. While the board is unavailable, use the desktop RTX 3060 Laptop GPU (6 GB) to build and validate inference, evaluation, training, pruning tools, exports, and deployment packaging.

Start with [immediate_priority.md](docs/immediate_priority.md), the current actionable queue. Follow [agents.md](agents.md) when working on the project and updating that queue. The [complete project plan](docs/fastdepth_edge_ai_plan.md) holds the implementation backlog and evaluator requirements. The local setup targets the desktop GPU; it does not install Jetson packages.

For quick navigation: [project story](docs/project_story.md) · [experiment index](docs/experiment_index.md) · [learning notes](docs/learning/README.md) · [dataset guide](data/README.md) · [script guide](scripts/README.md) · [results guide](results/README.md) · [papers](papers/README.md).

## Start here

The recommended environment is the independent **`.venv-desktop`**, using Python 3.12, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, ONNX Runtime CPU, and TensorRT 10.3.0 on the desktop GPU. Exact installed versions are recorded in `requirements-desktop-lock.txt`.

```bash
# First installation, or reproduce the tested desktop package set:
bash scripts/setup_desktop.sh --locked --with-tensorrt

# Run with project packages isolated from the shell's ROS/user Python paths:
bash scripts/python.sh scripts/check_environment.py --require-cuda --require-tensorrt --require-independent
bash scripts/python.sh -m pytest -q
bash scripts/python.sh scripts/inspect_validation.py

# Check the available converted community checkpoint and the random architectures:
bash scripts/python.sh scripts/check_models.py --structural --checkpoint checkpoints/converted/community-skipadd-pruned.pth
```

Use `bash scripts/python.sh ...` for project commands: the shell exposes unrelated ROS Python packages, so activating a venv alone does not fully isolate its import path. For notebooks select `.venv-desktop/bin/python` and clear inherited `PYTHONPATH` before launching. `EDGEAI_PYTHON` can override the wrapper's interpreter with another independent environment.

`--venv PATH` selects an installation location. Without `--locked`, setup installs the pinned PyTorch/torchvision pair plus the current core requirements; `--with-tensorrt` adds the separately pinned TensorRT requirements. The full lock includes TensorRT. The installer disables NVIDIA's nested pip installation and uses explicit official wheel indexes. A compatible NVIDIA driver, Python 3.12, and [uv](https://docs.astral.sh/uv/getting-started/installation/) are required. Binary wheels supply the tested CUDA runtimes; a system-wide CUDA toolkit or `nvcc` is not needed for these checks. ONNX Runtime's CPU provider is intentional for reference parity; GPU execution uses PyTorch and the TensorRT Python API. The pip TensorRT installation does not supply `trtexec`.

The previous `.venv` and the shared packages in `/home/omega_sus/droidrun/venv` are preserved. They are no longer the recommended installation; `requirements-desktop-tested.txt` remains a historical snapshot. To check the old environment without changing it, use `bash scripts/setup_desktop.sh --venv .venv --check-only`. Desktop wheels are not Jetson wheels: install board packages appropriate to the actual JetPack separately.

See the [Phase A results](docs/phase_a_results.md) for reproducibility checks, model provenance, cache cleanup, and the remaining checkpoint limitations.

The [living progress report](docs/progress_report.md) records wins, failed approaches, fixes and evidence gaps for writing the final project report. Its [evidence index](results/evidence_index.csv) connects claims to preserved artifacts. Update it alongside meaningful implementation or experimental progress, following the working rules.

## Layout

- `docs/immediate_priority.md` — current next steps, rewritten as work completes or priorities change.
- `agents.md` — project working rules; `AGENTS.md` makes them discoverable by coding agents.
- `docs/` — the consolidated project plan, immediate priorities, evaluator guidelines, and poster reference.
- `papers/` — reference papers, problem statement and supplied board screenshots; see its [index](papers/README.md).
- `third_party/fast-depth/` — original upstream source, pinned by the setup script to commit `e68492011609c9bfb7de6d402da5d1d201d95bd9`; includes an example RGB/depth pair under `deploy/data/`.
- `data/` — downloaded NYU-derived validation archives, evaluation files and small training/tuning subsets; see the [dataset guide](data/README.md) for their different roles and limitations.
- `checkpoints/` — preserved legacy community candidate, converted weights, and explicit channel configuration; see its README for provenance and missing original models.
- `src/`, `scripts/`, `tests/`, `notebooks/` — implementation, [commands](scripts/README.md), tests and notebooks.
- `configs/`, `exports/`, `results/` — experiment settings, exported models and [measurements](results/README.md). The data inspection preview is `results/validation_sample.png`.

The 654 validation files have also been extracted under `data/nyu_depth_v2/extracted/val`. The project evaluation adapter supports the recovered checkpoint and local data; see the commands below and the [dataset guide](data/README.md).

Small scene-grouped training/tuning subsets are present locally; the full NYU training dataset is not. Dataset files are excluded from the current Git tree and must be acquired separately; see the [dataset guide](data/README.md). Project documents, papers, results, model artifacts and vendored source remain in version control. Older Git history may still contain dataset files.

The original FastDepth code targets PyTorch 0.4.1. A restricted legacy loader and explicit architecture/weight conversion now run the recovered community pruned model without modifying upstream code. Its original-author provenance and the local evaluation split's identity are still unverified. The three unpruned architectures pass random-weight structural checks, but their pretrained weights are unavailable. Project adapters now support local evaluation, image/video inference, exploratory training and model-specific ONNX export; these are desktop results, not proof of successful board deployment. See the [experiment index](docs/experiment_index.md) for measured results and limitations.

## Deployment boundary

Choose the software stack for **Orin Nano**, then verify the actual module, JetPack, CUDA, TensorRT, memory, and power mode when board access is available. Original-Nano JetPack 4.6/TensorRT 8.2 assumptions do not apply. FP32 and FP16 are required comparisons; INT8 is optional and needs calibration and measured accuracy/latency. See [NVIDIA JetPack documentation](https://developer.nvidia.com/embedded/jetpack-sdk-62).

Export portable ONNX models and prepare engine-building commands on the desktop. Build final TensorRT engines on the Orin Nano and validate outputs there; desktop engines and desktop benchmark numbers do not establish board compatibility or performance. See [NVIDIA engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html). Final latency, memory, power, thermal behavior, energy, and hardware-aware pruning selection require the board.

## Desktop inference and evaluation

The current community-checkpoint baseline and its limitations are documented in [metrics](docs/learning/metrics.md) and [progress history](docs/progress_report.md). Original-author provenance, the mirror's complete depth conversion chain and split identity remain unverified. Adapted SciPy source matched the final preprocessing tensors on six tested cases; this was not a historical binary-environment reproduction.

```bash
bash scripts/python.sh scripts/infer_image.py path/to/480x640-rgb.png --output results/inference/my-image
bash scripts/python.sh scripts/infer_video.py path/to/video.mp4 --output results/inference/my-video
bash scripts/python.sh scripts/evaluate_validation.py --output results/evaluation/my-run
```

Use a new output directory for each run. Image inference requires a 480×640 source image; video inference resizes frames to that size and can change aspect ratio. Preview colors are normalized separately from raw depth, and their units are not yet verified.
