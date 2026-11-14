# Phase A results — desktop setup and checkpoint acquisition

Verified on 12 November 2026. This records completed work and limitations; the [consolidated plan](fastdepth_edge_ai_plan.md) remains the implementation backlog.

The [progress report](progress_report.md) maintains cross-phase wins, failures and report-writing context, with a preserved snapshot of this phase's evidence. Its initial history was compiled on 13 November without rerunning these experiments.

## Status

- **A1 complete:** an independent Python environment, pinned dependencies, fresh-source installation, and CPU/GPU/export/runtime execution checks are verified.
- **A2 partial:** legacy loading, explicit architecture/weight conversion, and reproducible forward checks work for one recovered community pruned candidate. The original-author endpoints for all four variants remain unavailable; the three unpruned pretrained checkpoints and original-author authentication remain pending.
- Dataset accuracy, model-specific ONNX/TensorRT exports, training, and Orin validation are not claimed by these checks.

## Installed and exercised software

Recommended interpreter: `.venv-desktop/bin/python`, Python 3.12.3. Use `bash scripts/python.sh ...` to exclude inherited ROS/user Python paths. The environment has no dependency on `/home/omega_sus/droidrun/venv`.

| Component | Verified version | Execution evidence |
| --- | --- | --- |
| GPU / driver | RTX 3060 Laptop GPU, 6 GB; driver 580.178.04 | Device query and CUDA work |
| PyTorch / torchvision | 2.9.1+cu128 / 0.24.1+cu128 | CPU convolution and GPU matrix operation |
| CUDA runtime for PyTorch | 12.8 wheel stack | GPU execution; no separate system toolkit required |
| ONNX / ONNX Script | 1.23.2 / 0.7.2 | Modern `torch.onnx` dynamo export, opset 18 |
| ONNX Runtime | 1.31.1, CPU execution provider | Synthetic graph and exported convolution output parity |
| TensorRT | 10.3.0, CUDA 12 desktop wheels | Built and executed a synthetic FP32 engine on the GPU |
| Data and plotting | NumPy, h5py, SciPy, matplotlib, imageio, scikit-image, headless OpenCV | Import checks; versions/locations in environment record |
| Configuration, acquisition, tests | PyYAML, tqdm, huggingface-hub, pytest | Import checks and model/loader tests |

All 70 installed packages are pinned in `requirements-desktop-lock.txt`. The TensorRT requirement file lists its Python facade, bindings and libraries explicitly. `NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=1` avoids a nested duplicate installation; explicit NVIDIA and PyTorch indexes allow the lock to resolve actual wheels.

ONNX Runtime CPU is the reference path; a second ONNX Runtime GPU package is unnecessary for the planned PyTorch/TensorRT GPU comparisons and would replace the same Python module. TensorRT pip wheels provide the Python API, not `trtexec`. CUDA `nvcc`, ROS, TensorFlow and TFLite tools are not required for these Phase A checks. Jetson packages must be selected for the actual board's JetPack later.

## Fresh-source reproducibility

A source-only copy of the current setup scripts, vendored source and requirement files was placed in a temporary directory. Its independent environment was deleted and freshly recreated using:

```bash
bash scripts/setup_desktop.sh --locked --with-tensorrt
```

The installer created the environment, installed all 70 pinned packages, passed dependency consistency checks, and passed all five runtime checks. The run used a warm download cache; it did not copy installed Python packages or reuse a shared environment. It did not duplicate the dataset. The temporary source/environment copy was removed after saving the evidence.

Evidence:

- [Desktop environment and runtime results](../results/environment/desktop-environment.json).
- [Fresh-install results, isolation checks, exact command and source hashes](../results/environment/fresh-install.json).
- [Legacy environment preservation check](../results/environment/legacy-environment-check.json).
- [Model tests](../results/environment/model-tests.txt).

The previous project `.venv` package snapshot is unchanged. Its shared PyTorch/torchvision still load from the original location, and a CUDA operation passed afterward. The original shared environment was not used for the new environment's runtime results.

## Checkpoint recovery and conversion

Official HTTP/HTTPS requests were attempted for all four exact checkpoint filenames with bounded timeouts. All failed to connect. Searches covered upstream issue comments, the listed forks, Hugging Face model cards, a converted PINTO package and a Wayback CDX query. Alternative reimplementations and converted deployment artifacts were not substituted for original checkpoints.

A checkpoint from an upstream fork was recovered at an immutable commit and pinned by SHA-256. It contains a pruned `MobileNetSkipAdd` with **1,360,786 parameters**. Its saved metadata reports approximately 603.74 mm RMSE and 0.77107 delta1, consistent with upstream reference claims; these are historical metadata, not our measured evaluation results. An original-author checksum or checkpoint-specific license statement was not established, so it remains labeled `community_mirror_unverified`.

Conversion preserved the original file and generated a modern weight file plus explicit per-layer architecture, including all pruned channels and depthwise groups. The legacy file needs a restricted allowlisted unpickler because PyTorch 0.4's persistent module IDs are not accepted by modern `weights_only=True`. Converted files load with `weights_only=True`; unexpected globals/types and incorrect hashes fail.

Verified results:

- Input/output: `1 × 3 × 224 × 224` → `1 × 1 × 224 × 224`, finite values, on the desktop GPU.
- Converted versus legacy forward: maximum absolute difference **0.0** on the recorded synthetic input.
- Converted save/reload: bit-exact output on the same check.
- Vendored upstream RGB/TVM prediction example: mean absolute difference approximately **0.000336 m**, maximum approximately **0.003664 m**, passing `rtol=1e-3, atol=1e-3`.
- The three unpruned architectures produce valid shapes with **random weights only**. They are not pretrained reproduction results.
- Eight tests passed, covering architecture outputs, explicit topology/weight round trips, rejection of unknown types/globals, checksum enforcement, and refusal to silently use unpruned channels for the pruned variant.

A single upstream-example match is supporting compatibility evidence, not original-author authentication or a dataset accuracy measurement. See [checkpoint instructions and commands](../checkpoints/README.md), [source-search record](../results/checkpoints/source-search.json), [original endpoint failures](../results/checkpoints/original-source-attempts.json), [conversion report](../results/checkpoints/community-pruned-conversion.json), and [model checks](../results/checkpoints/model-checks.json).

## Cache and storage cleanup

Pip and uv package caches and project source bytecode/test caches were cleared after installation and fresh-source verification. The final cleanup record shows approximately **14.98 GiB of actual filesystem free space** recovered, including removal of the remaining pip cache directory tree; cache logical/allocated totals are larger because some installed packages share hardlinks with uv's cache. Installed environments, datasets, model files, papers, Git/LFS objects and Hugging Face artifact caches were preserved. The disposable verification environment was also removed.

The installed environment was exercised again after cleanup. Small runtime or bytecode caches may regenerate normally; they are ignored by Git. The full cleanup commands and before/after storage measurements are in [cache-cleanup.json](../results/environment/cache-cleanup.json).

## Persistent artifacts and next work

Keep source, data, checkpoint and result paths project-relative. Install Git LFS before retrieving the repository's large archive; use `git lfs pull` after a clone made with LFS smudging disabled. Environments are ignored; the recovered/converted model files and provenance reports remain project artifacts. The current desktop environment occupies substantial space for GPU runtimes, while the original shared environment remains untouched; avoid downloading another stack unless testing a deliberate alternative.

Proceed with the dataset/preprocessing audit, required EDA, metric implementation and full-split evaluation of the community candidate under its explicit provenance label. Continue bounded acquisition of the missing original variants alongside that independent work. See [immediate priorities](immediate_priority.md).
