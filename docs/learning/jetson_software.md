# Jetson Orin Nano: reported software and deployment implications

Recorded 13 November 2026 from two user-supplied screenshots: [picture 1](../../papers/WhatsApp%20Image%202026-11-13%20at%2012.13.16.jpeg) and [picture 2](../../papers/WhatsApp%20Image%202026-11-13%20at%2012.13.17.jpeg). They show a **chat summary**, not raw terminal output. These are reported values to guide preparation; they are not independently verified board measurements. The screen is partially cropped/covered, so do not infer hidden details.

| Component | Screenshot reports | Interpretation / action |
| --- | --- | --- |
| Board | Jetson Orin Nano Engineering Reference Developer Kit | Record the exact module/carrier and board identification when available. |
| Memory | 8 GB; the visible text for memory available to Linux is cropped | Treat 8 GB as reported total; do not guess the usable amount. GPU and CPU memory reporting must account for shared memory. |
| Jetson Linux / L4T | 36.4.7 | Record as reported, not independently confirmed. |
| JetPack package | 6.2.1 | Record the package and BSP independently; see version discrepancy below. |
| CUDA | 12.6 | Target the board's CUDA 12.6 stack; desktop CUDA 12.8 wheels are not Jetson packages. |
| TensorRT | 10.3.0 | Desktop TensorRT is also 10.3.0; matching versions do not make desktop engines portable or prove ONNX/operator compatibility. |
| Python | 3.10.12 | Board scripts and libraries need Python 3.10-compatible syntax/dependencies; desktop uses Python 3.12.3. |
| TensorRT Python bindings | 10.3.0 | Reported available; check import and runtime version in the actual interpreter. |
| `trtexec` | Installed | Reported available; find its real path and version. The desktop pip TensorRT installation does not include it. |
| PyTorch / torchvision | Not visible in the screenshots | Unknown: inspect the board; do not assume a compatible CUDA-enabled PyTorch exists. TensorRT/ONNX inference can be prepared independently. |

## L4T / JetPack discrepancy

[NVIDIA's JetPack archive](https://developer.nvidia.com/embedded/jetpack-archive) associates the standard JetPack 6.2.1 release with Jetson Linux/L4T 36.4.4; the screenshots pair the JetPack package 6.2.1 with L4T 36.4.7. This is a **discrepancy to investigate**, not proof that the board is broken or that either reported value is false: an updated BSP and an installed package version can differ. Do not reflash, upgrade, downgrade or install board packages on the basis of this discrepancy. Record exact package versions and determine provenance when the board is accessible.

## What each software layer does

- **Jetson Linux / L4T:** board-specific Linux distribution components, drivers, kernel and boot support. Its release identifies the base platform.
- **JetPack:** NVIDIA's collection of platform and AI components. The installed package label and the actual versions of individual components should be recorded separately.
- **CUDA:** programming/runtime stack used for NVIDIA GPU computation. A CUDA version shown by a package or `nvcc` is not by itself proof that a Python runtime successfully executed a GPU kernel.
- **TensorRT:** inference optimizer and runtime. It parses a supported model graph, chooses tactics, and builds a serialized engine using the target's GPU/runtime. `trtexec` is a command-line utility for building and measuring engines; Python bindings expose a programmatic interface.
- **Python:** interpreter running our scripts. A Python wheel built for desktop Python 3.12/x86-64 is not interchangeable with a Python 3.10/AArch64 board wheel.

## Read-only board verification when access is available

Run commands on the board and save unedited output with date, model and context. A failed/missing command is a finding, not a reason to install or modify packages automatically:

```bash
uname -a
uname -m
cat /etc/nv_tegra_release
cat /proc/device-tree/model
dpkg-query -W nvidia-l4t-core nvidia-jetpack 'libnvinfer*' 'cuda-cudart*'
python3 --version
python3 -c 'import tensorrt; print(tensorrt.__version__)'
python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
command -v trtexec
free -h
```

Package names/wildcards and `trtexec` location can differ; a query failing does not prove a component is absent. If the command is not on `PATH`, search the installed TensorRT package file list with the actual package name. Verify an actual GPU operation if a compatible PyTorch is present. Save full output and errors, not only a summary. Record ambient conditions, cooling, power mode and clocks separately for measurements.

## Deployment plan derived from these values

1. Use the existing validated ONNX export as a starting artifact; test whether its opset 18 and actual operators are accepted by this board's TensorRT parser. Desktop ONNX Runtime parity on one image is preliminary evidence only.
2. On the board, build FP32 and FP16-enabled engines from that ONNX using its installed TensorRT 10.3.0; save parser/build settings, engine and model hashes. Do not transfer the desktop `.engine` as the board engine. The FP16 flag permits mixed tactics, so inspect the generated engine where possible.
3. Check numerical output for identical preprocessed inputs against desktop FP32; run accuracy evaluation using the same data and metric definitions. Python-based board inference requires dependencies built for Python 3.10 and ARM64, and possibly a board-compatible PyTorch; `trtexec` alone does not provide our complete validation and preprocessing pipeline.
4. Measure repeated model-only and full-pipeline latency separately, with synchronization, warm-up, raw samples, power mode and temperature. The screenshot is not evidence that the model runs or meets the 33 ms pipeline budget.
5. If INT8 is attempted, use representative training-only calibration data and evaluate depth accuracy. No calibrated model or INT8 performance claim exists yet.
