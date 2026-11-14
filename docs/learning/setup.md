# Setup and environments

## What the environment provides

Python executes the project scripts. PyTorch defines tensors, neural-network layers, automatic differentiation and GPU execution. CUDA is NVIDIA's GPU computation platform; PyTorch's CUDA wheels include runtime libraries, while the installed GPU driver provides access to the hardware. ONNX describes a model graph independently of PyTorch. ONNX Runtime executes that graph; TensorRT builds an optimized engine for a particular runtime and GPU target.

The verified desktop environment is `.venv-desktop`, Python 3.12.3, PyTorch 2.9.1 with CUDA 12.8 wheels, torchvision 0.24.1 and TensorRT 10.3. It is distinct from the earlier `.venv`, which accesses another project's packages. `requirements-desktop-lock.txt` records 70 pinned desktop packages. This is an x86 desktop stack; the Orin's ARM/JetPack packages must be selected on the actual board.

## Commands and isolation

Run `bash scripts/setup_desktop.sh --locked --with-tensorrt` to recreate the tested package set, and run Python commands through `bash scripts/python.sh ...`. The wrapper sets the project import path and suppresses inherited user-site packages. This matters because the shell previously exposed ROS Python packages; pytest loaded an incompatible ROS plugin as a result. The wrapper fixed that issue without changing ROS.

The environment checker performs synthetic CPU and CUDA computations, exports a tiny convolution to ONNX, runs it in ONNX Runtime and builds/executes a tiny TensorRT engine. Passing confirms those tools work for the test graph; it does not prove that FastDepth exports or runs on Orin. See [Phase A results](../phase_a_results.md) for versions, raw evidence, previous failures and cache cleanup.
