#!/usr/bin/env bash
# Install the desktop GPU stack without changing another project's environment.
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH
export PYTHONNOUSERSITE=1
project_env=".venv-desktop"
with_tensorrt=false
check_only=false
locked=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        --venv) project_env="${2:?--venv requires a path}"; shift 2 ;;
        --with-tensorrt) with_tensorrt=true; shift ;;
        --check-only) check_only=true; shift ;;
        --locked) locked=true; shift ;;
        -h|--help)
            echo 'Usage: scripts/setup_desktop.sh [--venv PATH] [--with-tensorrt] [--locked] [--check-only]'
            echo 'Default: independent .venv-desktop with Python 3.12 and PyTorch 2.9.1/CUDA 12.8.'
            echo '--locked uses the recorded desktop package versions; --with-tensorrt adds TensorRT 10.3.'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
if [ "$check_only" = false ]; then
    command -v uv >/dev/null || { echo 'Install uv: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
    shared_config=("$project_env"/lib/python*/site-packages/edgeai_shared_env.pth)
    if [ -f "${shared_config[0]}" ]; then
        echo 'This environment shares another project’s packages. Use --check-only or an independent --venv path.' >&2
        exit 1
    fi
    if [ ! -x "$project_env/bin/python" ]; then
        uv venv --python "${PYTHON:-python3.12}" "$project_env"
    fi
    if [ "$locked" = true ]; then
        NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=1 uv pip sync --python "$project_env/bin/python" --index-url https://pypi.org/simple \
            --extra-index-url https://download.pytorch.org/whl/cu128 \
            --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match \
            requirements-desktop-lock.txt
    else
        uv pip install --python "$project_env/bin/python" 'torch==2.9.1' 'torchvision==0.24.1' \
            --index-url https://download.pytorch.org/whl/cu128
        uv pip install --python "$project_env/bin/python" -r requirements-desktop.txt
    fi
    if [ "$with_tensorrt" = true ]; then
        NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=1 uv pip install --python "$project_env/bin/python" -r requirements-tensorrt.txt
    fi
    uv pip check --python "$project_env/bin/python"
fi
if [ ! -f third_party/fast-depth/models.py ]; then
    mkdir -p third_party
    git clone https://github.com/dwofk/fast-depth.git third_party/fast-depth
    git -C third_party/fast-depth checkout e68492011609c9bfb7de6d402da5d1d201d95bd9
fi
mkdir -p data/nyu_depth_v2/raw checkpoints exports results/environment
check_args=(--require-cuda)
shared_config=("$project_env"/lib/python*/site-packages/edgeai_shared_env.pth)
if [ ! -f "${shared_config[0]}" ]; then check_args+=(--require-independent); fi
if [ "$with_tensorrt" = true ]; then check_args+=(--require-tensorrt); fi
"$project_env/bin/python" scripts/check_environment.py "${check_args[@]}" \
    --output results/environment/desktop-environment.json
