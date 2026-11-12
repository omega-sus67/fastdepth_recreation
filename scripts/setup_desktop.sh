#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v uv >/dev/null; then
    echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2
    exit 1
fi
shared_config=(.venv/lib/python*/site-packages/edgeai_shared_env.pth)
if [ -f "${shared_config[0]}" ]; then
    # Preserve the existing shared-library setup and avoid downloading another GPU stack.
    echo 'Using the configured shared Python libraries.'
else
    uv venv --python "${PYTHON:-python3}" --allow-existing .venv
    # NVIDIA desktop GPU wheels; CUDA 12.8 requires a compatible NVIDIA driver.
    uv pip install --python .venv/bin/python 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' --index-url https://download.pytorch.org/whl/cu128
    uv pip install --python .venv/bin/python -r requirements-desktop.txt
fi
if [ ! -d third_party/fast-depth/.git ]; then
    mkdir -p third_party
    git clone https://github.com/dwofk/fast-depth.git third_party/fast-depth
    git -C third_party/fast-depth checkout e68492011609c9bfb7de6d402da5d1d201d95bd9
fi
mkdir -p data/nyu_depth_v2/raw checkpoints exports results
.venv/bin/python scripts/check_environment.py
