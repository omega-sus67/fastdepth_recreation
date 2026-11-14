#!/usr/bin/env bash
# Keep inherited ROS/user Python packages out of this project's independent environment.
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
project_python="${EDGEAI_PYTHON:-$project_root/.venv-desktop/bin/python}"
if [ ! -x "$project_python" ]; then
    echo 'Run bash scripts/setup_desktop.sh --with-tensorrt first.' >&2
    exit 1
fi
exec env PYTHONPATH="$project_root/src" PYTHONNOUSERSITE=1 "$project_python" "$@"
