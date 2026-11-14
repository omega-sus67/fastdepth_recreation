#!/usr/bin/env bash
# Build an engine on the target Jetson; this command is not a performance benchmark.
set -euo pipefail
if [ "$#" -ne 3 ] || [[ "$3" != fp32 && "$3" != fp16 ]]; then
    echo 'Usage: bash scripts/build_jetson_engine.sh MODEL.onnx NEW_OUTPUT_DIRECTORY fp32|fp16' >&2
    exit 2
fi
model_path="$1"
output_dir="$2"
precision="$3"
if [ "$(uname -m)" != aarch64 ] || [ ! -f /etc/nv_tegra_release ]; then
    echo 'Run this script on the Jetson target after checking its software versions.' >&2
    exit 1
fi
if [ ! -f "$model_path" ]; then echo 'ONNX model not found.' >&2; exit 1; fi
if [ -e "$output_dir" ]; then echo 'Output already exists; choose a new directory.' >&2; exit 1; fi
trt_command="$(command -v trtexec || true)"
if [ -z "$trt_command" ] && [ -x /usr/src/tensorrt/bin/trtexec ]; then trt_command=/usr/src/tensorrt/bin/trtexec; fi
if [ -z "$trt_command" ]; then echo 'trtexec not found; check the Jetson TensorRT installation.' >&2; exit 1; fi
mkdir -p "$output_dir"
sha256sum "$model_path" > "$output_dir/onnx-sha256.txt"
{ uname -a; cat /etc/nv_tegra_release; python3 --version; printf 'trtexec=%s\nprecision=%s\n' "$trt_command" "$precision"; } > "$output_dir/environment.txt" 2>&1
options=("--onnx=$model_path" "--saveEngine=$output_dir/model.engine" --skipInference)
if [ "$precision" = fp16 ]; then options+=(--fp16); fi
printf '%q ' "$trt_command" "${options[@]}" > "$output_dir/command.txt"
printf '\n' >> "$output_dir/command.txt"
"$trt_command" "${options[@]}" 2>&1 | tee "$output_dir/build.log"
sha256sum "$output_dir/model.engine" > "$output_dir/engine-sha256.txt"
