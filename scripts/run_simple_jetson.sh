#!/usr/bin/env bash
# Fixed comparison; run on the target after validating dependencies.
set -uo pipefail
if [ "$#" -ne 1 ]; then echo 'Usage: bash scripts/run_simple_jetson.sh NEW_RESULT_DIRECTORY' >&2; exit 2; fi
out="$1"
if [ "$(uname -m)" != aarch64 ] || [ ! -f /etc/nv_tegra_release ]; then echo 'Run on the target Jetson.' >&2; exit 1; fi
if [ -e "$out" ]; then echo 'Choose a new result directory.' >&2; exit 1; fi
if ! python3 -c 'import numpy, tensorrt, torch; assert torch.cuda.is_available()'; then echo 'Board Python dependencies or CUDA unavailable.' >&2; exit 1; fi
mkdir -p "$out"
{ date -Is; uname -a; cat /etc/nv_tegra_release; cat /proc/device-tree/model; printf '\n'; python3 --version; python3 -c 'import numpy,tensorrt,torch; print("numpy",numpy.__version__,"tensorrt",tensorrt.__version__,"torch",torch.__version__,"cuda",torch.version.cuda,"gpu",torch.cuda.get_device_name(0))'; nvpmodel -q; } > "$out/environment.log" 2>&1
stats_pid=''
if command -v tegrastats >/dev/null; then tegrastats > "$out/tegrastats.log" 2>&1 & stats_pid=$!; fi
cleanup() { if [ -n "$stats_pid" ]; then kill "$stats_pid" 2>/dev/null || true; wait "$stats_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT INT TERM
printf 'model,precision,build_status,evaluation_status\n' > "$out/status.csv"
failed=0
for entry in 'baseline:exports/simple-baseline/model.onnx' 'student:exports/simple-student/model.onnx'; do
  name="${entry%%:*}"; model="${entry#*:}"
  for precision in fp32 fp16; do
    build="$out/${name}_${precision}_build"; evaluation="$out/${name}_${precision}_evaluation"
    echo "Building $name $precision"
    if bash scripts/build_jetson_engine.sh "$model" "$build" "$precision"; then
      echo "Evaluating $name $precision"
      if python3 scripts/evaluate_jetson_bundle.py --engine "$build/model.engine" --data data/nyu_depth_v2/jetson-evaluation-654 --reference data/nyu_depth_v2/simple-reference-654/"$name".npy --output "$evaluation" > "$out/${name}_${precision}_evaluation.log" 2>&1; then
        printf '%s,%s,ok,ok\n' "$name" "$precision" >> "$out/status.csv"
      else
        printf '%s,%s,ok,failed\n' "$name" "$precision" >> "$out/status.csv"; failed=1
      fi
    else
      printf '%s,%s,failed,not_run\n' "$name" "$precision" >> "$out/status.csv"; failed=1
    fi
  done
done
cat "$out/status.csv"
exit "$failed"
