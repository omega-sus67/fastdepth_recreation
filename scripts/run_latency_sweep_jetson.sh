#!/usr/bin/env bash
# Run from the project root on the target. Existing results are never overwritten.
set -uo pipefail
if [ "$#" -ne 1 ]; then echo 'Usage: bash scripts/run_latency_sweep_jetson.sh NEW_RESULT_DIRECTORY' >&2; exit 2; fi
out="$1"
if [ "$(uname -m)" != aarch64 ] || [ ! -f /etc/nv_tegra_release ]; then echo 'Run on the Jetson Orin Nano.' >&2; exit 1; fi
if [ -e "$out" ]; then echo 'Choose a new result directory.' >&2; exit 1; fi
if ! python3 -c 'import numpy, tensorrt, torch; assert torch.cuda.is_available()'; then echo 'Python, TensorRT or CUDA unavailable.' >&2; exit 1; fi
for path in configs/experiment_2026-11-16_latency_sweep.json data/nyu_depth_v2/jetson-tuning-445/manifest.json data/nyu_depth_v2/jetson-evaluation-654/manifest.json data/nyu_depth_v2/latency-tuning-references/manifest.json data/nyu_depth_v2/latency-evaluation-references/manifest.json; do
  if [ ! -f "$path" ]; then echo "Missing $path; see the handoff guide." >&2; exit 1; fi
done
mkdir -p "$out/build" "$out/tuning" "$out/evaluation" "$out/logs/build" "$out/logs/tuning" "$out/logs/evaluation"
{ date -Is; uname -a; cat /etc/nv_tegra_release; cat /proc/device-tree/model; printf '\n'; python3 --version; python3 -c 'import numpy,tensorrt,torch; print("numpy",numpy.__version__,"tensorrt",tensorrt.__version__,"torch",torch.__version__,"cuda",torch.version.cuda,"gpu",torch.cuda.get_device_name(0))'; nvpmodel -q; } > "$out/environment.log" 2>&1
stats_pid=''
if command -v tegrastats >/dev/null; then tegrastats --interval 1000 > "$out/tegrastats.log" 2>&1 & stats_pid=$!; fi
cleanup() { if [ -n "$stats_pid" ]; then kill "$stats_pid" 2>/dev/null || true; wait "$stats_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf 'model,precision,build_status,tuning_status\n' > "$out/status.csv"
failed=0
for name in baseline width192 width176 width160; do
  case "$name" in
    baseline) float_model=exports/simple-baseline/model.onnx ;;
    width192) float_model=exports/simple-student/model.onnx ;;
    *) float_model="exports/sweep/${name}-fp32/model.onnx" ;;
  esac
  for precision in fp32 fp16 int8; do
    id="${name}_${precision}"
    build="$out/build/$id"
    echo "[$(date -Is)] Building $id"
    if [ "$precision" = int8 ]; then
      bash scripts/build_jetson_qdq.sh "exports/sweep/${name}-int8-floatbias/model.onnx" "$build" > "$out/logs/build/${id}.log" 2>&1
    else
      bash scripts/build_jetson_engine.sh "$float_model" "$build" "$precision" > "$out/logs/build/${id}.log" 2>&1
    fi
    if [ "$?" -ne 0 ]; then
      echo "Build failed: $id; see $out/logs/build/${id}.log"
      printf '%s,%s,failed,not_run\n' "$name" "$precision" >> "$out/status.csv"; failed=1; continue
    fi
    echo "[$(date -Is)] Evaluating $id on tuning data"
    if python3 scripts/evaluate_jetson_bundle.py --engine "$build/model.engine" --data data/nyu_depth_v2/jetson-tuning-445 --reference "data/nyu_depth_v2/latency-tuning-references/$name.npy" --output "$out/tuning/$id" --timing-rounds 3 > "$out/logs/tuning/${id}.log" 2>&1; then
      printf '%s,%s,ok,ok\n' "$name" "$precision" >> "$out/status.csv"
    else
      echo "Tuning failed: $id; see $out/logs/tuning/${id}.log"
      printf '%s,%s,ok,failed\n' "$name" "$precision" >> "$out/status.csv"; failed=1
    fi
  done
done
if python3 scripts/select_latency_candidate.py --config configs/experiment_2026-11-16_latency_sweep.json --runs "$out/tuning" --output "$out/selection.json" > "$out/selection.log" 2>&1; then
  selected="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_student"] or "")' "$out/selection.json")"
else
  echo "Selection failed; see $out/selection.log"; selected=''; failed=1
fi
if [ -z "$selected" ]; then echo 'No student selected; evaluation will include the baseline only.'; failed=1; fi
printf 'model,precision,evaluation_status\n' > "$out/evaluation-status.csv"
for name in baseline $selected; do
  for precision in fp32 fp16 int8; do
    id="${name}_${precision}"
    if [ ! -f "$out/build/$id/model.engine" ]; then
      printf '%s,%s,missing_engine\n' "$name" "$precision" >> "$out/evaluation-status.csv"; failed=1; continue
    fi
    echo "[$(date -Is)] Evaluating $id on the separate 654-file collection"
    if python3 scripts/evaluate_jetson_bundle.py --engine "$out/build/$id/model.engine" --data data/nyu_depth_v2/jetson-evaluation-654 --reference "data/nyu_depth_v2/latency-evaluation-references/$name.npy" --output "$out/evaluation/$id" --timing-rounds 3 > "$out/logs/evaluation/${id}.log" 2>&1; then
      printf '%s,%s,ok\n' "$name" "$precision" >> "$out/evaluation-status.csv"
    else
      printf '%s,%s,failed\n' "$name" "$precision" >> "$out/evaluation-status.csv"; failed=1
    fi
  done
done
cat "$out/status.csv" "$out/evaluation-status.csv"
echo "Results and logs: $out"
exit "$failed"
