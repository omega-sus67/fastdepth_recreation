#!/usr/bin/env bash
# Profile and repeatedly benchmark existing engines; optional PyTorch visualizations.
set -uo pipefail
if [ "$#" -gt 2 ] || { [ "$#" -eq 2 ] && [ "$1" != '--video' ]; } || [ "$#" -eq 1 ]; then
  echo 'Usage: bash run_orin_followup.sh [--video INPUT_VIDEO]' >&2; exit 2
fi
video=''
if [ "$#" -eq 2 ]; then video="$2"; fi
# Run from the handoff root; existing engines and selection are relative to it.
if [ ! -f scripts/select_latency_candidate.py ]; then echo 'First cd into jetson_handoff; run bash run_orin_followup.sh there.' >&2; exit 1; fi
if [ "$(uname -m)" != aarch64 ]; then echo 'Run on the Jetson Orin Nano.' >&2; exit 1; fi
trt_command="$(command -v trtexec || true)"
if [ -z "$trt_command" ] && [ -x /usr/src/tensorrt/bin/trtexec ]; then trt_command=/usr/src/tensorrt/bin/trtexec; fi
if [ -z "$trt_command" ]; then echo 'trtexec not found; activate the correct Jetson environment.' >&2; exit 1; fi
baseline='results/orin-sweep-visit/build/baseline_fp16/model.engine'
student='results/orin-sweep-visit/build/width176_fp16/model.engine'
if [ ! -f "$baseline" ] || [ ! -f "$student" ]; then echo 'Expected baseline and width176 FP16 engines from results/orin-sweep-visit are missing.' >&2; exit 1; fi
mkdir -p results
run="$(mktemp -d results/orin-followup-XXXXXXXX)" || exit 1
printf 'Results: %s\n' "$run"
printf 'step,status,log\n' > "$run/status.csv"
{ date -Is; uname -a; cat /etc/nv_tegra_release; cat /proc/device-tree/model; printf '\n'; nvpmodel -q; python3 --version; python3 -c 'import numpy,tensorrt,torch; print("numpy",numpy.__version__,"tensorrt",tensorrt.__version__,"torch",torch.__version__,"cuda",torch.version.cuda,"available",torch.cuda.is_available())'; } > "$run/environment.log" 2>&1
sha256sum "$baseline" "$student" > "$run/engine-sha256.txt"
stats_pid=''
if command -v tegrastats >/dev/null; then tegrastats --interval 1000 > "$run/tegrastats.log" 2>&1 & stats_pid=$!; fi
cleanup() { if [ -n "$stats_pid" ]; then kill "$stats_pid" 2>/dev/null || true; wait "$stats_pid" 2>/dev/null || true; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
core_failed=0
run_test() {
  label="$1"; engine="$2"; profile="$3"
  log="$run/$label.log"
  printf '[%s] %s started\n' "$(date -Is)" "$label" | tee -a "$run/timeline.log"
  args=("--loadEngine=$engine" --warmUp=1000 --duration=10)
  if [ "$profile" = yes ]; then args+=(--dumpProfile); fi
  if "$trt_command" "${args[@]}" > "$log" 2>&1; then
    printf '%s,ok,%s\n' "$label" "$log" >> "$run/status.csv"
    echo "$label: OK"
  else
    code=$?; printf '%s,failed_%s,%s\n' "$label" "$code" "$log" >> "$run/status.csv"
    echo "$label: FAILED; see $log" >&2; core_failed=1
  fi
  printf '[%s] %s ended\n' "$(date -Is)" "$label" | tee -a "$run/timeline.log"
}
run_test baseline_profile "$baseline" yes
run_test width176_profile "$student" yes
run_test baseline_repeat1 "$baseline" no
run_test width176_repeat1 "$student" no
run_test width176_repeat2 "$student" no
run_test baseline_repeat2 "$baseline" no
selection='results/orin-sweep-visit/selection.json'
visual_log="$run/visual-comparison.log"
if [ -f scripts/compare_visual_photos.py ] && [ -f visual_photos/manifest.json ] && [ -f "$selection" ]; then
  echo 'Generating visual comparisons (requires PyTorch image dependencies)...'
  if python3 scripts/compare_visual_photos.py --selection "$selection" --output "$run/visual-comparison" --device cuda > "$visual_log" 2>&1; then
    printf 'visual_comparison,ok,%s\n' "$visual_log" >> "$run/status.csv"
  else
    code=$?; printf 'visual_comparison,failed_%s,%s\n' "$code" "$visual_log" >> "$run/status.csv"
    echo "Visual comparison failed; see $visual_log. Profiling results are preserved." >&2
  fi
else
  printf 'visual_comparison,skipped,missing_script_photos_or_selection\n' >> "$run/status.csv"
  echo 'Visual comparison skipped: updated script, photos or selection is missing.'
fi
if [ -n "$video" ]; then
  for model in baseline student; do
    log="$run/video_${model}.log"
    if [ ! -f "$video" ]; then printf 'video_%s,skipped,missing_input\n' "$model" >> "$run/status.csv"; continue; fi
    if [ ! -f scripts/infer_video.py ]; then printf 'video_%s,skipped,missing_script\n' "$model" >> "$run/status.csv"; continue; fi
    echo "Creating $model PyTorch video..."
    if python3 scripts/infer_video.py "$video" --model "$model" --device cuda --side-by-side --output "$run/video_$model" > "$log" 2>&1; then
      printf 'video_%s,ok,%s\n' "$model" "$log" >> "$run/status.csv"
    else
      code=$?; printf 'video_%s,failed_%s,%s\n' "$model" "$code" "$log" >> "$run/status.csv"
      echo "Video $model failed; see $log. Ensure the updated video script and OpenCV are installed." >&2
    fi
  done
else
  printf 'video,skipped,no_input_provided\n' >> "$run/status.csv"
fi
cat "$run/status.csv"
printf '\nCopy this entire folder back: %s\n' "$run"
printf 'Core profiling/repeats failed: %s. Optional failures are shown in status.csv.\n' "$core_failed"
exit "$core_failed"
