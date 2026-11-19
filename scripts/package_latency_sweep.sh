#!/usr/bin/env bash
# Build a local, dataset-containing transfer archive. Do not publish it to GitHub.
set -euo pipefail
if [ "$#" -ne 1 ]; then echo 'Usage: bash scripts/package_latency_sweep.sh NEW_ARCHIVE.tar' >&2; exit 2; fi
archive="$1"
if [ -e "$archive" ]; then echo 'Archive already exists.' >&2; exit 1; fi
files=(README.md agents.md src scripts third_party/fast-depth configs
  docs/latency_sweep_jetson.md docs/jetson_first_test.md
  checkpoints/converted/community-skipadd-pruned.pth
  results/training/width_192/checkpoint.pth
  results/training/width_176/checkpoint.pth
  results/training/width_160/checkpoint.pth
  exports/simple-baseline/model.onnx exports/simple-student/model.onnx
  exports/sweep/width176-fp32/model.onnx exports/sweep/width160-fp32/model.onnx
  exports/sweep/baseline-int8-floatbias/model.onnx exports/sweep/width192-int8-floatbias/model.onnx
  exports/sweep/width176-int8-floatbias/model.onnx exports/sweep/width160-int8-floatbias/model.onnx
  exports/sweep/baseline-int8-floatbias/quantization.json exports/sweep/width192-int8-floatbias/quantization.json
  exports/sweep/width176-int8-floatbias/quantization.json exports/sweep/width160-int8-floatbias/quantization.json
  data/nyu_depth_v2/int8-calibration-2026-11-16/manifest.json
  data/nyu_depth_v2/jetson-tuning-445 data/nyu_depth_v2/jetson-evaluation-654
  data/nyu_depth_v2/latency-tuning-references data/nyu_depth_v2/latency-evaluation-references)
for item in "${files[@]}"; do if [ ! -e "$item" ]; then echo "Missing: $item" >&2; exit 1; fi; done
tar --exclude='__pycache__' --exclude='*.pyc' -cf "$archive" "${files[@]}"
(cd "$(dirname "$archive")" && sha256sum "$(basename "$archive")") > "$archive.sha256"
echo "Created $archive and $archive.sha256; contains local dataset samples."
