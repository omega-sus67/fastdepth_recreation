#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/nyu_depth_v2/raw
for shard in val-000000.tar val-000001.tar; do
    curl -fL --retry 3 --continue-at - -o "data/nyu_depth_v2/raw/$shard" \
      "https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/main/data/$shard"
done
printf '\nDownloaded archives. Inspect their contents and split before evaluation.\n'
