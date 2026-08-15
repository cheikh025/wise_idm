#!/usr/bin/env bash
set -euo pipefail
cd /workspace/wise_idm
source env.sh

echo "=== waiting for wrist preprocessing to finish ==="
wait 506645 2>/dev/null || true
while pgrep -f "preprocess_videos.py.*wrist_image_left" > /dev/null; do sleep 15; done

echo "=== wrist done, starting exterior_image_1_left ==="
.venv/bin/python preprocess_videos.py \
  --manifest manifests/train_21k.csv --manifest manifests/val_1k.csv \
  --camera exterior_image_1_left --input-height 96 --input-width 160 --resize-mode stretch \
  --workers 144

echo "=== exterior 1 done, starting exterior_image_2_left ==="
.venv/bin/python preprocess_videos.py \
  --manifest manifests/train_21k.csv --manifest manifests/val_1k.csv \
  --camera exterior_image_2_left --input-height 96 --input-width 160 --resize-mode stretch \
  --workers 144

echo "=== ALL THREE CAMERAS DONE ==="
