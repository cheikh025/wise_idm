#!/usr/bin/env bash
# Pre-flight for the production run: identical code path, distributed layout,
# precision, batch size and optimiser, but a 500-episode slice of the frozen
# train manifest (and a 100-episode, scene-disjoint slice of the frozen
# validation manifest) so it finishes in minutes.
#
# Verifies DDP + torch.compile together, per-epoch validation, checkpoint save,
# and (via `--resume`) that a killed run picks up exactly where it left off.
# Also gives real dataloader throughput off the real cache.
#
#   ./run_smoke.sh            fresh
#   ./run_smoke.sh --resume   continue from checkpoints_smoke/last.pt
set -euo pipefail
cd "$(dirname "$0")"
source env.sh

COMPILE_MODE=${COMPILE_MODE:-default}
SUFFIX=${SUFFIX:-}
OUT_DIR=/workspace/wise_idm/checkpoints_smoke${SUFFIX}
LOG_DIR=/workspace/wise_idm/tb_logs_smoke${SUFFIX}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)

RESUME=()
[[ "${1:-}" == "--resume" ]] && RESUME=(--resume "$OUT_DIR/last.pt")

.venv/bin/python -m torch.distributed.run --standalone --nproc_per_node="$GPUS" train.py \
  --mode train \
  --train-manifest manifests/smoke/train_smoke.csv \
  --val-manifest manifests/smoke/val_smoke.csv \
  --nonproduction-manifests \
  --arch wise \
  --batch-size 24 \
  --gradient-accumulation 1 \
  --epochs 3 \
  --limit-train-batches 60 \
  --limit-val-batches 20 \
  --lr 1e-4 \
  --amp-dtype bf16 \
  --compile \
  --compile-mode "$COMPILE_MODE" \
  --num-workers 10 \
  --prefetch-factor 3 \
  --log-interval 5 \
  --seed 0 \
  --out-dir "$OUT_DIR" \
  --log-dir "$LOG_DIR" \
  "${RESUME[@]}"
