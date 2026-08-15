#!/usr/bin/env bash
# Production WISE-IDM training run: composite single-panel architecture,
# 4x A100-80GB, DDP, bf16, channels_last, torch.compile.
#
# batch 16 x 4 GPUs (no accumulation) = effective batch 64. Measured directly
# on this architecture (not assumed): batch 16 gave 57.0 win/s/GPU at 34.2 GiB,
# batch 24 gave only 50.1 win/s/GPU at 51.2 GiB - 16 is both faster and safer
# here, unlike the frozen per-view architecture where throughput was flat
# across batch size. See RUN_NOTES.md.
#
# peak LR 3e-4 carries over from the LR probe run on the frozen architecture at
# effective batch 96 (this run's effective batch is 64, a bit lower, so 3e-4
# remains a reasonable-to-conservative choice, not miscalibrated upward).
#
#   ./run_production_composite.sh              start (or restart from scratch)
#   ./run_production_composite.sh --resume     continue from checkpoints_production_composite/last.pt
set -euo pipefail
cd "$(dirname "$0")"
source env.sh

OUT_DIR=/workspace/wise_idm/checkpoints_production_composite
LOG_DIR=/workspace/wise_idm/tb_logs_production_composite
RUN_LOG=/workspace/wise_idm/logs/production_composite_$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p "$OUT_DIR" "$LOG_DIR" /workspace/wise_idm/logs

RESUME=()
if [[ "${1:-}" == "--resume" ]]; then
  RESUME=(--resume "$OUT_DIR/last.pt")
fi

GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "launching on $GPUS GPU(s); log -> $RUN_LOG"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

setsid nohup .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node="$GPUS" train.py \
  --mode train \
  --train-manifest manifests/train_21k.csv \
  --val-manifest manifests/val_1k.csv \
  --arch composite \
  --batch-size 16 \
  --gradient-accumulation 1 \
  --epochs "${EPOCHS:-12}" \
  --lr "${LR:-3e-4}" \
  --amp-dtype bf16 \
  --compile \
  --compile-mode "${COMPILE_MODE:-default}" \
  --num-workers 10 \
  --prefetch-factor 3 \
  --val-batch-size 16 \
  --log-interval 25 \
  --seed 0 \
  --out-dir "$OUT_DIR" \
  --log-dir "$LOG_DIR" \
  "${RESUME[@]}" \
  > "$RUN_LOG" 2>&1 < /dev/null &

echo $! > /workspace/wise_idm/logs/production_composite.pid
echo "pid $(cat /workspace/wise_idm/logs/production_composite.pid)"
ln -sfn "$RUN_LOG" /workspace/wise_idm/logs/production_composite_latest.log
