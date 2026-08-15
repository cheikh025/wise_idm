#!/usr/bin/env bash
# Production WISE-IDM training run: 4x A100-80GB, DDP, bf16, channels_last,
# torch.compile. Checkpoints and logs land on the persistent volume.
#
# batch 24 x 4 GPUs = effective batch 96, peak LR 3e-4. That is exactly the
# configuration `run_lr_probe.sh` measured: 3e-4 gave the lowest training loss
# of {1e-4, 2e-4, 3e-4, 5e-4} at effective batch 96, and 1e-4 was worst on both
# training loss and validation MAE at every probe epoch.
#
#   ./run_production.sh              start (or restart from scratch)
#   ./run_production.sh --resume     continue from checkpoints_production/last.pt
set -euo pipefail
cd "$(dirname "$0")"
source env.sh

OUT_DIR=/workspace/wise_idm/checkpoints_production
LOG_DIR=/workspace/wise_idm/tb_logs_production
RUN_LOG=/workspace/wise_idm/logs/production_$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p "$OUT_DIR" "$LOG_DIR" /workspace/wise_idm/logs

RESUME=()
if [[ "${1:-}" == "--resume" ]]; then
  RESUME=(--resume "$OUT_DIR/last.pt")
fi

GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "launching on $GPUS GPU(s); log -> $RUN_LOG"

# expandable_segments keeps the reserved-vs-allocated gap small, so batch 24
# (~51 GiB allocated) stays comfortably inside 80 GiB.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

setsid nohup .venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node="$GPUS" train.py \
  --mode train \
  --train-manifest manifests/train_21k.csv \
  --val-manifest manifests/val_1k.csv \
  --arch wise \
  --batch-size 24 \
  --gradient-accumulation 1 \
  --epochs "${EPOCHS:-12}" \
  --lr "${LR:-3e-4}" \
  --amp-dtype bf16 \
  --compile \
  --compile-mode "${COMPILE_MODE:-default}" \
  --num-workers 10 \
  --prefetch-factor 3 \
  --val-batch-size 24 \
  --log-interval 25 \
  --seed 0 \
  --out-dir "$OUT_DIR" \
  --log-dir "$LOG_DIR" \
  "${RESUME[@]}" \
  > "$RUN_LOG" 2>&1 < /dev/null &

echo $! > /workspace/wise_idm/logs/production.pid
echo "pid $(cat /workspace/wise_idm/logs/production.pid)"
ln -sfn "$RUN_LOG" /workspace/wise_idm/logs/production_latest.log
