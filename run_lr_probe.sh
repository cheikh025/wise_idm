#!/usr/bin/env bash
# Short learning-rate probe: four peak LRs, one per GPU, in parallel.
#
# Faithful to the production run in everything that matters for LR transfer:
# same architecture, same manifests, same preprocessing, same optimizer/clip,
# and the same *effective* batch of 96 windows (24 per step x 4 accumulation
# on a single GPU, instead of 24 x 4 GPUs).
#
# Each run is a complete short OneCycle over ~450 optimizer steps (~43k windows,
# ~11% of an epoch), scored on a fixed 960-window slice of the frozen 1K
# validation manifest. Read it as a stability + direction check, not as proof
# about epoch 20: an LR that leads early can still lose over a full schedule.
#
# Every checkpoint written here records limit_train_batches != 0 in its config,
# so none of them can be mistaken for, or resumed into, the production run.
set -euo pipefail
cd "$(dirname "$0")"
source env.sh

LRS=(1e-4 2e-4 3e-4 5e-4)
ROOT=/workspace/wise_idm/lr_probe
TRAIN_MANIFEST=${TRAIN_MANIFEST:-manifests/probe/train_probe.csv}
VAL_MANIFEST=${VAL_MANIFEST:-manifests/probe/val_probe.csv}
WORKERS=${WORKERS:-6}
mkdir -p "$ROOT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

for i in "${!LRS[@]}"; do
  LR=${LRS[$i]}
  TAG=lr${LR}
  CUDA_VISIBLE_DEVICES=$i setsid nohup .venv/bin/python train.py \
    --mode train \
    --train-manifest "$TRAIN_MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --nonproduction-manifests \
    --arch wise \
    --batch-size 24 \
    --gradient-accumulation 4 \
    --epochs 3 \
    --limit-train-batches 600 \
    --limit-val-batches 40 \
    --lr "$LR" \
    --amp-dtype bf16 \
    --compile \
    --num-workers "$WORKERS" \
    --prefetch-factor 3 \
    --log-interval 25 \
    --seed 0 \
    --out-dir "$ROOT/$TAG" \
    --log-dir "$ROOT/tb_$TAG" \
    > "$ROOT/$TAG.log" 2>&1 < /dev/null &
  echo "GPU $i -> lr=$LR  log=$ROOT/$TAG.log"
done
wait
echo "all LR probes finished"
