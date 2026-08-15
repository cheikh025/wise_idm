# How the IDM training run works

Short version of what runs, in what order, and what each stage produces.
Full engineering detail is in `RUN_NOTES.md`.

## 0. Environment

```bash
cd /workspace/wise_idm && source env.sh
```

`env.sh` pins the three paths every stage uses:

| variable | path | why |
|---|---|---|
| `WISE_IDM_CACHE_DIR` | `/var/tmp/wise_idm/cache` | 1.75 TB frame cache — fast overlay disk, ephemeral, regenerable |
| `WISE_IDM_VIDEO_DIR` | `/var/tmp/wise_idm/mp4` | transient mp4 shards, deleted right after decode |
| `HF_HOME` | `/workspace/.hf_home` | pinned dataset `meta/` + `data/` parquets |

Checkpoints and logs go to `/workspace/...`, the **persistent** volume. Nothing
that must survive machine termination is written to `/var/tmp`.

## 1. Data preparation (one-off, done)

```
tools/fetch_droid_raw_metadata.py   ->  manifests/droid_raw_identity.parquet
tools/recover_missing_raw_identity.py -> manifests/droid_raw_identity_complete.parquet
build_selection_catalog.py          ->  manifests/catalog.parquet          (71,907 episodes)
select_manifests.py --seed 0        ->  manifests/train_21k.csv, val_1k.csv, selection_audit.json
preprocess_videos.py --workers 144  ->  1.75 TB of 128x224 uint8 frame cache
```

Result, matching `research/IDM_DESIGN.md` exactly:

- 21,000 train episodes / 1,000 validation episodes, per-lab and per-outcome
  counts equal to the frozen table, **zero scene overlap**;
- **391,674 train windows** (stride 16) and **10,308 validation windows**
  (stride 32), all end-aligned.

## 2. What one training sample is

One *window* = 33 consecutive frames from each of 3 cameras (wrist,
exterior 1, exterior 2) at 15 Hz, and the 32 actions aligned to the 32
frame-to-frame transitions.

```
frames    f0  f1  f2  ...  f31  f32          33 frames x 3 cameras
           \  /\  /          \  /
actions     a0  a1    ...     a31            32 x (7 joints + 1 gripper)
```

Action row `t` is the command across the visual transition `f[t] -> f[t+1]`.
Joints are absolute commanded positions; the gripper is the source value
thresholded at `> 0.5` (0 = open, 1 = closed).

The model sees **only pixels** — no proprioception, language, task, outcome, or
lab identity.

## 3. What one training step does

```
uint8 (B,33,H,W,3) x3 cameras          <- dataloader worker, straight off the cache
        |  .to(cuda) then permute/÷255 on the GPU
adjacent frames concatenated -> 6 channels
        |  B x 3 cameras x 32 transitions = 2,304 images at batch 24
shared ResNet-50 through layer3        -> 1024 x 8 x 14 per image
spatial softmax (all 1024 channels)    -> 2,048 coords -> one 512-d token
+ camera embedding, 2-layer cross-view transformer -> 1 fused token per transition
+ time embedding, 6-layer bidirectional temporal transformer over the 32 tokens
Linear(512,7) and Linear(512,1) at every transition
```

Loss = `SmoothL1` on the 7 joints standardised by **train-window** mean/std,
plus class-weighted `BCEWithLogits` on the binary gripper. 34.85 M parameters.

## 4. The runs

### a. Smoke test — minutes

```bash
./run_smoke.sh            # then, to prove restartability:
./run_smoke.sh --resume
```

500 train / 100 validation episodes (a scene-disjoint slice of the frozen
manifests), 60 batches per epoch, 3 epochs. Identical code path, distributed
layout, precision, batch size and optimiser as production. It exists to prove
DDP + `torch.compile` run together, that validation and checkpointing work, and
that a killed run resumes — not to produce a model.

### b. Learning-rate probe — ~20 min

```bash
./run_lr_probe.sh
```

Four peak LRs (1e-4, 2e-4, 3e-4, 5e-4), one per GPU, same **effective batch of
96** as production (24 x 4 accumulation on one GPU instead of 24 x 4 GPUs), ~450
optimiser steps each on ~3,000 episodes. Read it as a stability and direction
check — an LR that leads early can still lose over a full 20-epoch schedule.

### c. Production run — ~10 h

```bash
./run_production.sh              # ./run_production.sh --resume to continue
```

| | |
|---|---|
| GPUs | 4x A100-80GB, DDP over NVLink, one process per GPU |
| batch | 24 per GPU, no accumulation -> **effective batch 96 windows** (9,216 image-pairs/step) |
| precision | bf16 autocast, `channels_last`, `torch.compile`; validation in fp32 |
| optimiser | AdamW, weight decay 1e-4, gradient clip 1.0 |
| schedule | OneCycle, `pct_start=0.05`, peak LR 1e-4, 20 epochs |
| steps | ~4,079 optimiser steps/epoch, ~81.6k total |
| memory | ~51 GiB of 80 GiB per GPU (`expandable_segments` allocator) |
| throughput | ~215 windows/s across 4 GPUs -> ~30 min/epoch |

Each epoch: train over all 391,674 windows -> validate on all 10,308 validation
windows (sharded across the 4 ranks, summed with one all-reduce) -> rank 0
writes `last.pt`, plus `best.pt` whenever `val_mean_joint_mae` improves, plus
`history.json`.

## 5. Where the output lands

```
/workspace/wise_idm/checkpoints_production/best.pt      <- the deliverable
/workspace/wise_idm/checkpoints_production/last.pt      <- resume point
/workspace/wise_idm/checkpoints_production/history.json <- per-epoch metrics
/workspace/wise_idm/tb_logs_production/                 <- TensorBoard
/workspace/wise_idm/logs/production_latest.log          <- stdout
```

Every checkpoint carries the architecture, preprocessing version, panel layout,
pinned dataset revision, manifest SHA-256s, selection and window audits, joint
normalisation statistics, optimiser/scheduler/scaler state and per-rank RNG
state.

## 6. Watching it

```bash
tail -f /workspace/wise_idm/logs/production_latest.log
nvidia-smi dmon -s um            # utilisation + memory
```

Per-epoch lines look like:

```
epoch   0 train_loss=... val_mean_joint_mae=... val_gripper_acc=... (1800.0s, 215.0 win/s)
```

Progress lines every 500 batches report windows/s and peak GPU memory. If
`win/s` sags well below ~215 the dataloader is starving — raise
`--num-workers`; if memory climbs near 80 GiB, lower `--batch-size`.

## 7. Afterwards

```bash
.venv/bin/python verify_checkpoint.py \
  --checkpoint checkpoints_production/best.pt \
  --val-manifest manifests/val_1k.csv --batch-size 24
```

Reloads the checkpoint, re-checks the manifest and selection fingerprints, and
must reproduce the recorded validation metrics to within 1e-4. Then
`infer_on_dream.py` on a saved Cosmos dream confirms the fixed
`33 x 528 x 640 x 3` panel path, which is how the checkpoint becomes WISE's
`r_cons`.
