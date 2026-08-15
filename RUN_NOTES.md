# Production IDM run — implementation notes

Machine: Vast.ai container, **4x A100-SXM4-80GB** (NVLink NV12 all-to-all), 256 vCPU
(AMD EPYC 7763), 1.0 TiB RAM, driver 595.71.05, CUDA 13.0, torch 2.12.0+cu130.

## Storage layout

| Path | Backing | Size | Contents |
|---|---|---|---|
| `/workspace` | host volume, **persistent** through recycle/destroy (`workspace_is_volume=true`) | 1.0 TB (14 GB used) | repos, manifests, checkpoints, TensorBoard logs, run logs |
| `/var/tmp/wise_idm` | container overlay on the same NVMe RAID0 array, **ephemeral** | 4.1 TB | 1.75 TB frame cache + transient mp4 downloads |
| `/workspace/.hf_home` | persistent | ~6 GB | pinned Cosmos3-DROID `meta/` + `data/` parquets |

Measured: 3.5 GB/s write, 1.8 GB/s single-stream O_DIRECT read on both mounts.
The cache is deterministically regenerable from the manifests, so it is the
right thing to keep off the persistent volume (which is too small for it
anyway); checkpoints and logs are the things that must survive.

`source env.sh` sets `WISE_IDM_CACHE_DIR`, `WISE_IDM_VIDEO_DIR`, `HF_HOME`, and
`COSMOS3_DROID_ROOT` consistently for every stage.

## Data provenance

`nvidia/Cosmos3-DROID@5c11a20accb11497270a5247a7f1e66ad04c956c`, 758 GB / 71,907
episodes published. The frozen eligibility rule (`length >= 33`) reproduces the
contract exactly: **57,584 success + 13,669 failure = 71,253 eligible**.

Scene identity is not in the Cosmos metadata, so it is joined from raw DROID
1.0.1 on the public `gs://gresearch/robotics/droid_raw/1.0.1` mirror
(`tools/fetch_droid_raw_metadata.py`, 74,896 `metadata_*.json` files).

Three real defects in that source had to be handled to build the catalog at all:

1. **265 duplicate metadata files.** 264 episode directories ship the same
   metadata twice under different lab-name capitalisation (`TRI+...` vs
   `tri+...`); one (`TRI/success/2023-10-12/Thu_Oct_12_13:25:29_2023`) holds two
   genuinely different episodes. Resolved by preferring the row whose `lab`
   matches the official `episode_id` path component, then asserting that any
   remaining duplicate agrees on `building`/`scene_id`/`success`. The ambiguous
   one resolves to the `7dfa2da3` recording, whose raw length (243) matches the
   official length (242) — DROID raw counts one more frame than the Cosmos
   conversion for 70,113 of 71,876 joined episodes.
2. **31 pinned episodes have no `metadata_*.json` at all.** Their
   `trajectory.h5` root attributes still carry the authoritative
   `building`/`scene_id`/`robot_serial_number`, so
   `tools/recover_missing_raw_identity.py` reads those (1-3 MB per file) and
   marks the rows `scene_identity_source="trajectory_h5_attrs"`. Without this
   the catalog build fails outright.
3. **12 episodes whose raw `success` flag contradicts their own DROID
   directory.** The official root is authoritative — the frozen lab x outcome
   quotas are defined on it and its eligible counts reproduce exactly — so
   `build_selection_catalog.py` now records `raw_success_matches_root` as an
   audit column and only aborts if the disagreement rate exceeds 1% (it is
   0.017%).

## Selected manifests

`manifests/{catalog.parquet,train_21k.csv,val_1k.csv,selection_audit.json}`,
`select_manifests.py --seed 0`. Reproduces `research/IDM_DESIGN.md` exactly:

- train 21,000 episodes (16,971 success / 4,029 failure), per-lab counts equal
  to the frozen table (AUTOLab 3032, CLVR 1495, GuptaLab 375, ILIAD 945,
  IPRL 1666, IRIS 984, PennPAL 849, RAD 430, RAIL 2328, REAL 1464, RPL 731,
  TRI 5825, WEIRD 876);
- validation 1,000 episodes (808 / 192), same frozen per-lab table;
- `scene_overlap_count = 0` over 1,245 train scenes and 109 validation scenes;
- **391,674 train windows** (stride 16) and **10,308 validation windows**
  (stride 32), both end-aligned.

## Fixes made before training

| File | Problem | Change |
|---|---|---|
| `build_selection_catalog.py` | `drop(columns=("raw_lab","raw_success"))` passes a tuple, which pandas reads as one label — `KeyError` on every run. Never exercised before. | pass a list |
| `build_selection_catalog.py` | hard `ValueError` on the 12 raw/official `success` disagreements | record `raw_success_matches_root`, warn, abort only above a 1% rate |
| `selection.py` | manifest validation rejected duplicate raw `uuid`s, but DROID uuids omit the outcome, so 41 uuids legitimately appear once per root for different episodes | check `episode_id` uniqueness and **split-qualified** uuid uniqueness |
| `preprocess_videos.py` | serial, one shard at a time | 144-worker process pool, per-shard download → decode → delete, resumable, with retries |
| `preprocess_videos.py` | decoded and stored every frame up to each shard's last selected episode: 34.2M frames / 2.94 TB | store only frames a selected episode owns: 20.3M frames / **1.75 TB**; unowned frames stay holes in a sparse file so in-file indexing is unchanged |
| `preprocess_videos.py` | ffmpeg sizes its thread pool from the CPU count (256); dozens of concurrent workers exhausted the thread budget and ffmpeg silently produced 0 frames (43 shards failed) | `-threads 2`, plus ffmpeg stderr is now surfaced on short reads |
| `droid_dataset.py` | sparse caches could silently return black frames for an unwritten range | every cache gets a `.ranges.json` sidecar; `read_frames` raises if the requested span was never decoded |
| `droid_dataset.py` | `__getitem__` returned float32 `(T,3,H,W)` per camera — 34 MB/window over PCIe, with the permute/÷255 done on a dataloader CPU core | returns uint8 `(T,H,W,3)` (8.5 MB/window); `train.prepare_view` does the identical arithmetic on the GPU |
| `droid_dataset.py` | frames read one at a time (`np.stack([array[i] for i in ids])`) and shard locators looked up through pandas Series in the hot path | single contiguous slice copy; plain-Python `video_locator` |
| `train.py` | `loss.item()` on every micro-batch forced a host sync and stalled the input pipeline | losses accumulate on-GPU; scalars logged every `--log-interval` batches |
| `train.py` | `normalize_joints` rebuilt the joint mean/std from Python lists on **every micro-batch**. `torch.as_tensor(list, device=cuda)` is a *pageable* host-to-device copy, which blocks the host until everything already queued on the stream has completed — destroying CPU run-ahead and preventing the next batch's transfer from overlapping compute | `stats_tensors()` builds them once and caches per (stats, device, dtype) |
| `train.py` | no `channels_last`, no `cudnn.benchmark` | both, shared with `verify_checkpoint.py` via `model_factory.configure_backends()` so a reloaded metric still reproduces. TF32 flags deliberately left at PyTorch defaults |
| `train.py` | no `torch.compile` | opt-in `--compile`; checkpoints always save the unwrapped module, so state-dict keys never gain a `_orig_mod.`/`module.` prefix |
| `train.py` | ragged final batch reshaped every epoch | `drop_last=True` on the train loader (uniform shapes for compile/cudnn; the dropped tail is re-shuffled each epoch) |
| `train.py` | no way to smoke-test the production path without a full epoch | `--limit-train-batches` / `--limit-val-batches`, recorded in the checkpoint config so a limited run can never be mistaken for, or resumed into, the production run |

Not changed: architecture, loss, normalisation, window/label alignment, stride,
camera order, letterbox geometry, selection algorithm, checkpoint contract.

## Correctness verification

`tools/verify_window_alignment.py` re-derives random windows from the pinned
source — decoding the exact frames out of the mp4 with ffmpeg and re-reading the
action rows from the source parquet — and compares against what the dataset
hands the model. Result: **pixels bit-identical to a fresh decode** for all
three cameras, and **actions exactly rows `s..s+31` of the source parquet**.

Independently confirmed on the source: every shard is 640x360 at exactly 15/1
fps and its episodes tile it contiguously from frame 0
(`nb_read_frames == sum(length) == max(to_timestamp*15)`), so
`offset = round(from_timestamp*15)` is exactly the episode's first in-file frame.

`dist.gather_object` (used once per epoch to collect per-rank RNG state) was
verified to work on this NCCL build at world size 4 — it is unsupported on some
older builds and would have crashed at the first epoch boundary.

## Throughput and batch sizing (measured, single A100, real uint8 input path)

| config | win/s/GPU | peak alloc | reserved |
|---|---|---|---|
| eager, NCHW | 26.6 | 2.44 GiB/win | |
| eager, channels_last | 40.3 | 2.44 GiB/win | |
| **compiled + channels_last** | **~54** | 2.10 GiB/win | |
| batch 24, default allocator | 47.5 | 50.28 GiB | 66.53 GiB |
| **batch 24, `expandable_segments:True`** | **53.8** | 50.28 GiB | **50.77 GiB** |

channels_last is 1.51x on its own and `torch.compile` a further ~1.35x — 2.0x
combined. Throughput is flat in batch size from 8 upward (the backbone already
sees `batch x 3 cameras x 32 transitions` = 2,304 images per step at batch 24),
so batch size is an optimisation choice, not a throughput one. An earlier
reading that batch 16 beat batch 24 was a within-process run-order artefact: the
first configuration measured in a process is always ~13% faster, and the
ordering reverses when the order does.

`expandable_segments:True` cuts reserved memory at batch 24 from 66.5 GiB to
50.8 GiB, which is what makes batch 24 comfortably safe inside 80 GiB.

Projected: ~215 win/s across 4 GPUs -> **~30 min/epoch** over 391,674 windows,
~10 h for 20 epochs plus per-epoch validation.

### Kernel profile

`torch.profiler` on a compiled step: `aten::convolution_backward` 12.5%,
`aten::cudnn_convolution` 7.5%, then a long tail of already-fused Triton
BatchNorm/ReLU/max-pool kernels at ~1% each. No single non-convolution
hotspot; the spatial softmax does not appear in the top 22, so inductor fuses
it into the surrounding epilogue. This is a textbook conv+BN bandwidth-bound
ResNet-50 profile — there is no structural 2x available.

Cross-check: 54 win/s x 96 images = 5,184 img/s at 128x224, i.e. ~2,960
224x224-equivalent img/s of a *truncated* ResNet-50, against ~2,800 img/s for
NVIDIA's tuned **full** ResNet-50 AMP training on the same GPU. The pipeline is
already at reference-implementation efficiency.

## Learning-rate probe

`run_lr_probe.sh`: four peak LRs, one per GPU, same architecture / manifests /
preprocessing / optimiser / gradient clipping as production and the same
**effective batch of 96** (24 x 4 accumulation on one GPU). ~450 optimiser steps
each — a complete short OneCycle — on 2,999 train and 192 validation episodes
drawn from the frozen manifests (scene-disjoint, all 13 labs). All four start
from an identical loss of 1.3274, confirming identical initialisation.

| peak LR | e0 val MAE | e1 train loss | e1 val MAE | e1 gripper acc |
|---|---|---|---|---|
| 1e-4 | 0.32940 | 0.23377 | 0.31241 | 0.947 |
| 2e-4 | 0.29660 | 0.20570 | 0.26997 | 0.951 |
| **3e-4** | 0.31075 | **0.19836** | 0.26394 | 0.956 |
| 5e-4 | 0.29490 | 0.22580 | **0.24014** | **0.962** |

**1e-4 is worst on both metrics at every epoch** — the runbook's nominal default
is simply too low for effective batch 96. 3e-4 reaches the lowest training loss
(best optimisation); 5e-4 reaches the lowest validation MAE while its training
loss *rises* relative to 3e-4, the usual signature of a peak LR past the
optimisation optimum trading fit for regularisation. Since optimal peak LR
tends to fall as the schedule lengthens, and production runs ~100x more
optimiser steps than the probe, **3e-4** is the choice.

Caveat worth keeping: this is a 3-epoch probe. It is a stability and direction
check, not proof about epoch 12.

### Why 128x224 and not 224x224

Source frames are 640x360 (16:9). Letterboxed into 224x128 they occupy
224x126 — **98.4% of the canvas**. Letterboxed into 224x224 they still occupy
224x126, now inside a square canvas — **56% of the canvas**, i.e. 1.78x the
compute convolving over black padding for zero extra information. 128x224 is
the efficient choice, not a reduced one. Letterboxing is not cropping: the
entire field of view is retained.

`tools/render_input_comparison.py` writes a side-by-side of the source frame,
the model input nearest-upscaled back to 640x360, and their absolute
difference. The measurable loss is concentrated in high-frequency background
texture (lab calibration walls); the arm pose, gripper fingers, table edge and
manipulated objects all remain clearly resolved.

### Train vs inference geometry (asymmetric, worth remembering for r_cons)

`tools/render_cosmos_geometry.py`. In DROID all three cameras are natively
640x360. In the decoded Cosmos panel only the wrist is 640x360; both exteriors
arrive as 320x168.

| view | train content | inference content |
|---|---|---|
| wrist | 224x126 | 224x126 (identical path) |
| exterior 1 / 2 | 224x126 | **224x118** |

So the wrist is the high-fidelity signal at inference and the exteriors are the
degraded ones: half the linear resolution, plus a ~6% vertical scale mismatch
(118/126 = 0.937) caused by the 320x168 tile's 1.905 aspect ratio.

That mismatch is **not** fixed by raising the model input resolution - at
320x180 it would be 168/180 = 0.933, the same. Raising resolution buys
sharpness, not alignment. For reference, 320x180 would be the "natural" larger
size (DROID downsamples exactly 2x with zero padding, and the Cosmos exterior
tile would be used pixel-for-pixel), but it costs 2.01x compute, halves the
per-GPU batch, and needs a 3.5 TB cache and a full re-preprocess. Not adopted.
