# Storage layout for the production IDM run on this machine.
#
#   /workspace  -> 1.0 TB persistent host volume (survives recycle/destroy).
#                  Code, manifests, checkpoints, TensorBoard logs, run logs.
#   /var/tmp    -> 4.1 TB container overlay on the same NVMe RAID0 array
#                  (~3.5 GB/s write). Fast but ephemeral: the frame cache and
#                  the transient mp4 downloads live here because they are
#                  deterministically regenerable from the manifests.
export WISE_IDM_ROOT=/workspace/wise_idm
export WISE_IDM_CACHE_DIR=/var/tmp/wise_idm/cache
export WISE_IDM_VIDEO_DIR=/var/tmp/wise_idm/mp4
export HF_HOME=/workspace/.hf_home
export COSMOS3_DROID_ROOT=/workspace/.hf_home/hub/datasets--nvidia--Cosmos3-DROID/snapshots/5c11a20accb11497270a5247a7f1e66ad04c956c
export HF_XET_HIGH_PERFORMANCE=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$WISE_IDM_CACHE_DIR" "$WISE_IDM_VIDEO_DIR"
