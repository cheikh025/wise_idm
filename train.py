"""Train the three-view, vision-only WISE inverse dynamics model on DROID."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from droid_dataset import (
    CHUNK_LEN,
    HF_REPO,
    HF_REVISION,
    DroidIDMDataset,
    assert_scene_disjoint,
    load_episode_manifest,
)
from droid_panel_dataset import (
    PANEL_LAYOUT_VERSION as COMPOSITE_PANEL_LAYOUT_VERSION,
)
from droid_panel_dataset import (
    PANEL_TARGET_HEIGHT,
    PANEL_TARGET_WIDTH,
    DroidIDMPanelDataset,
)
from model_composite import ARCH_ID as COMPOSITE_ARCH_ID
from model_factory import build_model, canonical_arch, configure_backends, forward_model
from model_wise import ARCH_ID, BACKBONE_WEIGHTS, CAMERA_ORDER
from selection import (
    TRAIN_EPISODES,
    TRAIN_LAB_OUTCOME_QUOTAS,
    VAL_EPISODES,
    VAL_LAB_OUTCOME_QUOTAS,
    validate_production_manifest,
    window_audit,
)
from vision import (
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    PANEL_LAYOUT_VERSION,
    VISION_PREPROCESS_VERSION,
)


def batch_view_keys(arch: str) -> list[str]:
    """Which batch dict keys `call_model` reads, for the given architecture."""
    return ["panel"] if canonical_arch(arch) == COMPOSITE_ARCH_ID else list(CAMERA_ORDER)


def compute_stats_fast(dataset: DroidIDMDataset) -> tuple[dict, float, int, int]:
    """Compute train-window action statistics without decoding any video."""
    joint_sum = np.zeros(7, dtype=np.float64)
    joint_square_sum = np.zeros(7, dtype=np.float64)
    joint_count = 0
    positive = 0
    negative = 0
    for window in dataset.windows:
        joints, gripper = dataset.episode_actions[window.episode_key]
        action_slice = slice(window.chunk_start, window.chunk_start + dataset.chunk_len)
        window_joints = joints[action_slice].astype(np.float64)
        window_gripper = gripper[action_slice] > 0.5
        joint_sum += window_joints.sum(axis=0)
        joint_square_sum += np.square(window_joints).sum(axis=0)
        joint_count += len(window_joints)
        positive += int(window_gripper.sum())
        negative += int((~window_gripper).sum())
    if joint_count == 0:
        raise ValueError("training selection produced no valid 33-frame windows")

    mean = joint_sum / joint_count
    variance = np.maximum(joint_square_sum / joint_count - np.square(mean), 0.0)
    std = np.sqrt(variance) + 1e-6
    return (
        {"mean": mean.tolist(), "std": std.tolist()},
        negative / max(positive, 1),
        positive,
        negative,
    )


_STATS_TENSORS: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def stats_tensors(stats: dict, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Device-resident joint mean/std, built once per (stats, device, dtype).

    Rebuilding these from Python lists on every micro-batch costs a pageable
    host-to-device copy, and a pageable copy blocks the host until everything
    already queued on the stream has finished - which stalls CPU run-ahead and
    stops the next batch's transfer overlapping with compute.
    """
    key = (tuple(stats["mean"]), tuple(stats["std"]), str(device), dtype)
    cached = _STATS_TENSORS.get(key)
    if cached is None:
        cached = (
            torch.as_tensor(stats["mean"], device=device, dtype=dtype),
            torch.as_tensor(stats["std"], device=device, dtype=dtype),
        )
        _STATS_TENSORS[key] = cached
    return cached


def normalize_joints(joints: torch.Tensor, stats: dict, device) -> torch.Tensor:
    mean, std = stats_tensors(stats, device, joints.dtype)
    return (joints - mean) / std


def denormalize_joints(joints: torch.Tensor, stats: dict, device) -> torch.Tensor:
    mean, std = stats_tensors(stats, device, joints.dtype)
    return joints * std + mean


def loss_fn(
    joints_pred_norm: torch.Tensor,
    gripper_logit_pred: torch.Tensor,
    joints_target_norm: torch.Tensor,
    gripper_target: torch.Tensor,
    gripper_pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    joint_loss = F.smooth_l1_loss(joints_pred_norm, joints_target_norm)
    gripper_loss = F.binary_cross_entropy_with_logits(
        gripper_logit_pred.squeeze(-1),
        gripper_target,
        pos_weight=gripper_pos_weight,
    )
    return joint_loss + gripper_loss, joint_loss.detach(), gripper_loss.detach()


def prepare_view(frames: torch.Tensor, device) -> torch.Tensor:
    """Move one camera clip to ``device`` as float (B, T, 3, H, W) in [0, 1].

    The dataset hands over uint8 ``(B, T, H, W, 3)`` so that the host->device
    copy carries a quarter of the bytes; the permute and 1/255 scaling are the
    same arithmetic the CPU path used to do, just executed on the GPU. Float
    inputs (dream inference, tests) are passed through untouched.
    """
    tensor = frames.to(device, non_blocking=True)
    if tensor.dtype == torch.uint8:
        tensor = tensor.permute(0, 1, 4, 2, 3).float().div_(255.0)
    return tensor


def call_model(model, arch: str, batch: dict, device, cameras: list[str]):
    views = [prepare_view(batch[camera], device) for camera in cameras]
    return forward_model(model, arch, views)


def amp_context(device: torch.device, dtype: torch.dtype | None):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def evaluate(
    model,
    loader,
    stats,
    device,
    arch: str = ARCH_ID,
    cameras: list[str] = CAMERA_ORDER,
    amp_dtype: torch.dtype | None = None,
    distributed: bool = False,
    max_batches: int = 0,
) -> dict:
    model.eval()
    totals = torch.zeros(11, device=device, dtype=torch.float64)
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            target = batch["action"].to(device, non_blocking=True)
            joints_target, gripper_target = target[..., :7], target[..., 7]
            with amp_context(device, amp_dtype):
                output = call_model(model, arch, batch, device, list(cameras))
            joints_pred = denormalize_joints(output["joints"].float(), stats, device)
            gripper_probability = torch.sigmoid(output["gripper_logit"].float().squeeze(-1))

            error = (joints_pred - joints_target).abs().double()
            totals[:7] += error.sum(dim=(0, 1))
            totals[7] += target.shape[0] * target.shape[1]
            predicted_binary = gripper_probability > 0.5
            target_binary = gripper_target > 0.5
            totals[8] += (predicted_binary == target_binary).sum()
            totals[9] += target_binary.numel()
            totals[10] += (gripper_probability - gripper_target).abs().sum()
    if distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if totals[7].item() == 0 or totals[9].item() == 0:
        raise ValueError("validation selection produced no action steps")
    model.train()
    per_joint_mae = totals[:7] / totals[7]
    return {
        "per_joint_mae": per_joint_mae.tolist(),
        "mean_joint_mae": per_joint_mae.mean().item(),
        "gripper_mae": (totals[10] / totals[9]).item(),
        "gripper_accuracy": (totals[8] / totals[9]).item(),
    }


def build_datasets(args: argparse.Namespace):
    if bool(args.train_manifest) != bool(args.val_manifest):
        raise ValueError("--train-manifest and --val-manifest must be provided together")

    arch = canonical_arch(args.arch)
    common = dict(
        input_height=args.input_height,
        input_width=args.input_width,
        num_frames=CHUNK_LEN + 1,
        chunk_len=CHUNK_LEN,
        end_align_tail=True,
    )
    if args.train_manifest:
        train_refs = load_episode_manifest(args.train_manifest, require_scene_id=True)
        val_refs = load_episode_manifest(args.val_manifest, require_scene_id=True)
        assert_scene_disjoint(train_refs, val_refs)
        if args.mode == "train" and args.nonproduction_manifests:
            args.train_selection_audit = None
            args.val_selection_audit = None
        elif args.mode == "train":
            args.train_selection_audit = validate_production_manifest(
                train_refs,
                args.train_manifest,
                expected_count=TRAIN_EPISODES,
                lab_outcome_quotas=TRAIN_LAB_OUTCOME_QUOTAS,
            )
            args.val_selection_audit = validate_production_manifest(
                val_refs,
                args.val_manifest,
                expected_count=VAL_EPISODES,
                lab_outcome_quotas=VAL_LAB_OUTCOME_QUOTAS,
            )
        if arch == COMPOSITE_ARCH_ID:
            train_dataset = DroidIDMPanelDataset(episodes=train_refs, stride=args.train_stride)
            val_dataset = DroidIDMPanelDataset(episodes=val_refs, stride=args.val_stride)
        else:
            train_dataset = DroidIDMDataset(episodes=train_refs, stride=args.train_stride, **common)
            val_dataset = DroidIDMDataset(episodes=val_refs, stride=args.val_stride, **common)
    elif arch == COMPOSITE_ARCH_ID:
        raise ValueError(f"{COMPOSITE_ARCH_ID} requires --train-manifest and --val-manifest")
    else:
        train_dataset = DroidIDMDataset(
            episode_indices=args.train_episodes,
            dataset_split="success",
            stride=args.train_stride,
            **common,
        )
        val_dataset = DroidIDMDataset(
            episode_indices=args.val_episodes,
            dataset_split="success",
            stride=args.val_stride,
            **common,
        )
    if train_dataset.zero_window_episodes or val_dataset.zero_window_episodes:
        raise ValueError(
            "selected episodes shorter than 33 frames cannot be preserved: "
            f"train={train_dataset.zero_window_episodes[:20]}, "
            f"val={val_dataset.zero_window_episodes[:20]}"
        )
    return train_dataset, val_dataset


def episode_selection_digest(dataset: DroidIDMDataset) -> str:
    selection = [
        [
            ref.dataset_split,
            ref.episode_index,
            ref.scene_id,
            ref.lab,
            ref.episode_id,
            ref.length,
        ]
        for ref in dataset.episodes
    ]
    serialized = json.dumps(selection, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def manifest_file_digest(path: str | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_config(
    args: argparse.Namespace,
    train_dataset: DroidIDMDataset,
    val_dataset: DroidIDMDataset,
    world_size: int,
) -> dict:
    arch = canonical_arch(args.arch)
    config = {
        "config_schema_version": 2,
        "arch": arch,
        "input_height": args.input_height,
        "input_width": args.input_width,
        "num_frames": train_dataset.num_frames,
        "chunk_len": train_dataset.chunk_len,
        "action_horizon": train_dataset.chunk_len,
        "cameras": list(CAMERA_ORDER),
        "camera_order_droid": [
            "wrist_image_left",
            "exterior_image_1_left",
            "exterior_image_2_left",
        ],
        "panel_layout_version": (
            COMPOSITE_PANEL_LAYOUT_VERSION if arch == COMPOSITE_ARCH_ID else PANEL_LAYOUT_VERSION
        ),
        "vision_preprocess_version": VISION_PREPROCESS_VERSION,
        "dataset_repo": HF_REPO,
        "dataset_revision": HF_REVISION,
        "vision_only": True,
        "uses_proprioception": False,
        "uses_language": False,
        "train_stride": args.train_stride,
        "val_stride": args.val_stride,
        "end_align_tail": True,
        "train_episode_count": len(train_dataset.episodes),
        "val_episode_count": len(val_dataset.episodes),
        "train_selection_sha256": episode_selection_digest(train_dataset),
        "val_selection_sha256": episode_selection_digest(val_dataset),
        "train_manifest_sha256": manifest_file_digest(args.train_manifest),
        "val_manifest_sha256": manifest_file_digest(args.val_manifest),
        "joint_loss": "train-stat standardized SmoothL1",
        "gripper_loss": "weighted BCEWithLogits",
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "batch_size_per_rank": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "world_size": world_size,
        "effective_batch_size": args.batch_size * args.gradient_accumulation * world_size,
        "amp_dtype": args.amp_dtype,
        "train_drop_last": True,
        "memory_format": "channels_last",
        "limit_train_batches": args.limit_train_batches,
        "limit_val_batches": args.limit_val_batches,
        "nonproduction_manifests": args.nonproduction_manifests,
        "train_selection_audit": getattr(args, "train_selection_audit", None),
        "val_selection_audit": getattr(args, "val_selection_audit", None),
        "train_window_audit": window_audit(train_dataset, args.train_manifest),
        "val_window_audit": window_audit(val_dataset, args.val_manifest),
    }
    if arch == ARCH_ID:
        config.update(
            {
                "backbone": "resnet50_layer3",
                "backbone_weights": None if args.no_pretrained_backbone else BACKBONE_WEIGHTS,
                "backbone_input_channels": 6,
                "spatial_softmax": "full_layer3_channels",
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "cross_view_layers": args.cross_view_layers,
                "temporal_layers": args.temporal_layers,
                "ffn_dim": args.ffn_dim,
                "dropout": args.dropout,
            }
        )
    elif arch == COMPOSITE_ARCH_ID:
        config.update(
            {
                "backbone": "resnet50_layer3",
                "backbone_weights": None if args.no_pretrained_backbone else BACKBONE_WEIGHTS,
                "backbone_input_channels": 6,
                "panel_height": PANEL_TARGET_HEIGHT,
                "panel_width": PANEL_TARGET_WIDTH,
                "compressed_channels": args.compressed_channels,
                "pool_height": args.pool_height,
                "pool_width": args.pool_width,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "temporal_layers": args.temporal_layers,
                "ffn_dim": args.ffn_dim,
                "dropout": args.dropout,
            }
        )
    else:
        config.update(
            {
                "image_size": args.input_height,
                "cnn_width": args.cnn_width,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_encoder_layers": args.n_encoder_layers,
                "n_decoder_layers": args.n_decoder_layers,
                "num_keypoints": args.num_keypoints if arch == "v2" else None,
            }
        )
    return config


def save_checkpoint(path: str, payload: dict) -> None:
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(..., map_location=device) moves every tensor in the
    # checkpoint onto that device, including this CPU-only RNG state buried
    # inside a plain dict - but torch.set_rng_state() requires a CPU tensor
    # specifically (verified: a CUDA-resident uint8 tensor raises "RNG state
    # must be a torch.ByteTensor" even though the dtype matches). The CUDA RNG
    # states below need no such fix - torch.cuda.get_rng_state_all() already
    # returns one CPU-side snapshot tensor per GPU, and set_rng_state_all
    # itself handles placing each back on its own device.
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        # Each per-GPU snapshot from torch.cuda.get_rng_state_all() is also a
        # CPU-side tensor internally (same map_location hazard as above).
        torch.cuda.set_rng_state_all([tensor.cpu() for tensor in state["cuda"]])


def gather_rng_states(distributed: bool, world_size: int, is_main: bool) -> list[dict] | None:
    local_state = capture_rng_state()
    if not distributed:
        return [local_state]
    gathered = [None] * world_size if is_main else None
    dist.gather_object(local_state, gathered, dst=0)
    return gathered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("overfit", "train"), default="train")
    parser.add_argument("--train-manifest")
    parser.add_argument("--val-manifest")
    parser.add_argument("--train-episodes", type=int, nargs="+", default=list(range(22)))
    parser.add_argument("--val-episodes", type=int, nargs="+", default=list(range(22, 27)))
    parser.add_argument("--input-height", type=int, default=DEFAULT_INPUT_HEIGHT)
    parser.add_argument("--input-width", type=int, default=DEFAULT_INPUT_WIDTH)
    parser.add_argument("--train-stride", type=int, default=16)
    parser.add_argument("--val-stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1, help="per-GPU batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8, help="dataloader workers per rank")
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=None, help="defaults to --batch-size")
    parser.add_argument("--log-interval", type=int, default=25, help="micro-batches between scalar logs")
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=0,
        help="smoke test only: stop each epoch after this many micro-batches (0 = full epoch). "
        "Recorded in the checkpoint config, so a limited run can never be mistaken for, "
        "or resumed into, the production run.",
    )
    parser.add_argument(
        "--limit-val-batches", type=int, default=0, help="smoke test only: cap validation batches"
    )
    parser.add_argument(
        "--nonproduction-manifests",
        action="store_true",
        help="probes only: skip the frozen 21K/1K lab x outcome quota check so a small "
        "subset manifest can be used. Recorded in the checkpoint config, so such a "
        "checkpoint can never be mistaken for, or resumed into, the production run. "
        "Scene disjointness is still enforced.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model (about 1.35x faster; adds a one-off warmup)",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"),
        default="default",
        help="torch.compile mode; max-autotune trades a longer one-off compile for faster kernels",
    )
    parser.add_argument("--out-dir", default="/workspace/wise_idm/checkpoints")
    parser.add_argument("--log-dir", default="/workspace/wise_idm/tb_logs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--arch", choices=("wise", ARCH_ID, "composite", COMPOSITE_ARCH_ID, "v1", "v2"), default=ARCH_ID
    )
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--cross-view-layers", type=int, default=2, help="wise (3-view) architecture only")
    parser.add_argument("--temporal-layers", type=int, default=6)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cnn-width", type=int, default=64, help="legacy architectures only")
    parser.add_argument("--n-encoder-layers", type=int, default=4, help="legacy architectures only")
    parser.add_argument("--n-decoder-layers", type=int, default=4, help="legacy architectures only")
    parser.add_argument("--num-keypoints", type=int, default=48, help="legacy v2 only")
    parser.add_argument("--compressed-channels", type=int, default=256, help="composite architecture only")
    parser.add_argument("--pool-height", type=int, default=8, help="composite architecture only")
    parser.add_argument("--pool-width", type=int, default=10, help="composite architecture only")
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--resume", help="resume from a last.pt training checkpoint")
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    args = parser.parse_args()

    if args.mode == "train" and not (args.train_manifest and args.val_manifest):
        parser.error("--mode train requires the finalized --train-manifest and --val-manifest")
    if args.input_height <= 0 or args.input_width <= 0:
        raise ValueError("input dimensions must be positive")
    if args.gradient_accumulation <= 0:
        raise ValueError("gradient accumulation must be positive")
    if args.mode == "train" and (args.train_stride != 16 or args.val_stride != 32):
        raise ValueError("production training is frozen to train stride 16 and validation stride 32")
    arch = canonical_arch(args.arch)
    if arch in ("v1", "v2") and args.input_height != args.input_width:
        raise ValueError("legacy architectures require square input dimensions")

    configure_backends()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda" and args.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("--amp-dtype bf16 requires a CUDA device with native BF16 support")
    is_main = not distributed or dist.get_rank() == 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir) if is_main else None

    train_dataset, val_dataset = build_datasets(args)
    if is_main:
        print(
            f"train: {len(train_dataset.episodes)} episodes, {len(train_dataset)} windows, stride={args.train_stride}; "
            f"val: {len(val_dataset.episodes)} episodes, {len(val_dataset)} windows, stride={args.val_stride}"
        )
        if train_dataset.zero_window_episodes or val_dataset.zero_window_episodes:
            print(
                "episodes shorter than 33 frames: "
                f"train={train_dataset.zero_window_episodes}, val={val_dataset.zero_window_episodes}"
            )
        print("computing train-window joint statistics and gripper class balance (no video decode) ...")
    stats, positive_weight_value, positive_count, negative_count = compute_stats_fast(train_dataset)
    if is_main:
        print("joint mean:", np.round(stats["mean"], 3))
        print("joint std:", np.round(stats["std"], 3))
        print(
            f"gripper pos_weight={positive_weight_value:.3f} "
            f"(positive={positive_count}, negative={negative_count})"
        )

    config = checkpoint_config(args, train_dataset, val_dataset, world_size)
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location=device, weights_only=False)
        if resume_payload["config"] != config:
            raise ValueError("resume checkpoint config does not exactly match this training configuration")
        checkpoint_stats = resume_payload.get("joint_stats")
        if checkpoint_stats is None or not (
            np.array_equal(checkpoint_stats["mean"], stats["mean"])
            and np.array_equal(checkpoint_stats["std"], stats["std"])
        ):
            raise ValueError("resume checkpoint train-action statistics do not match this dataset")
    load_pretrained_backbone = (
        arch in (ARCH_ID, COMPOSITE_ARCH_ID)
        and resume_payload is None
        and not args.no_pretrained_backbone
    )
    model = build_model(
        config,
        load_pretrained_backbone=load_pretrained_backbone and is_main,
    ).to(device)
    if distributed and load_pretrained_backbone:
        # Rank 0 may need to download the weights; all other ranks wait before
        # DDP broadcasts rank 0's initialized parameters.
        dist.barrier()
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if is_main:
        print(f"arch={arch}, parameters={parameter_count / 1e6:.2f}M, batch_keys={batch_view_keys(arch)}")
    # NHWC convolutions are ~1.5x faster for this backbone on A100 and change
    # only the kernel layout, not the arithmetic the checkpoint encodes.
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    # ``unwrapped`` stays the plain module: checkpoints and evaluation use it,
    # so state_dict keys never gain a DDP/compile wrapper prefix.
    unwrapped = model
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], gradient_as_bucket_view=True
        )
    step_model = (
        torch.compile(model, mode=None if args.compile_mode == "default" else args.compile_mode)
        if args.compile
        else model
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    positive_weight = (
        torch.tensor(positive_weight_value, device=device) if args.mode == "train" else None
    )

    amp_dtype = None
    if device.type == "cuda" and args.amp_dtype != "none":
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16
    )

    if args.mode == "overfit":
        loader = DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), shuffle=False)
        batch = next(iter(loader))
        target = batch["action"].to(device)
        joints_target, gripper_target = target[..., :7], target[..., 7]
        joints_target_norm = normalize_joints(joints_target, stats, device)
        started = time.time()
        model.train()
        for step in range(300):
            optimizer.zero_grad(set_to_none=True)
            output = call_model(step_model, arch, batch, device, batch_view_keys(arch))
            loss, joint_loss, gripper_loss = loss_fn(
                output["joints"], output["gripper_logit"], joints_target_norm, gripper_target
            )
            loss.backward()
            optimizer.step()
            if step % 20 == 0 or step == 299:
                joints_pred = denormalize_joints(output["joints"], stats, device)
                joint_mae = (joints_pred - joints_target).abs().mean().item()
                gripper_probability = torch.sigmoid(output["gripper_logit"].squeeze(-1))
                gripper_accuracy = (
                    ((gripper_probability > 0.5) == (gripper_target > 0.5)).float().mean().item()
                )
                print(
                    f"step {step:4d} loss={loss.item():.5f} joint_loss={joint_loss.item():.5f} "
                    f"gripper_bce={gripper_loss.item():.5f} joint_mae={joint_mae:.5f} "
                    f"gripper_acc={gripper_accuracy:.3f} ({time.time() - started:.1f}s)"
                )
        if writer:
            writer.close()
        if distributed:
            dist.destroy_process_group()
        return

    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed) if distributed else None
    prefetch = {"prefetch_factor": args.prefetch_factor} if args.num_workers > 0 else {}
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=True,
        **prefetch,
    )
    rank = dist.get_rank() if distributed else 0
    val_rank_indices = list(range(rank, len(val_dataset), world_size))
    val_subset = torch.utils.data.Subset(val_dataset, val_rank_indices)
    val_workers = max(1, args.num_workers // 2) if args.num_workers > 0 else 0
    val_loader = DataLoader(
        val_subset,
        batch_size=args.val_batch_size or args.batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=val_workers > 0,
        **({"prefetch_factor": args.prefetch_factor} if val_workers > 0 else {}),
    )

    batches_per_epoch = len(train_loader)
    if args.limit_train_batches:
        batches_per_epoch = min(batches_per_epoch, args.limit_train_batches)
    if batches_per_epoch == 0:
        raise ValueError("the training loader yields no batches; lower --batch-size")
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation)
    total_steps = args.epochs * optimizer_steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.05,
        anneal_strategy="cos",
    )
    start_epoch = 0
    global_step = 0
    best_val_mae = float("inf")
    history = []
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        scaler.load_state_dict(resume_payload.get("scaler_state_dict", {}))
        start_epoch = int(resume_payload["epoch"]) + 1
        global_step = int(resume_payload["global_step"])
        best_val_mae = float(resume_payload["best_val_mae"])
        history = list(resume_payload.get("history", []))
        rng_states = resume_payload.get("rng_states")
        if rng_states:
            if len(rng_states) != world_size:
                raise ValueError(
                    f"resume checkpoint has RNG state for {len(rng_states)} ranks, current world_size={world_size}"
                )
            restore_rng_state(rng_states[rank])

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        unwrapped.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.time()
        # Accumulated on the GPU: reading these every micro-batch would force a
        # host sync and stall the input pipeline.
        running = torch.zeros(3, device=device, dtype=torch.float64)
        batch_count = 0
        windows_seen = 0
        last_report = started

        for batch_index, batch in enumerate(train_loader):
            if batch_index >= batches_per_epoch:
                break
            group_start = (batch_index // args.gradient_accumulation) * args.gradient_accumulation
            group_size = min(args.gradient_accumulation, batches_per_epoch - group_start)
            should_step = batch_index + 1 == group_start + group_size
            target = batch["action"].to(device, non_blocking=True)
            joints_target, gripper_target = target[..., :7], target[..., 7]
            joints_target_norm = normalize_joints(joints_target, stats, device)
            sync_context = nullcontext() if should_step or not distributed else model.no_sync()
            with sync_context:
                with amp_context(device, amp_dtype):
                    output = call_model(step_model, arch, batch, device, batch_view_keys(arch))
                    loss, joint_loss, gripper_loss = loss_fn(
                        output["joints"],
                        output["gripper_logit"],
                        joints_target_norm,
                        gripper_target,
                        gripper_pos_weight=positive_weight,
                    )
                    scaled_loss = loss / group_size
                scaler.scale(scaled_loss).backward()

            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running += torch.stack((loss.detach().double(), joint_loss.double(), gripper_loss.double()))
            batch_count += 1
            windows_seen += target.shape[0]
            if writer and batch_index % args.log_interval == 0:
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/joint_loss", joint_loss.item(), global_step)
                writer.add_scalar("train/gripper_bce", gripper_loss.item(), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            if is_main and batch_index and batch_index % (args.log_interval * 20) == 0:
                now = time.time()
                rate = args.log_interval * 20 * args.batch_size / (now - last_report)
                last_report = now
                print(
                    f"  epoch {epoch} batch {batch_index}/{batches_per_epoch} "
                    f"loss={loss.item():.5f} {rate:.1f} win/s/rank "
                    f"({rate * world_size:.1f} win/s total) "
                    f"mem={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                    flush=True,
                )
                if writer:
                    writer.add_scalar("throughput/windows_per_second_total", rate * world_size, global_step)

        evaluation_model = unwrapped
        if distributed:
            for buffer in evaluation_model.buffers():
                dist.broadcast(buffer, src=0)
        metrics = evaluate(
            evaluation_model,
            val_loader,
            stats,
            device,
            arch=arch,
            cameras=batch_view_keys(arch),
            amp_dtype=None,
            distributed=distributed,
            max_batches=args.limit_val_batches,
        )
        train_totals = torch.cat(
            (running, torch.tensor([batch_count, windows_seen], device=device, dtype=torch.float64))
        )
        if distributed:
            dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
        mean_train_loss = (train_totals[0] / train_totals[3]).item()
        mean_joint_loss = (train_totals[1] / train_totals[3]).item()
        mean_gripper_loss = (train_totals[2] / train_totals[3]).item()
        rng_states = gather_rng_states(distributed, world_size, is_main)
        if is_main:
            elapsed = time.time() - started
            print(
                f"epoch {epoch:3d} train_loss={mean_train_loss:.5f} "
                f"val_mean_joint_mae={metrics['mean_joint_mae']:.5f} "
                f"val_gripper_acc={metrics['gripper_accuracy']:.3f} "
                f"({elapsed:.1f}s, {train_totals[4].item() / elapsed:.1f} win/s)",
                flush=True,
            )
            writer.add_scalar("val/mean_joint_mae", metrics["mean_joint_mae"], epoch)
            writer.add_scalar("val/gripper_accuracy", metrics["gripper_accuracy"], epoch)
            writer.add_scalar("val/gripper_mae", metrics["gripper_mae"], epoch)
            writer.add_scalar("train/epoch_loss", mean_train_loss, epoch)
            writer.add_scalar("train/epoch_seconds", elapsed, epoch)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": mean_train_loss,
                    "train_joint_loss": mean_joint_loss,
                    "train_gripper_loss": mean_gripper_loss,
                    "epoch_seconds": elapsed,
                    "train_windows": int(train_totals[4].item()),
                    **metrics,
                }
            )

            improved = metrics["mean_joint_mae"] < best_val_mae
            if improved:
                best_val_mae = metrics["mean_joint_mae"]
            payload = {
                "model_state_dict": evaluation_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "rng_states": rng_states,
                "joint_stats": stats,
                "config": config,
                "epoch": epoch,
                "global_step": global_step,
                "best_val_mae": best_val_mae,
                "val_metrics": metrics,
                "history": history,
            }
            save_checkpoint(os.path.join(args.out_dir, "last.pt"), payload)
            if improved:
                save_checkpoint(os.path.join(args.out_dir, "best.pt"), payload)
                print(f"  -> saved new best checkpoint (val_mean_joint_mae={best_val_mae:.5f})")
            with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)

    if writer:
        writer.close()
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
