"""Reload an IDM checkpoint and reproduce its recorded validation metrics."""
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from droid_dataset import DroidIDMDataset, load_episode_manifest
from droid_panel_dataset import DroidIDMPanelDataset
from model_composite import ARCH_ID as COMPOSITE_ARCH_ID
from model_factory import (
    build_model,
    canonical_arch,
    checkpoint_input_geometry,
    configure_backends,
)
from model_wise import ARCH_ID, CAMERA_ORDER
from train import batch_view_keys, episode_selection_digest, evaluate, manifest_file_digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--val-manifest")
    source.add_argument("--val-episodes", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    configure_backends()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    print(f"checkpoint epoch={checkpoint['epoch']} recorded val_metrics={checkpoint['val_metrics']}")
    if config.get("use_proprio", config.get("uses_proprioception", False)):
        raise RuntimeError("proprioception checkpoints are not supported by the vision-only WISE-IDM")
    arch = canonical_arch(config.get("arch"))
    if arch not in (ARCH_ID, COMPOSITE_ARCH_ID):
        raise RuntimeError(
            "legacy validation caches used a different resize pipeline; this verifier only "
            f"supports {ARCH_ID} and {COMPOSITE_ARCH_ID} checkpoints"
        )
    required = {
        "dataset_repo",
        "dataset_revision",
        "train_selection_sha256",
        "val_selection_sha256",
        "train_manifest_sha256",
        "val_manifest_sha256",
    }
    missing = sorted(key for key in required if not config.get(key))
    if missing:
        raise RuntimeError(
            "production checkpoint is missing data provenance fields: " + ", ".join(missing)
        )

    model = build_model(config, load_pretrained_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    if arch == COMPOSITE_ARCH_ID:
        if not args.val_manifest:
            raise RuntimeError(f"{COMPOSITE_ARCH_ID} checkpoints require --val-manifest")
        dataset = DroidIDMPanelDataset(
            episodes=load_episode_manifest(args.val_manifest),
            stride=int(config.get("val_stride", 32)),
            end_align_tail=bool(config.get("end_align_tail", True)),
        )
    else:
        input_height, input_width = checkpoint_input_geometry(config)
        dataset_args = dict(
            input_height=input_height,
            input_width=input_width,
            num_frames=int(config.get("num_frames", 33)),
            chunk_len=int(config.get("chunk_len", 32)),
            stride=int(config.get("val_stride", 32)),
            end_align_tail=bool(config.get("end_align_tail", True)),
        )
        if args.val_manifest:
            dataset = DroidIDMDataset(
                episodes=load_episode_manifest(args.val_manifest),
                **dataset_args,
            )
        else:
            dataset = DroidIDMDataset(
                episode_indices=args.val_episodes,
                dataset_split="success",
                **dataset_args,
            )
    expected_selection = config.get("val_selection_sha256")
    if expected_selection is not None:
        actual_selection = episode_selection_digest(dataset)
        if actual_selection != expected_selection:
            raise ValueError(
                "validation episode selection does not match the checkpoint fingerprint"
            )
    expected_manifest = config.get("val_manifest_sha256")
    if expected_manifest is not None:
        if not args.val_manifest:
            raise ValueError("this checkpoint requires its validation manifest file")
        if manifest_file_digest(args.val_manifest) != expected_manifest:
            raise ValueError("validation manifest file does not match the checkpoint fingerprint")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    metrics = evaluate(
        model,
        loader,
        checkpoint["joint_stats"],
        device,
        arch=config.get("arch"),
        cameras=batch_view_keys(arch),
    )
    print(f"reloaded, re-evaluated val_metrics={metrics}")

    recorded = checkpoint["val_metrics"]
    joint_mae_difference = abs(metrics["mean_joint_mae"] - recorded["mean_joint_mae"])
    gripper_accuracy_difference = abs(metrics["gripper_accuracy"] - recorded["gripper_accuracy"])
    print(
        f"\ndiff: joint_mae={joint_mae_difference:.6f}, "
        f"gripper_acc={gripper_accuracy_difference:.6f}"
    )
    matches = joint_mae_difference < 1e-4 and gripper_accuracy_difference < 1e-4
    print("MATCH" if matches else "MISMATCH -- investigate")
    if not matches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
