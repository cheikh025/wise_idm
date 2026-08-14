"""Versioned model construction and forward dispatch for IDM checkpoints."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from model import DroidIDM
from model_v2 import DroidIDMv2
from model_wise import ARCH_ID, CAMERA_ORDER, WiseIDM
from vision import PANEL_LAYOUT_VERSION, VISION_PREPROCESS_VERSION


def canonical_arch(arch: str | None) -> str:
    if arch in (None, "v1"):
        return "v1"
    if arch == "v2":
        return "v2"
    if arch in ("wise", ARCH_ID):
        return ARCH_ID
    raise ValueError(f"unknown IDM architecture: {arch}")


def build_model(config: Mapping, *, load_pretrained_backbone: bool = False) -> nn.Module:
    """Build from checkpoint/training config.

    Checkpoint reload defaults to no ImageNet download because the full
    backbone state is already present in ``model_state_dict``. New training
    explicitly passes ``load_pretrained_backbone=True``.
    """
    arch = canonical_arch(config.get("arch"))
    if arch == ARCH_ID:
        if config.get("panel_layout_version", PANEL_LAYOUT_VERSION) != PANEL_LAYOUT_VERSION:
            raise ValueError(
                f"{ARCH_ID} requires panel layout {PANEL_LAYOUT_VERSION}, "
                f"got {config.get('panel_layout_version')!r}"
            )
        if (
            config.get("vision_preprocess_version", VISION_PREPROCESS_VERSION)
            != VISION_PREPROCESS_VERSION
        ):
            raise ValueError(
                f"{ARCH_ID} requires vision preprocessing {VISION_PREPROCESS_VERSION}, "
                f"got {config.get('vision_preprocess_version')!r}"
            )
        cameras = tuple(config.get("cameras", CAMERA_ORDER))
        if cameras != CAMERA_ORDER:
            raise ValueError(f"{ARCH_ID} requires camera order {CAMERA_ORDER}, got {cameras}")
        return WiseIDM(
            input_height=int(config.get("input_height", 128)),
            input_width=int(config.get("input_width", 224)),
            num_frames=int(config.get("num_frames", 33)),
            num_cameras=len(cameras),
            action_horizon=int(config.get("action_horizon", config.get("chunk_len", 32))),
            d_model=int(config.get("d_model", 512)),
            n_heads=int(config.get("n_heads", 8)),
            cross_view_layers=int(config.get("cross_view_layers", 2)),
            temporal_layers=int(config.get("temporal_layers", 6)),
            ffn_dim=int(config.get("ffn_dim", 2048)),
            dropout=float(config.get("dropout", 0.1)),
            pretrained_backbone=load_pretrained_backbone,
        )

    image_size = int(config.get("image_size", 128))
    cameras = list(config.get("cameras", ("wrist", "left", "right")))
    common = dict(
        image_size=image_size,
        num_frames=int(config.get("num_frames", 33)),
        cnn_width=int(config.get("cnn_width", 64)),
        d_model=int(config.get("d_model", 256)),
        n_heads=int(config.get("n_heads", 8)),
        n_encoder_layers=int(config.get("n_encoder_layers", 4)),
        n_decoder_layers=int(config.get("n_decoder_layers", 4)),
    )
    if arch == "v2":
        if cameras != ["wrist", "left", "right"]:
            raise ValueError("legacy v2 checkpoints require all three views")
        return DroidIDMv2(num_keypoints=int(config.get("num_keypoints", 48)), **common)
    return DroidIDM(num_cameras=len(cameras), **common)


def forward_model(
    model: nn.Module, arch: str | None, views: Sequence[torch.Tensor]
) -> dict[str, torch.Tensor]:
    arch = canonical_arch(arch)
    if arch == "v2":
        return model(*views)
    return model(views)


def checkpoint_input_geometry(config: Mapping) -> tuple[int, int]:
    if canonical_arch(config.get("arch")) == ARCH_ID:
        return int(config.get("input_height", 128)), int(config.get("input_width", 224))
    size = int(config.get("image_size", 128))
    return size, size
