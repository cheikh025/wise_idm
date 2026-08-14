"""Shared image geometry for DROID training data and Cosmos dream videos."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


CAMERA_ORDER = ("wrist", "left", "right")
DROID_CAMERA_ORDER = (
    "wrist_image_left",
    "exterior_image_1_left",
    "exterior_image_2_left",
)
PANEL_LAYOUT_VERSION = "cosmos3_droid_dream_528x640_v1"
PANEL_HEIGHT = 528
PANEL_WIDTH = 640
PANEL_SEAM = 360
DEFAULT_INPUT_HEIGHT = 128
DEFAULT_INPUT_WIDTH = 224
VISION_PREPROCESS_VERSION = "letterbox_torch_bilinear_antialias_v1"


def validate_rgb_video(frames: np.ndarray, *, expected_frames: int | None = None) -> None:
    if not isinstance(frames, np.ndarray):
        raise TypeError("video must be a numpy array")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"video must have shape (T, H, W, 3), got {frames.shape}")
    if frames.dtype != np.uint8:
        raise TypeError(f"video must contain uint8 RGB pixels, got {frames.dtype}")
    if expected_frames is not None and frames.shape[0] != expected_frames:
        raise ValueError(f"expected {expected_frames} frames, got {frames.shape[0]}")
    if frames.shape[1] < 3 or frames.shape[2] < 2:
        raise ValueError(f"video panels are too small: {frames.shape[1:3]}")


def split_cosmos_panel(dream: np.ndarray, *, expected_frames: int = 33) -> dict[str, np.ndarray]:
    """Split the empirically verified decoded Cosmos DROID dream layout."""
    validate_rgb_video(dream, expected_frames=expected_frames)
    _, height, width, _ = dream.shape
    if (height, width) != (PANEL_HEIGHT, PANEL_WIDTH):
        raise ValueError(
            f"Cosmos DROID decoded dream must be {PANEL_HEIGHT}x{PANEL_WIDTH}, "
            f"got {height}x{width}"
        )

    seam = PANEL_SEAM
    half_width = width // 2
    views = {
        "wrist": dream[:, :seam, :],
        "left": dream[:, seam:, :half_width],
        "right": dream[:, seam:, half_width:],
    }
    expected_shapes = {"wrist": (360, 640), "left": (168, 320), "right": (168, 320)}
    shapes = {name: value.shape[1:3] for name, value in views.items()}
    if shapes != expected_shapes:
        raise ValueError(f"decoded dream panel shapes {shapes} differ from {expected_shapes}")
    return views


def letterbox_rgb(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an RGB video without cropping or changing its aspect ratio."""
    validate_rgb_video(frames)
    if height <= 0 or width <= 0:
        raise ValueError("target image dimensions must be positive")

    source_height, source_width = frames.shape[1:3]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, min(width, round(source_width * scale)))
    resized_height = max(1, min(height, round(source_height * scale)))
    tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float()
    resized = F.interpolate(
        tensor,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    ).round().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()

    output = np.zeros((frames.shape[0], height, width, 3), dtype=frames.dtype)
    top = (height - resized_height) // 2
    left = (width - resized_width) // 2
    output[:, top : top + resized_height, left : left + resized_width] = resized
    return output


def resize_rgb_stretch(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    """Legacy direct resize used by v1/v2 checkpoints."""
    validate_rgb_video(frames)
    if height <= 0 or width <= 0:
        raise ValueError("target image dimensions must be positive")
    tensor = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float()
    return (
        F.interpolate(
            tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        .round()
        .clamp_(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .numpy()
    )


def video_to_tensor(
    frames: np.ndarray,
    height: int,
    width: int,
    *,
    preserve_aspect: bool = True,
) -> torch.Tensor:
    resized = (
        letterbox_rgb(frames, height, width)
        if preserve_aspect
        else resize_rgb_stretch(frames, height, width)
    )
    return torch.from_numpy(resized.copy()).permute(0, 3, 1, 2).float().div_(255.0)
