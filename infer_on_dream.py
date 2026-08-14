"""Run a trained WISE-IDM checkpoint on one Cosmos DROID dream video."""
from __future__ import annotations

import argparse
import subprocess

import numpy as np
import torch

from model_factory import build_model, canonical_arch, checkpoint_input_geometry, forward_model
from model_wise import ARCH_ID, CAMERA_ORDER
from vision import split_cosmos_panel, video_to_tensor


def decode_mp4(path: str) -> np.ndarray:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    width, height = (int(value) for value in probe.stdout.strip().split(","))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    ).stdout
    frame_bytes = height * width * 3
    if len(raw) % frame_bytes:
        raise RuntimeError("decoded video byte count is not divisible by the probed frame size")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dream", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    if config.get("use_proprio", config.get("uses_proprioception", False)):
        raise RuntimeError("proprioception checkpoints are not supported by the vision-only WISE-IDM")
    cameras = tuple(config.get("cameras", CAMERA_ORDER))
    arch = canonical_arch(config.get("arch"))
    model = build_model(config, load_pretrained_backbone=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"loaded checkpoint: epoch={checkpoint['epoch']} val_metrics={checkpoint['val_metrics']}")

    dream = decode_mp4(args.dream)
    expected_frames = int(config.get("num_frames", 33))
    panel_views = split_cosmos_panel(dream, expected_frames=expected_frames)
    input_height, input_width = checkpoint_input_geometry(config)
    views = [
        video_to_tensor(
            panel_views[camera],
            input_height,
            input_width,
            preserve_aspect=arch == ARCH_ID,
        ).unsqueeze(0).to(device)
        for camera in cameras
    ]
    print(f"dream video: {dream.shape}")
    for camera in cameras:
        print(f"  {camera}: source={panel_views[camera].shape}, model={tuple(views[cameras.index(camera)].shape)}")

    with torch.no_grad():
        output = forward_model(model, arch, views)
        mean = torch.as_tensor(checkpoint["joint_stats"]["mean"], device=device)
        std = torch.as_tensor(checkpoint["joint_stats"]["std"], device=device)
        joints = (output["joints"] * std + mean)[0].cpu().numpy()
        gripper = torch.sigmoid(output["gripper_logit"])[0, :, 0].cpu().numpy()

    middle = len(joints) // 2
    print("\npredicted joints (rad), first/middle/last step:")
    print(" t=0 :", np.round(joints[0], 3))
    print(f" t={middle}:", np.round(joints[middle], 3))
    print(f" t={len(joints) - 1}:", np.round(joints[-1], 3))
    print("\npredicted gripper probability, first 8 / last 8:")
    print(" start:", np.round(gripper[:8], 3))
    print(" end  :", np.round(gripper[-8:], 3))

    velocity = np.diff(joints, axis=0) * 15.0
    print(
        f"\njoint velocity: mean|v|={np.abs(velocity).mean():.3f} rad/s, "
        f"max|v|={np.abs(velocity).max():.3f} rad/s"
    )


if __name__ == "__main__":
    main()
