"""M5: run the trained IDM on real Cosmos3-Edge dream video (not held-out DROID
data) -- reuses the exact panel-splitting logic from research/tools/score_robometer.py
(seam-ratio h*2/3, with detected-edge fallback if it disagrees by >4px).
"""
import argparse
import subprocess

import cv2
import numpy as np
import torch

from model import DroidIDM


def decode_mp4(path: str) -> np.ndarray:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True)
    w, h, n = proc.stdout.strip().split(",")
    w, h, n = int(w), int(h), int(n)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(n, h, w, 3)


def detect_seam(frame: np.ndarray) -> int:
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    d = np.abs(np.diff(g, axis=0)).mean(axis=1)
    lo, hi = int(len(d) * 0.55), int(len(d) * 0.78)
    return lo + int(d[lo:hi].argmax())


def split_panel(dream: np.ndarray) -> dict:
    h, w = dream.shape[1:3]
    seam = round(h * 2 / 3)
    found = detect_seam(dream[0])
    if abs(found - seam) > 4:
        seam = found
    return {"wrist": dream[:, :seam], "left": dream[:, seam:, :w // 2], "right": dream[:, seam:, w // 2:]}


def resize_stack(frames: np.ndarray, size: int) -> torch.Tensor:
    out = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames])
    return torch.from_numpy(out).permute(0, 3, 1, 2).float() / 255.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dream", required=True)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    if cfg.get("use_proprio", False):
        raise RuntimeError(
            "This checkpoint was trained with proprioception, which model.py's DroidIDM no longer "
            "supports (removed permanently, project direction). Not loadable via current code -- "
            "check out an earlier commit (045c0e3 or before) in the wise_idm git history if needed.")
    cameras = cfg.get("cameras", ["wrist", "left", "right"])
    model = DroidIDM(
        image_size=cfg["image_size"], num_frames=cfg["num_frames"], num_cameras=len(cameras),
        cnn_width=cfg.get("cnn_width", 64), d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 8), n_encoder_layers=cfg.get("n_encoder_layers", 4),
        n_decoder_layers=cfg.get("n_decoder_layers", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    stats = ckpt["joint_stats"]
    print(f"loaded checkpoint: epoch={ckpt['epoch']} val_metrics={ckpt['val_metrics']}")

    dream = decode_mp4(a.dream)
    print(f"dream video: {dream.shape}")
    panel_views = split_panel(dream)
    for k, v in panel_views.items():
        print(f"  {k}: {v.shape}")

    views = [resize_stack(panel_views[cam], cfg["image_size"]).unsqueeze(0).to(device) for cam in cameras]

    with torch.no_grad():
        out = model(views)
        mean = torch.tensor(stats["mean"], device=device)
        std = torch.tensor(stats["std"], device=device)
        joints = (out["joints"] * std + mean)[0].cpu().numpy()  # (32,7) denormalized radians
        gripper = torch.sigmoid(out["gripper_logit"])[0, :, 0].cpu().numpy()  # (32,)

    print("\npredicted joints (rad), first/mid/last step:")
    print(" t=0 :", np.round(joints[0], 3))
    print(" t=16:", np.round(joints[16], 3))
    print(" t=31:", np.round(joints[-1], 3))
    print("\npredicted gripper (prob open->closed), first 8 / last 8:")
    print(" start:", np.round(gripper[:8], 3))
    print(" end  :", np.round(gripper[-8:], 3))

    vel = np.diff(joints, axis=0) * 15.0  # rad/s at 15fps
    print(f"\njoint velocity: mean|v|={np.abs(vel).mean():.3f} rad/s, max|v|={np.abs(vel).max():.3f} rad/s")
    jump_at_seam = np.abs(vel).max(axis=1)
    print(f"largest single-step jump: {jump_at_seam.max():.3f} rad/s at step {int(jump_at_seam.argmax())}")


if __name__ == "__main__":
    main()
