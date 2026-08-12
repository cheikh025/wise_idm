#!/usr/bin/env python3
"""IDM r_cons scoring HTTP server for WISE Best-of-K (M6 prep).

RoboLab (Python 3.11, torch 2.7.0) and wise_idm (Python 3.11, torch 2.11.0)
run different torch versions -- following the same cross-process isolation
pattern already used for Robometer (research/tools/robometer_server.py):
wrap the IDM behind a minimal HTTP endpoint rather than import it directly
into RoboLab's process.

r_cons per METHOD.md: run the IDM on the candidate's imagined video to get
atilde^(i), compare against Cosmos's own co-generated a^(i) in a normalized
action space, gripper handled separately since its executed representation
is thresholded. No hard-reject, purely continuous, matching r_exec's
established convention in this project.

Endpoints:
  GET  /health -> {"status": "ok", "checkpoint_epoch": ..., "val_metrics": ...}
  POST /score_consistency -> multipart form:
      dream="dream.npy"  ((T,H,W,3) uint8 RGB, full panel -- wrist+left+right
                           stacked, same layout Cosmos3-Edge produces)
      action="action.npy" ((32,8) float32 -- Cosmos's own co-generated action
                            for this candidate: 7 joints + gripper)
    returns {"r_cons": float, "joint_cons": float, "gripper_cons": float,
             "joint_mae_std_units": float}
"""
import argparse
import io

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile

from model import DroidIDM

app = FastAPI(title="WISE IDM r_cons scoring server (M6 prep)")
_state = {}


def detect_seam(frame: np.ndarray) -> int:
    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    d = np.abs(np.diff(g, axis=0)).mean(axis=1)
    lo, hi = int(len(d) * 0.55), int(len(d) * 0.78)
    return lo + int(d[lo:hi].argmax())


def split_panel(dream: np.ndarray) -> dict:
    """Same panel geometry as research/tools/score_robometer.py's split_panel
    and policies/wise/bestofk_client.py's _dream_wrist_view, extended to all
    3 views (r_cons needs the full multi-camera input the IDM was trained
    with, not just wrist)."""
    h, w = dream.shape[1:3]
    seam = round(h * 2 / 3)
    found = detect_seam(dream[0])
    if abs(found - seam) > 4:
        seam = found
    return {"wrist": dream[:, :seam], "left": dream[:, seam:, :w // 2], "right": dream[:, seam:, w // 2:]}


def resize_stack(frames: np.ndarray, size: int) -> torch.Tensor:
    out = np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames])
    return torch.from_numpy(out).permute(0, 3, 1, 2).float() / 255.0


def load_idm(checkpoint: str, device: str):
    dev = torch.device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)
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
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return {"model": model, "stats": ckpt["joint_stats"], "cfg": cfg, "cameras": cameras, "device": dev,
            "epoch": ckpt["epoch"], "val_metrics": ckpt["val_metrics"]}


def score_consistency(idm, dream: np.ndarray, cosmos_action: np.ndarray) -> dict:
    """dream: (T,H,W,3) full panel. cosmos_action: (32,8) 7 joints + gripper,
    Cosmos's own co-generated action for this same candidate."""
    device = idm["device"]
    cfg = idm["cfg"]
    panel_views = split_panel(dream)
    views = [resize_stack(panel_views[cam], cfg["image_size"]).unsqueeze(0).to(device) for cam in idm["cameras"]]

    with torch.no_grad():
        out = idm["model"](views)
        mean = torch.tensor(idm["stats"]["mean"], device=device)
        std = torch.tensor(idm["stats"]["std"], device=device)
        idm_joints = (out["joints"] * std + mean)[0].cpu().numpy()  # (32,7) radians
        idm_gripper = torch.sigmoid(out["gripper_logit"])[0, :, 0].cpu().numpy()  # (32,)

    cosmos_joints = cosmos_action[:, :7]
    cosmos_gripper = cosmos_action[:, 7]

    # Joint consistency: normalize both sides into the IDM's own standardized
    # units (per-joint std from training) before comparing, so no single
    # naturally-larger-range joint dominates the error -- same reasoning as
    # the standardized SmoothL1 training loss.
    std_np = np.array(idm["stats"]["std"])
    joint_diff_std_units = np.abs(idm_joints - cosmos_joints) / std_np
    joint_mae_std_units = float(joint_diff_std_units.mean())
    joint_cons = float(np.exp(-joint_mae_std_units / 1.0))  # continuous, no hard reject, matches r_exec's style

    # Gripper: thresholded per METHOD.md ("treat the gripper separately if
    # its executed representation is discrete/thresholded") -- agreement
    # rate on the binary open/closed decision, not raw continuous distance.
    idm_bin = (idm_gripper > 0.5).astype(np.float32)
    cosmos_bin = (cosmos_gripper > 0.5).astype(np.float32)
    gripper_cons = float((idm_bin == cosmos_bin).mean())

    r_cons = float(0.5 * joint_cons + 0.5 * gripper_cons)
    return {"r_cons": r_cons, "joint_cons": joint_cons, "gripper_cons": gripper_cons,
            "joint_mae_std_units": joint_mae_std_units}


@app.get("/health")
async def health():
    return {"status": "ok", "checkpoint_epoch": _state["idm"]["epoch"],
            "val_metrics": _state["idm"]["val_metrics"]}


@app.post("/score_consistency")
async def score_endpoint(dream: UploadFile = File(...), action: UploadFile = File(...)):
    dream_arr = np.load(io.BytesIO(await dream.read()))
    action_arr = np.load(io.BytesIO(await action.read()))
    return score_consistency(_state["idm"], dream_arr, action_arr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/workspace/wise_idm/checkpoints_500ep/best.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8101)
    a = p.parse_args()

    print(f"[idm-server] loading {a.checkpoint} on {a.device} ...")
    _state["idm"] = load_idm(a.checkpoint, a.device)
    print(f"[idm-server] loaded: epoch={_state['idm']['epoch']} val_metrics={_state['idm']['val_metrics']}")
    print(f"[idm-server] ready on {a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port)


if __name__ == "__main__":
    main()
