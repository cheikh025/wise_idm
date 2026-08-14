#!/usr/bin/env python3
"""HTTP scoring service for WISE video/action consistency (``r_cons``)."""
from __future__ import annotations

import argparse
import io

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from model_factory import build_model, canonical_arch, checkpoint_input_geometry, forward_model
from model_wise import ARCH_ID, CAMERA_ORDER
from vision import PANEL_LAYOUT_VERSION, split_cosmos_panel, video_to_tensor


app = FastAPI(title="WISE-IDM r_cons scoring server")
_state: dict = {}


def load_idm(checkpoint_path: str, device: str) -> dict:
    target_device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
    config = checkpoint["config"]
    if config.get("use_proprio", config.get("uses_proprioception", False)):
        raise RuntimeError("proprioception checkpoints are not supported by the vision-only WISE-IDM")
    if canonical_arch(config.get("arch")) == ARCH_ID:
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

    model = build_model(config, load_pretrained_backbone=False).to(target_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return {
        "model": model,
        "stats": checkpoint["joint_stats"],
        "config": config,
        "arch": canonical_arch(config.get("arch")),
        "cameras": tuple(config.get("cameras", CAMERA_ORDER)),
        "device": target_device,
        "epoch": checkpoint["epoch"],
        "val_metrics": checkpoint["val_metrics"],
    }


def score_consistency(idm: dict, dream: np.ndarray, cosmos_action: np.ndarray) -> dict:
    """Score one paired 33-frame dream and 32x8 absolute action chunk."""
    config = idm["config"]
    expected_frames = int(config.get("num_frames", 33))
    expected_horizon = int(config.get("action_horizon", config.get("chunk_len", 32)))
    if cosmos_action.shape != (expected_horizon, 8):
        raise ValueError(
            f"action must have shape ({expected_horizon}, 8), got {cosmos_action.shape}"
        )
    if not np.issubdtype(cosmos_action.dtype, np.number) or not np.isfinite(cosmos_action).all():
        raise ValueError("action must contain only finite numeric values")

    panel_views = split_cosmos_panel(dream, expected_frames=expected_frames)
    input_height, input_width = checkpoint_input_geometry(config)
    views = [
        video_to_tensor(
            panel_views[camera],
            input_height,
            input_width,
            preserve_aspect=idm["arch"] == ARCH_ID,
        )
        .unsqueeze(0)
        .to(idm["device"])
        for camera in idm["cameras"]
    ]

    with torch.no_grad():
        output = forward_model(idm["model"], idm["arch"], views)
        mean = torch.as_tensor(idm["stats"]["mean"], device=idm["device"])
        std = torch.as_tensor(idm["stats"]["std"], device=idm["device"])
        idm_joints = (output["joints"] * std + mean)[0].cpu().numpy()
        idm_gripper = torch.sigmoid(output["gripper_logit"])[0, :, 0].cpu().numpy()

    cosmos_action = cosmos_action.astype(np.float32, copy=False)
    cosmos_joints = cosmos_action[:, :7]
    cosmos_gripper = cosmos_action[:, 7]
    training_std = np.asarray(idm["stats"]["std"], dtype=np.float32)
    joint_mae_std_units = float((np.abs(idm_joints - cosmos_joints) / training_std).mean())
    joint_consistency = float(np.exp(-joint_mae_std_units))

    idm_binary = idm_gripper > 0.5
    cosmos_binary = cosmos_gripper > 0.5
    gripper_consistency = float((idm_binary == cosmos_binary).mean())
    r_cons = float(0.5 * joint_consistency + 0.5 * gripper_consistency)
    return {
        "r_cons": r_cons,
        "joint_cons": joint_consistency,
        "gripper_cons": gripper_consistency,
        "joint_mae_std_units": joint_mae_std_units,
    }


@app.get("/health")
async def health() -> dict:
    idm = _state["idm"]
    return {
        "status": "ok",
        "arch": idm["config"].get("arch", "v1"),
        "checkpoint_epoch": idm["epoch"],
        "val_metrics": idm["val_metrics"],
        "camera_order": idm["cameras"],
        "panel_layout_version": idm["config"].get("panel_layout_version", PANEL_LAYOUT_VERSION),
    }


@app.post("/score_consistency")
async def score_endpoint(
    dream: UploadFile = File(...), action: UploadFile = File(...)
) -> dict:
    try:
        dream_array = np.load(io.BytesIO(await dream.read()), allow_pickle=False)
        action_array = np.load(io.BytesIO(await action.read()), allow_pickle=False)
        return score_consistency(_state["idm"], dream_array, action_array)
    except (TypeError, ValueError, IndexError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/workspace/wise_idm/checkpoints/best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8101)
    args = parser.parse_args()

    print(f"[idm-server] loading {args.checkpoint} on {args.device} ...")
    _state["idm"] = load_idm(args.checkpoint, args.device)
    print(
        f"[idm-server] loaded: epoch={_state['idm']['epoch']} "
        f"val_metrics={_state['idm']['val_metrics']}"
    )
    print(f"[idm-server] ready on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
