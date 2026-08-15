"""Generate exactly one real Cosmos3-Edge-Policy-DROID dream, then exit.

Calls RobolabPolicyService.infer() directly (bypassing the WebSocket server
and RoboLab/Isaac Sim entirely) using one real cached DROID frame + its real
joint/gripper state + its real task instruction as the observation. This is
purely to look at what a genuine Cosmos dream looks like next to our training
composite panel - it plays no role in IDM training and this script does not
import anything from wise_idm.

Run with the cosmos-framework venv:
  /workspace/cosmos-framework/.venv/bin/python tools/get_one_cosmos_dream.py

Writes:
  /workspace/cosmos_dream_sample.npy   - decoded dream, (33, 528, 640, 3) uint8
  /workspace/cosmos_dream_sample.mp4   - same, as a playable video
  prints the generated action chunk shape/summary
"""
from __future__ import annotations

import gc
import json
import sys

import numpy as np
import pandas as pd
import torch

WISE_IDM_ROOT = "/workspace/wise_idm"
sys.path.insert(0, WISE_IDM_ROOT)


def _find_episode_with_instruction() -> tuple[pd.Series, str]:
    """Pick a manifest row whose episode has a real, non-empty task instruction."""
    import glob

    manifest = pd.read_csv(f"{WISE_IDM_ROOT}/manifests/train_21k.csv")
    success_rows = manifest[manifest["dataset_split"] == "success"].reset_index(drop=True)
    meta_path = sorted(
        glob.glob(
            "/workspace/.hf_home/hub/datasets--nvidia--Cosmos3-DROID/snapshots/"
            "5c11a20accb11497270a5247a7f1e66ad04c956c/success/meta/episodes/*/*.parquet"
        )
    )
    tasks_meta = pd.concat(
        [pd.read_parquet(p, columns=["episode_index", "tasks"]) for p in meta_path], ignore_index=True
    )
    tasks_by_index = dict(zip(tasks_meta["episode_index"], tasks_meta["tasks"]))
    for _, row in success_rows.iterrows():
        tasks = tasks_by_index.get(int(row["episode_index"]))
        if tasks is None or len(tasks) == 0:
            continue
        instruction = str(tasks[0]).split("|")[0].strip()
        if instruction:
            return row, instruction
    raise RuntimeError("no episode with a non-empty instruction found in the first success rows")


def load_one_real_observation() -> dict:
    """Pull one real DROID frame + state + instruction from the pinned manifests."""
    from droid_dataset import FPS, _local_or_download

    row, instruction = _find_episode_with_instruction()
    dataset_split = str(row["dataset_split"])
    episode_index = int(row["episode_index"])
    frame_index = min(80, int(row["length"]) - 1)

    images = {}
    for camera, key in (
        ("wrist_image_left", "observation/wrist_image_left"),
        ("exterior_image_1_left", "observation/exterior_image_1_left"),
        ("exterior_image_2_left", "observation/exterior_image_2_left"),
    ):
        prefix = f"videos/observation.image.{camera}"
        chunk, file_index = int(row[f"{prefix}/chunk_index"]), int(row[f"{prefix}/file_index"])
        offset = round(float(row[f"{prefix}/from_timestamp"]) * FPS)
        relative = f"{prefix}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
        path = _local_or_download(dataset_split, relative)
        import subprocess

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "csv=p=0", path],
            capture_output=True, text=True, check=True,
        )
        width, height = (int(x) for x in probe.stdout.strip().split(","))
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path,
             "-vf", f"select='eq(n\\,{offset + frame_index})'", "-vsync", "0", "-vframes", "1",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, check=True,
        ).stdout
        images[key] = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)

    data_relative = f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
    action_frame = pd.read_parquet(
        _local_or_download(dataset_split, data_relative),
        columns=["episode_index", "frame_index", "action.joint_position", "action.gripper_position"],
    )
    action_frame = action_frame[action_frame["episode_index"] == episode_index].sort_values("frame_index")
    joints = np.stack(action_frame["action.joint_position"].to_numpy()).astype(np.float32)
    gripper = action_frame["action.gripper_position"].to_numpy().astype(np.float32)

    obs = {
        "prompt": instruction,
        "observation/joint_position": joints[frame_index : frame_index + 1],
        "observation/gripper_position": gripper[frame_index : frame_index + 1],
        **images,
    }
    print(f"episode ({dataset_split}, {episode_index}), frame {frame_index}, instruction={instruction!r}")
    return obs


def main() -> None:
    from cosmos_framework.scripts.action_policy_server_robolab import (
        RobolabPolicyService,
        RobolabServerArgs,
    )

    args = RobolabServerArgs(
        checkpoint_path="nvidia/Cosmos3-Edge-Policy-DROID",
        decode_video=True,
        format_prompt_as_json=True,
    )
    print("loading Cosmos3-Edge-Policy-DROID (first run downloads ~8GB+) ...", flush=True)
    service = RobolabPolicyService(args)
    print("loaded. GPU memory after load:", flush=True)
    print(torch.cuda.memory_summary(abbreviated=True), flush=True)

    obs = load_one_real_observation()
    print("running one real inference call (diffusion denoising) ...", flush=True)
    outputs = service.infer(obs)

    action = outputs["action"]
    print(f"action chunk shape={action.shape} dtype={action.dtype}")
    print("action[0] (first predicted step):", np.round(action[0], 4))

    video = outputs["video"]
    print(f"decoded dream video shape={video.shape} dtype={video.dtype}")
    np.save("/workspace/cosmos_dream_sample.npy", video)

    try:
        import imageio.v3 as iio

        iio.imwrite("/workspace/cosmos_dream_sample.mp4", video, fps=15)
        print("wrote /workspace/cosmos_dream_sample.mp4")
    except Exception as exc:  # noqa: BLE001
        print(f"mp4 write skipped ({exc}); .npy is saved, that's enough to render frames from")

    print("wrote /workspace/cosmos_dream_sample.npy")

    # Free the GPU before returning control - training the IDM needs it clean.
    del service, outputs, video, action
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU memory after teardown: {torch.cuda.memory_allocated() / 2**20:.1f} MiB allocated")


if __name__ == "__main__":
    main()
