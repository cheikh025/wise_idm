"""DROID dataset for the WISE IDM (M4).

Loads from the LeRobotDataset v3.0 layout of nvidia/Cosmos3-DROID (success
split, data/chunk-000/file-000.parquet + however many per-camera video files
the requested episode range spans -- see preprocess_videos.py).

Hard constraints (per project direction):
- commanded targets come from action.joint_position [7] + action.gripper_position [1],
  never action.joint_velocity / action.cartesian_position / action.cartesian_velocity;
- no VAE roundtrip anywhere -- frames are read as raw RGB uint8 and fed to a
  plain CNN backbone directly, never encoded/decoded through any VAE.

Each sample is one "window": an episode's action chunk of length
CHUNK_LEN (32, matching the verified Cosmos3-Edge chunk size from M1), with
NUM_FRAMES video frames spanning [chunk_start, chunk_start+CHUNK_LEN]
(inclusive of the initial frame, subsampled via linspace -- same
first/last-preserving subsampling convention used by Robometer in M2/M3)
for each of the 3 views, plus the proprioceptive state at chunk_start.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

BASE = "/workspace/.hf_home/hub/datasets--nvidia--Cosmos3-DROID/snapshots/5c11a20accb11497270a5247a7f1e66ad04c956c/success"
CACHE_DIR = "/workspace/wise_idm/cache"
CAMERAS = ["wrist_image_left", "exterior_image_1_left", "exterior_image_2_left"]
CHUNK_LEN = 32          # verified Cosmos3-Edge action chunk size (M1 RUNBOOK)
NUM_FRAMES = CHUNK_LEN + 1  # full temporal density, zero subsampling: every frame
                            # chunk_start..chunk_start+CHUNK_LEN (33 frames -> 32
                            # adjacent pairs), matching Cosmos's own 33-frame dream
                            # length exactly. Compute is controlled via spatial
                            # token compression in the CNN backbone instead
                            # (IDM_DESIGN.md's stated preference order).
FPS = 15.0


@dataclass
class Window:
    episode_index: int
    chunk_start: int  # frame_index within the episode


def _video_path(camera: str, chunk: int, file: int) -> str:
    return os.path.join(BASE, "videos", f"observation.image.{camera}", f"chunk-{chunk:03d}", f"file-{file:03d}.mp4")


class VideoFrameReader:
    """Reads pre-decoded, pre-resized frames from preprocess_videos.py's
    memmapped .npy caches, one per (camera, chunk, file) -- see that script's
    docstring for why: opencv-python-headless's bundled ffmpeg fails to
    software-decode these AV1 videos, and repeated ffmpeg-CLI decodes-from-
    start-of-file per window is far too slow for training (~46s/sample).
    Decode once per video file, memmap, slice.
    """

    def __init__(self):
        self._arrays: dict[tuple, np.ndarray] = {}

    def _get(self, camera: str, chunk: int, file: int) -> np.ndarray:
        key = (camera, chunk, file)
        if key not in self._arrays:
            path = os.path.join(CACHE_DIR, f"{camera}_chunk{chunk:03d}_file{file:03d}.npy")
            self._arrays[key] = np.load(path, mmap_mode="r")
        return self._arrays[key]

    def read_frames(self, camera: str, chunk: int, file: int, frame_indices: list[int]) -> np.ndarray:
        arr = self._get(camera, chunk, file)
        return np.stack([arr[i] for i in frame_indices])

    def close(self):
        self._arrays.clear()


class DroidIDMDataset(Dataset):
    def __init__(self, episode_indices: list[int], image_size: int = 128, num_frames: int = NUM_FRAMES,
                 chunk_len: int = CHUNK_LEN, stride: int | None = None):
        self.image_size = image_size
        self.num_frames = num_frames
        self.chunk_len = chunk_len
        self.reader = VideoFrameReader()

        meta = pd.read_parquet(f"{BASE}/meta/episodes/chunk-000/file-000.parquet")
        self.meta = meta[meta["episode_index"].isin(episode_indices)].set_index("episode_index")

        # Episodes beyond ~1216 span multiple DATA parquet files (independent
        # chunking from video files -- data_files_size_in_mb=100 targets ~100MB
        # per file, video_files_size_in_mb=200 targets ~200MB, so they don't
        # align). Load exactly the data files this episode range touches.
        needed_data_files = self.meta[["data/chunk_index", "data/file_index"]].drop_duplicates()
        data_parts = []
        for _, (chunk, file) in needed_data_files.iterrows():
            chunk, file = int(chunk), int(file)
            path = f"{BASE}/data/chunk-{chunk:03d}/file-{file:03d}.parquet"
            if not os.path.exists(path):
                from huggingface_hub import hf_hub_download
                hf_hub_download("nvidia/Cosmos3-DROID",
                                 f"success/data/chunk-{chunk:03d}/file-{file:03d}.parquet", repo_type="dataset")
                path = f"{BASE}/data/chunk-{chunk:03d}/file-{file:03d}.parquet"
            data_parts.append(pd.read_parquet(path))
        data = pd.concat(data_parts, ignore_index=True)
        self.data = data[data["episode_index"].isin(episode_indices)].reset_index(drop=True)

        # Per-episode video frame offset: cv2 frame index within the episode's
        # video FILE, not within the whole episode's own timeline. from_timestamp
        # (seconds, within that video file) * fps = starting frame in the file.
        self.video_frame_offset = {}
        for ep in episode_indices:
            row = self.meta.loc[ep]
            offsets = {}
            for cam in CAMERAS:
                from_ts = row[f"videos/observation.image.{cam}/from_timestamp"]
                offsets[cam] = round(from_ts * FPS)
            self.video_frame_offset[ep] = offsets

        stride = stride or chunk_len
        self.windows: list[Window] = []
        for ep in episode_indices:
            length = int(self.meta.loc[ep]["length"])
            start = 0
            while start + chunk_len < length:
                self.windows.append(Window(ep, start))
                start += stride

    def __len__(self) -> int:
        return len(self.windows)

    def _resize(self, frames: np.ndarray) -> np.ndarray:
        out = np.stack([cv2.resize(f, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
                         for f in frames])
        return out

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]
        ep_df = self.data[self.data["episode_index"] == w.episode_index].sort_values("frame_index").reset_index(drop=True)

        frame_ids_in_episode = np.linspace(w.chunk_start, w.chunk_start + self.chunk_len, self.num_frames)
        frame_ids_in_episode = np.round(frame_ids_in_episode).astype(int)

        views = {}
        row = self.meta.loc[w.episode_index]
        for cam in CAMERAS:
            offset = self.video_frame_offset[w.episode_index][cam]
            file_frame_ids = (frame_ids_in_episode + offset).tolist()
            chunk = int(row[f"videos/observation.image.{cam}/chunk_index"])
            file = int(row[f"videos/observation.image.{cam}/file_index"])
            frames = self.reader.read_frames(cam, chunk, file, file_frame_ids)  # (T, IMAGE_SIZE, IMAGE_SIZE, 3) uint8, pre-resized
            if frames.shape[1] != self.image_size:
                frames = self._resize(frames)
            views[cam] = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0  # (T,3,H,W)

        chunk_rows = ep_df.iloc[w.chunk_start:w.chunk_start + self.chunk_len]
        joint_pos = np.stack(chunk_rows["action.joint_position"].to_numpy())          # (chunk_len, 7)
        gripper = chunk_rows["action.gripper_position"].to_numpy().astype(np.float32)  # (chunk_len,)
        action = np.concatenate([joint_pos, gripper[:, None]], axis=-1)               # (chunk_len, 8)

        init_row = ep_df.iloc[w.chunk_start]
        proprio = np.concatenate([
            np.asarray(init_row["observation.state.joint_positions"], dtype=np.float32),
            np.asarray([init_row["observation.state.gripper_position"]], dtype=np.float32),
        ])  # (8,)

        return {
            "wrist": views["wrist_image_left"],
            "left": views["exterior_image_1_left"],
            "right": views["exterior_image_2_left"],
            "proprio": torch.from_numpy(proprio),
            "action": torch.from_numpy(action.astype(np.float32)),
            "episode_index": w.episode_index,
            "chunk_start": w.chunk_start,
        }
