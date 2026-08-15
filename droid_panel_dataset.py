"""DROID windows assembled into the single-panel composite layout.

Experimental counterpart to ``droid_dataset.DroidIDMDataset`` for the
composite-panel architecture (`model_composite.py`). Does not modify
``droid_dataset.py``: it imports and reuses every low-level piece (episode
metadata loading, action-row loading, window enumeration, the sparse
per-shard video cache and its ``VideoFrameReader``) unchanged, and only adds
the panel-assembly step in ``__getitem__``.

Panel size: 272 x 320 (wrist 176 x 320 on top, exteriors 96 x 160 each on the
bottom), chosen so both internal seams land exactly on the shared ResNet's
stride-16 feature grid (176/16=11, 160/16=10 - no feature-map row or column
straddles two cameras) at close to the frozen per-view architecture's total
pixel budget (87,040 vs 86,016). Native full-resolution (640 x 528, matching
the real Cosmos decode exactly) measured ~3.8x slower in practice - see
RUN_NOTES.md.

Resize method: plain stretch (``vision.resize_rgb_stretch``), not the
aspect-preserving letterbox the frozen per-view architecture uses. This was
verified two ways, not assumed: (1) generating one real Cosmos3-Edge dream
and checking pixel values at both panel seams found real content edge to
edge, never a black band; (2) the actual client code that builds this exact
panel (``cosmos_framework...action_policy_server_robolab._compose_roboarena_views``)
resizes each exterior view with one direct ``F.interpolate(..., mode="bilinear")``
call - no aspect preservation, no intermediate stage, no padding. An earlier
version of this file emulated a nonexistent "resize through Cosmos's native
168x320 tile" intermediate step; that stage never exists on the input side -
168 only ever appears as an output-side VAE-decode artifact - so it has been
removed. Wrist and exterior views are each resized in one direct stretch
straight from the raw DROID source to their target tile size, matching the
real client exactly.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from droid_dataset import (
    CAMERAS,
    CHUNK_LEN,
    DATASET_SPLITS,
    FPS,
    NUM_FRAMES,
    EpisodeRef,
    VideoFrameReader,
    Window,
    _local_or_download,
    _metadata_files,
    window_starts,
)
WRIST_CAMERA = "wrist_image_left"
EXTERIOR_CAMERAS = ("exterior_image_1_left", "exterior_image_2_left")

# Seam-aligned compute-budget tile sizes actually cached/fed to the network.
WRIST_TILE = (176, 320)  # (height, width); DROID native 640x360 -> 320x176 in one step
EXTERIOR_TILE = (96, 160)  # via the 168x320 intermediate, see module docstring
TILE_SIZE = {WRIST_CAMERA: WRIST_TILE, **{camera: EXTERIOR_TILE for camera in EXTERIOR_CAMERAS}}
PANEL_TARGET_HEIGHT = WRIST_TILE[0] + EXTERIOR_TILE[0]  # 272
PANEL_TARGET_WIDTH = WRIST_TILE[1]  # 320, == 2 * EXTERIOR_TILE[1]

# Distinct from vision.PANEL_LAYOUT_VERSION, which describes the *native*
# 528x640 Cosmos decode geometry - this tags the downscaled 272x320 composite
# contract instead, so a checkpoint can never be silently reloaded/compared
# against the wrong geometry.
PANEL_LAYOUT_VERSION = "composite_seam_aligned_272x320_v1"


class DroidIDMPanelDataset(Dataset):
    def __init__(
        self,
        *,
        episodes: Sequence[EpisodeRef],
        num_frames: int = NUM_FRAMES,
        chunk_len: int = CHUNK_LEN,
        stride: int,
        end_align_tail: bool = True,
    ):
        if num_frames != chunk_len + 1:
            raise ValueError("num_frames must equal chunk_len + 1; temporal subsampling is not allowed")
        if stride <= 0:
            raise ValueError("stride must be positive")
        refs = list(episodes)
        if not refs:
            raise ValueError("at least one episode is required")
        if len({ref.key for ref in refs}) != len(refs):
            raise ValueError("episode list contains duplicate split/index keys")

        self.episodes = refs
        self.num_frames = num_frames
        self.chunk_len = chunk_len
        self.stride = stride
        self.cameras = tuple(CAMERAS)  # for selection.window_audit compatibility
        self.readers = {camera: VideoFrameReader(*TILE_SIZE[camera]) for camera in CAMERAS}

        self.meta: dict[tuple[str, int], pd.Series] = {}
        metadata_columns = [
            "episode_index",
            "episode_id",
            "length",
            "data/chunk_index",
            "data/file_index",
            "dataset_from_index",
            "dataset_to_index",
        ]
        for camera in CAMERAS:
            metadata_columns.extend(
                [
                    f"videos/observation.image.{camera}/from_timestamp",
                    f"videos/observation.image.{camera}/to_timestamp",
                    f"videos/observation.image.{camera}/chunk_index",
                    f"videos/observation.image.{camera}/file_index",
                ]
            )
        for split in DATASET_SPLITS:
            requested = {ref.episode_index for ref in refs if ref.dataset_split == split}
            if not requested:
                continue
            metadata = pd.concat(
                [pd.read_parquet(path, columns=metadata_columns) for path in _metadata_files(split)],
                ignore_index=True,
            )
            selected = metadata[metadata["episode_index"].isin(requested)].set_index("episode_index")
            missing = requested - set(selected.index.astype(int))
            if missing:
                raise KeyError(f"{split} metadata is missing episode indices: {sorted(missing)[:20]}")
            for episode_index in requested:
                self.meta[(split, episode_index)] = selected.loc[episode_index]

        self.episode_actions: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        for split in DATASET_SPLITS:
            split_refs = [ref for ref in refs if ref.dataset_split == split]
            if not split_refs:
                continue
            needed_files = {
                (int(self.meta[ref.key]["data/chunk_index"]), int(self.meta[ref.key]["data/file_index"]))
                for ref in split_refs
            }
            episode_indices_by_file: dict[tuple[int, int], set[int]] = {}
            for ref in split_refs:
                file_key = (
                    int(self.meta[ref.key]["data/chunk_index"]),
                    int(self.meta[ref.key]["data/file_index"]),
                )
                episode_indices_by_file.setdefault(file_key, set()).add(ref.episode_index)
            for chunk, file_index in sorted(needed_files):
                relative = f"data/chunk-{chunk:03d}/file-{file_index:03d}.parquet"
                data = pd.read_parquet(
                    _local_or_download(split, relative),
                    columns=[
                        "episode_index",
                        "frame_index",
                        "action.joint_position",
                        "action.gripper_position",
                    ],
                )
                wanted = episode_indices_by_file[(chunk, file_index)]
                data = data[data["episode_index"].isin(wanted)]
                for episode_index, rows in data.groupby("episode_index"):
                    key = split, int(episode_index)
                    rows = rows.sort_values("frame_index").reset_index(drop=True)
                    joints = np.stack(rows["action.joint_position"].to_numpy()).astype(np.float32)
                    gripper = rows["action.gripper_position"].to_numpy().astype(np.float32)
                    self.episode_actions[key] = joints, gripper

        self.video_locator: dict[tuple[str, int], dict[str, tuple[int, int, int]]] = {}
        for ref in refs:
            row = self.meta[ref.key]
            self.video_locator[ref.key] = {
                camera: (
                    int(row[f"videos/observation.image.{camera}/chunk_index"]),
                    int(row[f"videos/observation.image.{camera}/file_index"]),
                    round(float(row[f"videos/observation.image.{camera}/from_timestamp"]) * FPS),
                )
                for camera in CAMERAS
            }

        self.windows = []
        self.zero_window_episodes: list[tuple[str, int]] = []
        for ref in refs:
            length = int(self.meta[ref.key]["length"])
            starts = window_starts(length, num_frames=num_frames, stride=stride, end_align_tail=end_align_tail)
            if not starts:
                self.zero_window_episodes.append(ref.key)
                continue
            self.windows.extend(Window(ref.dataset_split, ref.episode_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        window = self.windows[index]
        dataset_split, episode_index, chunk_start = window.dataset_split, window.episode_index, window.chunk_start
        key = window.episode_key

        tiles: dict[str, np.ndarray] = {}
        for camera in CAMERAS:
            chunk, file_index, offset = self.video_locator[key][camera]
            tiles[camera] = self.readers[camera].read_frames(
                dataset_split, camera, chunk, file_index, chunk_start + offset, self.num_frames
            )

        bottom = np.concatenate([tiles[camera] for camera in EXTERIOR_CAMERAS], axis=2)  # (T, 96, 320, 3)
        panel = np.concatenate([tiles[WRIST_CAMERA], bottom], axis=1)  # (T, 272, 320, 3)
        if panel.shape[1:3] != (PANEL_TARGET_HEIGHT, PANEL_TARGET_WIDTH):
            raise ValueError(
                f"assembled panel has shape {panel.shape}; "
                f"expected (T, {PANEL_TARGET_HEIGHT}, {PANEL_TARGET_WIDTH}, 3)"
            )

        joints, gripper = self.episode_actions[key]
        action_slice = slice(chunk_start, chunk_start + self.chunk_len)
        binary_gripper = (gripper[action_slice] > 0.5).astype(np.float32)
        action = np.concatenate((joints[action_slice], binary_gripper[:, None]), axis=-1)

        return {
            "panel": torch.from_numpy(panel),
            "action": torch.from_numpy(action.astype(np.float32, copy=False)),
            "dataset_split": dataset_split,
            "episode_index": episode_index,
            "chunk_start": chunk_start,
        }
