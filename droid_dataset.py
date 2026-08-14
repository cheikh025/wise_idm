"""DROID windows for the three-view, vision-only WISE inverse dynamics model.

Every sample contains 33 consecutive RGB frames from each synchronized camera
and the 32 absolute executed actions aligned to the adjacent-frame transitions.
Success and failure episode indices live in separate namespaces, so episode
identity is always the pair ``(dataset_split, episode_index)``.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from vision import (
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DROID_CAMERA_ORDER,
    VISION_PREPROCESS_VERSION,
)


HF_REPO = "nvidia/Cosmos3-DROID"
HF_REVISION = "5c11a20accb11497270a5247a7f1e66ad04c956c"
SNAPSHOT_ROOT = Path(
    os.environ.get(
        "COSMOS3_DROID_ROOT",
        "/workspace/.hf_home/hub/datasets--nvidia--Cosmos3-DROID/"
        f"snapshots/{HF_REVISION}",
    )
)
CACHE_DIR = Path(os.environ.get("WISE_IDM_CACHE_DIR", "/workspace/wise_idm/cache"))
DATASET_SPLITS = ("success", "failure")
CAMERAS = list(DROID_CAMERA_ORDER)
CHUNK_LEN = 32
NUM_FRAMES = CHUNK_LEN + 1
FPS = 15.0


@dataclass(frozen=True)
class EpisodeRef:
    dataset_split: str
    episode_index: int
    scene_id: str | None = None
    lab: str | None = None
    episode_id: str | None = None
    length: int | None = None

    @property
    def key(self) -> tuple[str, int]:
        return self.dataset_split, self.episode_index


@dataclass(frozen=True)
class Window:
    dataset_split: str
    episode_index: int
    chunk_start: int

    @property
    def episode_key(self) -> tuple[str, int]:
        return self.dataset_split, self.episode_index


def load_episode_manifest(path: str | os.PathLike, *, require_scene_id: bool = False) -> list[EpisodeRef]:
    manifest = pd.read_csv(path)
    required = {"dataset_split", "episode_index"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest {path} is missing columns: {sorted(missing)}")
    if manifest.empty:
        raise ValueError(f"manifest {path} is empty")

    manifest["dataset_split"] = manifest["dataset_split"].astype(str).str.strip()
    invalid_splits = sorted(set(manifest["dataset_split"]) - set(DATASET_SPLITS))
    if invalid_splits:
        raise ValueError(f"manifest {path} has invalid dataset_split values: {invalid_splits}")
    duplicated = manifest.duplicated(["dataset_split", "episode_index"], keep=False)
    if duplicated.any():
        duplicate_rows = manifest.loc[duplicated, ["dataset_split", "episode_index"]]
        raise ValueError(f"manifest {path} contains duplicate episodes:\n{duplicate_rows.head()}")

    refs = []
    for _, row in manifest.iterrows():
        scene_id = row.get("scene_key")
        if pd.isna(scene_id) or not str(scene_id).strip():
            scene_id = None
        if scene_id is None and {"lab", "building", "scene_id"}.issubset(manifest.columns):
            values = (row["lab"], row["building"], row["scene_id"])
            if not any(pd.isna(value) for value in values):
                scene_id = "|".join(str(value).strip() for value in values)
        if scene_id is None and "scene_id" in manifest.columns and not pd.isna(row["scene_id"]):
            scene_id = str(row["scene_id"]).strip() or None
        elif scene_id is not None:
            scene_id = str(scene_id).strip() or None
        if require_scene_id and scene_id is None:
            raise ValueError(f"manifest {path} contains an empty scene identity")

        lab = row.get("lab")
        lab = None if pd.isna(lab) else str(lab).strip() or None
        episode_id = row.get("episode_id")
        episode_id = None if pd.isna(episode_id) else str(episode_id).strip() or None
        length = row.get("length")
        length = None if pd.isna(length) else int(length)
        refs.append(
            EpisodeRef(
                str(row["dataset_split"]),
                int(row["episode_index"]),
                scene_id,
                lab,
                episode_id,
                length,
            )
        )
    return refs


def assert_scene_disjoint(train_refs: Sequence[EpisodeRef], val_refs: Sequence[EpisodeRef]) -> None:
    key_overlap = {ref.key for ref in train_refs} & {ref.key for ref in val_refs}
    if key_overlap:
        raise ValueError(
            f"train/validation episode overlap detected ({len(key_overlap)} episodes): "
            f"{sorted(key_overlap)[:10]}"
        )
    if any(ref.scene_id is None for ref in (*train_refs, *val_refs)):
        raise ValueError("train and validation manifests need a non-empty scene_id for every episode")
    overlap = {ref.scene_id for ref in train_refs} & {ref.scene_id for ref in val_refs}
    if overlap:
        preview = sorted(overlap)[:10]
        raise ValueError(f"train/validation scene overlap detected ({len(overlap)} scenes): {preview}")


def cache_path(
    dataset_split: str,
    camera: str,
    chunk: int,
    file_index: int,
    input_height: int,
    input_width: int,
) -> Path:
    return CACHE_DIR / (
        f"{dataset_split}_{camera}_chunk{chunk:03d}_file{file_index:03d}_"
        f"{input_width}x{input_height}_{VISION_PREPROCESS_VERSION}.npy"
    )


def window_starts(
    length: int,
    *,
    num_frames: int = NUM_FRAMES,
    stride: int,
    end_align_tail: bool = True,
) -> list[int]:
    """Return regular window starts plus the final valid end-aligned start."""
    if length < 0 or num_frames <= 0 or stride <= 0:
        raise ValueError("length must be non-negative and num_frames/stride must be positive")
    last_start = length - num_frames
    if last_start < 0:
        return []
    starts = list(range(0, last_start + 1, stride))
    if end_align_tail and starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _local_or_download(dataset_split: str, relative_path: str) -> str:
    local = SNAPSHOT_ROOT / dataset_split / relative_path
    if local.exists():
        return str(local)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        HF_REPO,
        f"{dataset_split}/{relative_path}",
        repo_type="dataset",
        revision=HF_REVISION,
    )


def _metadata_files(dataset_split: str) -> list[str]:
    local_dir = SNAPSHOT_ROOT / dataset_split / "meta" / "episodes"
    local_files = sorted(local_dir.rglob("*.parquet"))
    if local_files:
        return [str(path) for path in local_files]

    from huggingface_hub import list_repo_files

    prefix = f"{dataset_split}/meta/episodes/"
    remote = sorted(
        path for path in list_repo_files(HF_REPO, repo_type="dataset", revision=HF_REVISION)
        if path.startswith(prefix) and path.endswith(".parquet")
    )
    if not remote:
        raise FileNotFoundError(f"no episode metadata found for {dataset_split}")
    return [_local_or_download(dataset_split, path.removeprefix(f"{dataset_split}/")) for path in remote]


class VideoFrameReader:
    def __init__(self, input_height: int, input_width: int, max_open_files: int = 64):
        self.input_height = input_height
        self.input_width = input_width
        self.max_open_files = max_open_files
        self._arrays: OrderedDict[tuple, np.ndarray] = OrderedDict()

    def _get(self, dataset_split: str, camera: str, chunk: int, file_index: int) -> np.ndarray:
        key = (dataset_split, camera, chunk, file_index)
        if key in self._arrays:
            self._arrays.move_to_end(key)
        else:
            path = cache_path(
                dataset_split,
                camera,
                chunk,
                file_index,
                self.input_height,
                self.input_width,
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"missing rectangular video cache {path}; run preprocess_videos.py for this manifest"
                )
            array = np.load(path, mmap_mode="r")
            expected_tail = (self.input_height, self.input_width, 3)
            if array.ndim != 4 or tuple(array.shape[1:]) != expected_tail:
                raise ValueError(
                    f"cache {path} has shape {array.shape}; expected (N, {expected_tail}). "
                    "Old square caches cannot be reused."
                )
            self._arrays[key] = array
            while len(self._arrays) > self.max_open_files:
                _, oldest = self._arrays.popitem(last=False)
                mmap = getattr(oldest, "_mmap", None)
                if mmap is not None:
                    mmap.close()
        return self._arrays[key]

    def read_frames(
        self,
        dataset_split: str,
        camera: str,
        chunk: int,
        file_index: int,
        frame_indices: Sequence[int],
    ) -> np.ndarray:
        array = self._get(dataset_split, camera, chunk, file_index)
        if not frame_indices:
            raise ValueError("frame_indices cannot be empty")
        if min(frame_indices) < 0 or max(frame_indices) >= len(array):
            raise IndexError(
                f"requested frames [{min(frame_indices)}, {max(frame_indices)}] "
                f"from cache with {len(array)} frames"
            )
        return np.stack([array[index] for index in frame_indices])

    def close(self) -> None:
        for array in self._arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._arrays.clear()

    def __del__(self) -> None:
        if hasattr(self, "_arrays"):
            try:
                self.close()
            except Exception:
                pass


class DroidIDMDataset(Dataset):
    def __init__(
        self,
        episode_indices: Sequence[int] | None = None,
        *,
        episodes: Sequence[EpisodeRef] | None = None,
        manifest_path: str | os.PathLike | None = None,
        dataset_split: str = "success",
        input_height: int = DEFAULT_INPUT_HEIGHT,
        input_width: int = DEFAULT_INPUT_WIDTH,
        image_size: int | None = None,
        num_frames: int = NUM_FRAMES,
        chunk_len: int = CHUNK_LEN,
        stride: int = CHUNK_LEN,
        end_align_tail: bool = True,
    ):
        sources = sum(value is not None for value in (episode_indices, episodes, manifest_path))
        if sources != 1:
            raise ValueError("provide exactly one of episode_indices, episodes, or manifest_path")
        if image_size is not None:
            input_height = input_width = image_size
        if num_frames != chunk_len + 1:
            raise ValueError("num_frames must equal chunk_len + 1; temporal subsampling is not allowed")
        if stride <= 0:
            raise ValueError("stride must be positive")
        if input_height <= 0 or input_width <= 0:
            raise ValueError("input dimensions must be positive")
        if dataset_split not in DATASET_SPLITS:
            raise ValueError(f"invalid dataset_split: {dataset_split}")

        if manifest_path is not None:
            refs = load_episode_manifest(manifest_path)
        elif episodes is not None:
            refs = list(episodes)
        else:
            refs = [EpisodeRef(dataset_split, int(index)) for index in episode_indices or ()]
        if not refs:
            raise ValueError("at least one episode is required")
        if len({ref.key for ref in refs}) != len(refs):
            raise ValueError("episode list contains duplicate split/index keys")

        self.episodes = refs
        self.input_height = input_height
        self.input_width = input_width
        self.image_size = input_height if input_height == input_width else None
        self.num_frames = num_frames
        self.chunk_len = chunk_len
        self.stride = stride
        self.cameras = tuple(CAMERAS)
        self.reader = VideoFrameReader(input_height, input_width)

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
            if selected.index.has_duplicates:
                raise ValueError(f"{split} episode metadata contains duplicate episode indices")
            missing = requested - set(selected.index.astype(int))
            if missing:
                raise KeyError(f"{split} metadata is missing episode indices: {sorted(missing)[:20]}")
            refs_by_index = {ref.episode_index: ref for ref in refs if ref.dataset_split == split}
            for episode_index in requested:
                row = selected.loc[episode_index]
                ref = refs_by_index[episode_index]
                length = int(row["length"])
                dataset_from = int(row["dataset_from_index"])
                dataset_to = int(row["dataset_to_index"])
                if dataset_to - dataset_from != length:
                    raise ValueError(
                        f"{ref.key} has inconsistent data range [{dataset_from}, {dataset_to}) "
                        f"for length {length}"
                    )
                official_episode_id = str(row["episode_id"])
                parts = official_episode_id.split("/")
                if len(parts) != 4 or any(not part for part in parts) or parts[1] != split:
                    raise ValueError(
                        f"{ref.key} has malformed or split-inconsistent episode_id "
                        f"{official_episode_id!r}"
                    )
                if ref.lab is not None and parts[0] != ref.lab:
                    raise ValueError(
                        f"{ref.key} manifest lab {ref.lab!r} disagrees with official "
                        f"episode_id {official_episode_id!r}"
                    )
                if ref.episode_id is not None and official_episode_id != ref.episode_id:
                    raise ValueError(f"{ref.key} manifest episode_id disagrees with official metadata")
                if ref.length is not None and length != ref.length:
                    raise ValueError(f"{ref.key} manifest length disagrees with official metadata")
                for camera in CAMERAS:
                    prefix = f"videos/observation.image.{camera}"
                    from_timestamp = float(row[f"{prefix}/from_timestamp"])
                    to_timestamp = float(row[f"{prefix}/to_timestamp"])
                    from_frame = round(from_timestamp * FPS)
                    to_frame = round(to_timestamp * FPS)
                    if (
                        not np.isfinite(from_timestamp)
                        or not np.isfinite(to_timestamp)
                        or from_timestamp < 0
                        or to_timestamp <= from_timestamp
                        or to_frame - from_frame != length
                    ):
                        raise ValueError(
                            f"{ref.key}/{camera} has timestamp range "
                            f"[{from_timestamp}, {to_timestamp}) inconsistent with length {length}"
                        )
                self.meta[(split, episode_index)] = row

        self.episode_actions: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        for split in DATASET_SPLITS:
            split_refs = [ref for ref in refs if ref.dataset_split == split]
            if not split_refs:
                continue
            needed_files = {
                (
                    int(self.meta[ref.key]["data/chunk_index"]),
                    int(self.meta[ref.key]["data/file_index"]),
                )
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
                        "index",
                        "timestamp",
                        "action.joint_position",
                        "action.gripper_position",
                    ],
                )
                wanted = episode_indices_by_file[(chunk, file_index)]
                data = data[data["episode_index"].isin(wanted)]
                for episode_index, rows in data.groupby("episode_index"):
                    key = split, int(episode_index)
                    rows = rows.sort_values("frame_index").reset_index(drop=True)
                    expected_length = int(self.meta[key]["length"])
                    frame_indices = rows["frame_index"].to_numpy()
                    if len(rows) != expected_length or not np.array_equal(
                        frame_indices, np.arange(expected_length)
                    ):
                        raise ValueError(
                            f"{key} has non-contiguous action rows: "
                            f"metadata length={expected_length}, rows={len(rows)}"
                        )
                    dataset_from = int(self.meta[key]["dataset_from_index"])
                    dataset_to = int(self.meta[key]["dataset_to_index"])
                    if not np.array_equal(rows["index"].to_numpy(), np.arange(dataset_from, dataset_to)):
                        raise ValueError(f"{key} has non-contiguous dataset-global frame indices")
                    expected_timestamps = np.arange(expected_length, dtype=np.float64) / FPS
                    if not np.allclose(
                        rows["timestamp"].to_numpy(dtype=np.float64),
                        expected_timestamps,
                        rtol=0.0,
                        atol=1e-4,
                    ):
                        raise ValueError(f"{key} has timestamps inconsistent with frame_index/{FPS}")
                    joints = np.stack(rows["action.joint_position"].to_numpy()).astype(np.float32)
                    gripper = rows["action.gripper_position"].to_numpy().astype(np.float32)
                    if joints.shape != (expected_length, 7):
                        raise ValueError(
                            f"{key} joint actions have shape {joints.shape}, "
                            f"expected {(expected_length, 7)}"
                        )
                    if not np.isfinite(joints).all() or not np.isfinite(gripper).all():
                        raise ValueError(f"{key} contains non-finite action values")
                    if ((gripper < 0.0) | (gripper > 1.0)).any():
                        raise ValueError(f"{key} gripper actions must lie in [0, 1]")
                    self.episode_actions[key] = joints, gripper
            missing = {ref.key for ref in split_refs} - set(self.episode_actions)
            if missing:
                raise KeyError(f"{split} data files are missing episodes: {sorted(missing)[:20]}")

        self.video_frame_offset: dict[tuple[str, int], dict[str, int]] = {}
        for ref in refs:
            row = self.meta[ref.key]
            self.video_frame_offset[ref.key] = {
                camera: round(float(row[f"videos/observation.image.{camera}/from_timestamp"]) * FPS)
                for camera in CAMERAS
            }

        self.windows: list[Window] = []
        self.zero_window_episodes: list[tuple[str, int]] = []
        for ref in refs:
            length = int(self.meta[ref.key]["length"])
            starts = window_starts(
                length,
                num_frames=num_frames,
                stride=stride,
                end_align_tail=end_align_tail,
            )
            if not starts:
                self.zero_window_episodes.append(ref.key)
                continue
            self.windows.extend(Window(ref.dataset_split, ref.episode_index, start) for start in starts)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict:
        window = self.windows[index]
        key = window.episode_key
        frame_ids = np.arange(window.chunk_start, window.chunk_start + self.num_frames)

        views: dict[str, torch.Tensor] = {}
        row = self.meta[key]
        for camera in CAMERAS:
            offset = self.video_frame_offset[key][camera]
            file_frame_ids = (frame_ids + offset).tolist()
            chunk = int(row[f"videos/observation.image.{camera}/chunk_index"])
            file_index = int(row[f"videos/observation.image.{camera}/file_index"])
            frames = self.reader.read_frames(
                window.dataset_split,
                camera,
                chunk,
                file_index,
                file_frame_ids,
            )
            views[camera] = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float().div_(255.0)

        joints, gripper = self.episode_actions[key]
        action_slice = slice(window.chunk_start, window.chunk_start + self.chunk_len)
        binary_gripper = (gripper[action_slice] > 0.5).astype(np.float32)
        action = np.concatenate((joints[action_slice], binary_gripper[:, None]), axis=-1)

        return {
            "wrist": views["wrist_image_left"],
            "left": views["exterior_image_1_left"],
            "right": views["exterior_image_2_left"],
            "action": torch.from_numpy(action.astype(np.float32, copy=False)),
            "dataset_split": window.dataset_split,
            "episode_index": window.episode_index,
            "chunk_start": window.chunk_start,
        }
