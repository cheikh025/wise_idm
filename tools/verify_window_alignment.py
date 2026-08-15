"""End-to-end check that a dataset window really is the frames/actions it claims.

For randomly chosen windows this re-derives everything from the pinned source
instead of from the cache: it decodes the exact frames out of the source mp4
with ffmpeg, letterboxes them with the canonical implementation, and re-reads
the action rows straight from the source Parquet. Then it compares against what
``DroidIDMDataset`` hands the model.

Checks performed per window:
  1. pixels are bit-identical to a fresh decode of the source video;
  2. frame ``s`` of the window is in-file frame ``s + round(from_timestamp*15)``;
  3. action row ``t`` is the source action at episode frame ``chunk_start + t``,
     i.e. the command across the visual transition frame[t] -> frame[t+1];
  4. the gripper target is the source gripper thresholded at > 0.5.
"""
from __future__ import annotations

import argparse
import random
import subprocess

import numpy as np
import pandas as pd

from droid_dataset import (
    CAMERAS,
    FPS,
    DroidIDMDataset,
    _local_or_download,
    load_episode_manifest,
)
from vision import letterbox_rgb


def decode_frames(video_path: str, start: int, count: int) -> np.ndarray:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", video_path],
        capture_output=True, text=True, check=True,
    )
    width, height = (int(value) for value in probe.stdout.strip().split(","))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video_path,
         "-vf", f"select='between(n\\,{start}\\,{start + count - 1})'",
         "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    frame_bytes = height * width * 3
    if len(raw) != frame_bytes * count:
        raise RuntimeError(f"expected {count} frames, decoded {len(raw) / frame_bytes}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(count, height, width, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    refs = load_episode_manifest(args.manifest)
    dataset = DroidIDMDataset(episodes=refs, stride=args.stride)
    print(f"{len(dataset.episodes)} episodes, {len(dataset)} windows")

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), min(args.windows, len(dataset)))
    failures = 0
    for index in indices:
        window = dataset.windows[index]
        key = window.episode_key
        sample = dataset[index]
        row = dataset.meta[key]
        print(f"\nwindow {index}: {key} chunk_start={window.chunk_start}")

        for camera, name in zip(CAMERAS, ("wrist", "left", "right")):
            chunk, file_index, offset = dataset.video_locator[key][camera]
            relative = (
                f"videos/observation.image.{camera}/"
                f"chunk-{chunk:03d}/file-{file_index:03d}.mp4"
            )
            path = _local_or_download(window.dataset_split, relative)
            expected_offset = round(
                float(row[f"videos/observation.image.{camera}/from_timestamp"]) * FPS
            )
            assert offset == expected_offset, f"{camera}: offset {offset} != {expected_offset}"
            source = decode_frames(path, window.chunk_start + offset, dataset.num_frames)
            expected = letterbox_rgb(source, dataset.input_height, dataset.input_width)
            actual = sample[name].numpy()
            if actual.shape != expected.shape or not np.array_equal(actual, expected):
                failures += 1
                difference = np.abs(actual.astype(int) - expected.astype(int))
                print(f"  {camera}: MISMATCH shapes {actual.shape} vs {expected.shape} "
                      f"max|diff|={difference.max() if actual.shape == expected.shape else 'n/a'}")
            else:
                print(f"  {camera}: pixels bit-identical to a fresh decode {actual.shape}")

        data_relative = (
            f"data/chunk-{int(row['data/chunk_index']):03d}/"
            f"file-{int(row['data/file_index']):03d}.parquet"
        )
        frame = pd.read_parquet(
            _local_or_download(window.dataset_split, data_relative),
            columns=["episode_index", "frame_index", "action.joint_position", "action.gripper_position"],
        )
        frame = frame[frame["episode_index"] == window.episode_index].sort_values("frame_index")
        joints = np.stack(frame["action.joint_position"].to_numpy()).astype(np.float32)
        gripper = frame["action.gripper_position"].to_numpy().astype(np.float32)
        window_slice = slice(window.chunk_start, window.chunk_start + dataset.chunk_len)
        expected_action = np.concatenate(
            (joints[window_slice], (gripper[window_slice] > 0.5).astype(np.float32)[:, None]),
            axis=-1,
        )
        actual_action = sample["action"].numpy()
        if np.array_equal(actual_action, expected_action):
            print(f"  actions: exact match, rows {window.chunk_start}"
                  f"..{window.chunk_start + dataset.chunk_len - 1} of the source parquet")
        else:
            failures += 1
            print(f"  actions: MISMATCH max|diff|={np.abs(actual_action - expected_action).max()}")

    print("\nALIGNMENT OK" if not failures else f"\n{failures} MISMATCHES")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
