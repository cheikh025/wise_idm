"""Decode selected Cosmos3-DROID video shards into rectangular RGB caches.

Every source frame is retained in temporal order. Frames are resized with an
aspect-preserving letterbox to 224x128 by default; old square caches are not
reused because their geometry cannot be recovered.
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from droid_dataset import (
    CAMERAS,
    DATASET_SPLITS,
    FPS,
    EpisodeRef,
    _local_or_download,
    _metadata_files,
    cache_path,
    load_episode_manifest,
)
from vision import DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH
from vision import letterbox_rgb


RESIZE_BATCH_FRAMES = 256


def load_split_metadata(dataset_split: str) -> pd.DataFrame:
    columns = [
        "episode_index",
        "episode_id",
        "length",
        "dataset_from_index",
        "dataset_to_index",
    ]
    for camera in CAMERAS:
        columns.extend(
            [
                f"videos/observation.image.{camera}/chunk_index",
                f"videos/observation.image.{camera}/file_index",
                f"videos/observation.image.{camera}/from_timestamp",
                f"videos/observation.image.{camera}/to_timestamp",
            ]
        )
    return pd.concat(
        [pd.read_parquet(path, columns=columns) for path in _metadata_files(dataset_split)],
        ignore_index=True,
    )


def _read_at_most(stream, requested_bytes: int) -> bytes:
    """Fill a raw-video batch unless the stream reaches EOF."""
    chunks = []
    remaining = requested_bytes
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_resized_video(
    video_path: str,
    output_path: Path,
    frames_needed: int,
    input_height: int,
    input_width: int,
) -> tuple[int, int, int, int]:
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
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    source_width, source_height = (int(value) for value in probe.stdout.strip().split(","))
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        video_path,
        "-vframes",
        str(frames_needed),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]

    temporary_path = Path(str(output_path) + ".tmp")
    output = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.uint8,
        shape=(frames_needed, input_height, input_width, 3),
    )
    frame_bytes = source_height * source_width * 3
    decoded = 0
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    try:
        while decoded < frames_needed:
            batch_count = min(RESIZE_BATCH_FRAMES, frames_needed - decoded)
            requested_bytes = batch_count * frame_bytes
            raw = _read_at_most(process.stdout, requested_bytes)
            complete_frames = len(raw) // frame_bytes
            if complete_frames == 0:
                break
            if len(raw) % frame_bytes:
                raise RuntimeError(
                    f"ffmpeg returned a partial raw frame ({len(raw)} bytes, frame size {frame_bytes})"
                )
            frames = np.frombuffer(raw, dtype=np.uint8).reshape(
                complete_frames, source_height, source_width, 3
            )
            output[decoded : decoded + complete_frames] = letterbox_rgb(
                frames, input_height, input_width
            )
            decoded += complete_frames
        if decoded != frames_needed:
            raise RuntimeError(
                f"decoded only {decoded}/{frames_needed} requested frames from {video_path}"
            )
        process.stdout.close()
        assert process.stderr is not None
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, command, stderr=stderr)
        output.flush()
        output._mmap.close()
        output = None
        os.replace(temporary_path, output_path)
    except BaseException:
        process.kill()
        process.wait()
        if output is not None:
            output._mmap.close()
        temporary_path.unlink(missing_ok=True)
        raise
    return frames_needed, input_height, input_width, 3


def selected_episodes(args: argparse.Namespace) -> list[EpisodeRef]:
    if args.manifest:
        return load_episode_manifest(args.manifest)
    if args.episodes_file:
        frame = pd.read_csv(args.episodes_file)
        if "dataset_split" in frame:
            return [
                EpisodeRef(str(row.dataset_split), int(row.episode_index))
                for row in frame.itertuples(index=False)
            ]
        return [EpisodeRef(args.dataset_split, int(value)) for value in frame["episode_index"]]
    if args.num_episodes is not None:
        return [EpisodeRef(args.dataset_split, index) for index in range(args.num_episodes)]
    raise ValueError("pass exactly one of --manifest, --episodes-file, or --num-episodes")


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="CSV with dataset_split and episode_index columns")
    source.add_argument("--episodes-file", help="Legacy CSV with at least episode_index")
    source.add_argument("--num-episodes", type=int, help="Legacy sequential 0..N-1 mode")
    parser.add_argument("--dataset-split", choices=DATASET_SPLITS, default="success")
    parser.add_argument("--input-height", type=int, default=DEFAULT_INPUT_HEIGHT)
    parser.add_argument("--input-width", type=int, default=DEFAULT_INPUT_WIDTH)
    args = parser.parse_args()
    if args.input_height <= 0 or args.input_width <= 0:
        raise ValueError("input dimensions must be positive")

    episodes = selected_episodes(args)
    if len({ref.key for ref in episodes}) != len(episodes):
        raise ValueError("episode selection contains duplicate split/index keys")
    print(f"preprocessing {len(episodes)} episodes at {args.input_width}x{args.input_height}")

    grouped = {
        split: [ref.episode_index for ref in episodes if ref.dataset_split == split]
        for split in DATASET_SPLITS
    }
    for dataset_split, episode_indices in grouped.items():
        if not episode_indices:
            continue
        metadata = load_split_metadata(dataset_split)
        metadata = metadata[metadata["episode_index"].isin(episode_indices)].set_index("episode_index")
        if metadata.index.has_duplicates:
            raise ValueError(f"{dataset_split} metadata contains duplicate episode indices")
        missing = set(episode_indices) - set(metadata.index.astype(int))
        if missing:
            raise KeyError(f"{dataset_split} metadata is missing episodes: {sorted(missing)[:20]}")

        for camera in CAMERAS:
            chunk_col = f"videos/observation.image.{camera}/chunk_index"
            file_col = f"videos/observation.image.{camera}/file_index"
            from_timestamp_col = f"videos/observation.image.{camera}/from_timestamp"
            to_timestamp_col = f"videos/observation.image.{camera}/to_timestamp"
            requirements = metadata[
                [chunk_col, file_col, from_timestamp_col, to_timestamp_col, "length"]
            ].copy()
            requirements["start_frame"] = (
                requirements[from_timestamp_col].astype(float).mul(FPS).round().astype(int)
            )
            requirements["exclusive_end_frame"] = (
                requirements[to_timestamp_col].astype(float).mul(FPS).round().astype(int)
            )
            duration = requirements["exclusive_end_frame"] - requirements["start_frame"]
            invalid = (
                (requirements["start_frame"] < 0)
                | (duration != requirements["length"].astype(int))
            )
            if invalid.any():
                bad = requirements.loc[invalid].head()
                raise ValueError(
                    f"{dataset_split}/{camera} contains timestamp ranges inconsistent "
                    f"with episode length:\n{bad}"
                )
            needed = requirements.groupby([chunk_col, file_col])["exclusive_end_frame"].max()
            print(f"{dataset_split}/{camera}: {len(needed)} video shard(s)")

            for (chunk, file_index), exclusive_end_frame in needed.items():
                chunk, file_index = int(chunk), int(file_index)
                output_path = cache_path(
                    dataset_split,
                    camera,
                    chunk,
                    file_index,
                    args.input_height,
                    args.input_width,
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                frames_needed = int(exclusive_end_frame)

                if output_path.exists():
                    cached = np.load(output_path, mmap_mode="r")
                    expected_tail = (args.input_height, args.input_width, 3)
                    if cached.ndim == 4 and tuple(cached.shape[1:]) == expected_tail and len(cached) >= frames_needed:
                        print(f"  chunk{chunk} file{file_index}: cached ({len(cached)} frames), skipping")
                        continue

                relative = f"videos/observation.image.{camera}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
                local_video = _local_or_download(dataset_split, relative)
                shape = decode_resized_video(
                    local_video,
                    output_path,
                    frames_needed,
                    args.input_height,
                    args.input_width,
                )
                print(f"  chunk{chunk} file{file_index}: saved {shape}")

    print("done")


if __name__ == "__main__":
    main()
