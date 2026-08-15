"""Decode selected Cosmos3-DROID video shards into rectangular RGB caches.

Frames are resized with the canonical aspect-preserving letterbox to 224x128
(``vision.letterbox_rgb``); old square caches are not reused because their
geometry cannot be recovered.

Two properties matter for a production-scale run and are handled here:

* **Only the selected episodes' frames are materialised.** A shard is decoded
  from frame 0 (mp4 GOP seeking is not frame-exact) but a frame is letterboxed
  and stored only when some selected episode owns it. The cache keeps absolute
  in-file frame indexing, so unowned frames stay as holes in a sparse file and
  cost no disk. Across the frozen 21K/1K selection this is 20.3M stored frames
  instead of 34.2M - 1.75 TB instead of 2.94 TB.
* **Shards are processed in parallel**, each worker downloading its own mp4,
  decoding it, and (unless ``--keep-videos``) deleting the mp4 immediately so
  peak disk stays near the cache size.

Every cache file gets a ``<name>.ranges.json`` sidecar recording exactly which
frame ranges were written. ``droid_dataset.VideoFrameReader`` validates reads
against it, so a frame that was never decoded raises instead of silently
returning black pixels.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from droid_dataset import (
    CAMERAS,
    DATASET_SPLITS,
    FPS,
    HF_REPO,
    HF_REVISION,
    SNAPSHOT_ROOT,
    EpisodeRef,
    _metadata_files,
    cache_path,
    load_episode_manifest,
    ranges_path,
)
from vision import DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH, letterbox_rgb, resize_rgb_stretch

RESIZE_BATCH_FRAMES = 64
VIDEO_DIR = Path(os.environ.get("WISE_IDM_VIDEO_DIR", "/data/droid_mp4"))


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


def merge_ranges(ranges: list[tuple[int, int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def covers(merged: list[list[int]], start: int, end: int) -> bool:
    return any(block[0] <= start and end <= block[1] for block in merged)


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


def download_shard(dataset_split: str, relative_path: str) -> tuple[str, bool]:
    """Return a local mp4 path plus whether this call created it."""
    snapshot = SNAPSHOT_ROOT / dataset_split / relative_path
    if snapshot.exists():
        return str(snapshot), False
    local = VIDEO_DIR / dataset_split / relative_path
    if local.exists():
        return str(local), False

    from huggingface_hub import hf_hub_download

    local.parent.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        HF_REPO,
        f"{dataset_split}/{relative_path}",
        repo_type="dataset",
        revision=HF_REVISION,
        local_dir=str(VIDEO_DIR),
    )
    return path, True


def decode_shard(task: dict) -> dict:
    """Decode one camera shard, storing only the frames selected episodes own."""
    import torch

    torch.set_num_threads(1)

    dataset_split = task["dataset_split"]
    camera = task["camera"]
    chunk = task["chunk"]
    file_index = task["file_index"]
    height = task["input_height"]
    width = task["input_width"]
    intermediate = task.get("intermediate_size")
    resize_fn = resize_rgb_stretch if task.get("resize_mode") == "stretch" else letterbox_rgb
    wanted = merge_ranges([tuple(block) for block in task["ranges"]])
    frames_needed = max(block[1] for block in wanted)
    output_path = Path(task["output_path"])
    sidecar_path = Path(task["sidecar_path"])
    started = time.time()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="ascii"))
            cached = np.load(output_path, mmap_mode="r")
            complete = (
                cached.ndim == 4
                and tuple(cached.shape[1:]) == (height, width, 3)
                and len(cached) >= frames_needed
                and all(covers(sidecar["ranges"], start, end) for start, end in wanted)
            )
            del cached
            if complete:
                return {"status": "cached", "written": 0, "seconds": 0.0, **task["identity"]}
        except (ValueError, KeyError, json.JSONDecodeError, OSError):
            pass

    relative = f"videos/observation.image.{camera}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
    video_path, downloaded = download_shard(dataset_split, relative)
    download_seconds = time.time() - started

    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    source_width, source_height = (int(value) for value in probe.stdout.strip().split(","))

    temporary_path = Path(str(output_path) + ".tmp")
    output = np.lib.format.open_memmap(
        temporary_path,
        mode="w+",
        dtype=np.uint8,
        shape=(frames_needed, height, width, 3),
    )
    frame_bytes = source_height * source_width * 3
    command = [
        # ffmpeg defaults its thread pool to the CPU count; with 256 cores and
        # dozens of shard workers that exhausts the process's thread budget and
        # ffmpeg exits having produced no frames. Pin it low - decode is not
        # the bottleneck, the letterbox resize is.
        "ffmpeg", "-v", "error", "-threads", "2", "-i", video_path,
        "-vframes", str(frames_needed),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    decoded = 0
    written = 0
    try:
        while decoded < frames_needed:
            batch_count = min(RESIZE_BATCH_FRAMES, frames_needed - decoded)
            raw = _read_at_most(process.stdout, batch_count * frame_bytes)
            complete_frames = len(raw) // frame_bytes
            if complete_frames == 0:
                break
            if len(raw) % frame_bytes:
                raise RuntimeError(
                    f"ffmpeg returned a partial raw frame ({len(raw)} bytes, frame {frame_bytes})"
                )
            frames = np.frombuffer(raw, dtype=np.uint8).reshape(
                complete_frames, source_height, source_width, 3
            )
            batch_start = decoded
            batch_end = decoded + complete_frames
            for range_start, range_end in wanted:
                start = max(range_start, batch_start)
                end = min(range_end, batch_end)
                if start >= end:
                    continue
                resized = frames[start - batch_start : end - batch_start]
                if intermediate is not None:
                    # Chain through the real Cosmos tile size before the final
                    # resize, so training applies the exact same final resize
                    # (intermediate -> target) that inference will apply to a
                    # real decoded dream tile. Resizing straight from the
                    # DROID source to `target` in one step would use a
                    # different source aspect than inference ever sees, and -
                    # with resize_mode=stretch - a different anisotropic squash
                    # factor too (verified: a real Cosmos3 dream generation has
                    # no black bars at either panel seam, so its own tiles were
                    # evidently built without letterbox padding; matching that
                    # requires resize_mode=stretch here, not just any resize).
                    resized = resize_fn(resized, *intermediate)
                output[start:end] = resize_fn(resized, height, width)
                written += end - start
            decoded = batch_end
        if decoded != frames_needed:
            assert process.stderr is not None
            detail = process.stderr.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"decoded only {decoded}/{frames_needed} frames from {video_path}: {detail[:400]}"
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
        sidecar_temporary = Path(str(sidecar_path) + ".tmp")
        sidecar_temporary.write_text(
            json.dumps(
                {
                    "frames": frames_needed,
                    "height": height,
                    "width": width,
                    "ranges": wanted,
                    "source": relative,
                    "dataset_revision": HF_REVISION,
                },
                separators=(",", ":"),
            ),
            encoding="ascii",
        )
        os.replace(sidecar_temporary, sidecar_path)
    except BaseException:
        process.kill()
        process.wait()
        if output is not None:
            output._mmap.close()
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        # Drop the mp4 as soon as its frames are cached so peak disk stays near
        # the cache size. Anything under VIDEO_DIR is ours to remove, including
        # leftovers from an interrupted earlier pass; the pinned HF snapshot
        # tree is never touched.
        if not task["keep_videos"] and str(video_path).startswith(str(VIDEO_DIR)):
            Path(video_path).unlink(missing_ok=True)

    return {
        "status": "decoded",
        "written": written,
        "decoded": frames_needed,
        "seconds": time.time() - started,
        "download_seconds": download_seconds,
        **task["identity"],
    }


def selected_episodes(args: argparse.Namespace) -> list[EpisodeRef]:
    if args.manifest:
        refs: list[EpisodeRef] = []
        for path in args.manifest:
            refs.extend(load_episode_manifest(path))
        return refs
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


def build_tasks(args: argparse.Namespace, episodes: list[EpisodeRef]) -> list[dict]:
    tasks: dict[tuple, dict] = {}
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

        for camera in ([args.camera] if getattr(args, "camera", None) else CAMERAS):
            prefix = f"videos/observation.image.{camera}"
            frame = metadata[
                [f"{prefix}/chunk_index", f"{prefix}/file_index",
                 f"{prefix}/from_timestamp", f"{prefix}/to_timestamp", "length"]
            ].copy()
            frame.columns = ["chunk", "file_index", "from_timestamp", "to_timestamp", "length"]
            frame["start_frame"] = frame["from_timestamp"].astype(float).mul(FPS).round().astype(int)
            frame["exclusive_end_frame"] = (
                frame["to_timestamp"].astype(float).mul(FPS).round().astype(int)
            )
            duration = frame["exclusive_end_frame"] - frame["start_frame"]
            invalid = (frame["start_frame"] < 0) | (duration != frame["length"].astype(int))
            if invalid.any():
                raise ValueError(
                    f"{dataset_split}/{camera} has timestamp ranges inconsistent with length:\n"
                    f"{frame.loc[invalid].head()}"
                )
            for row in frame.itertuples(index=False):
                chunk = int(row.chunk)
                file_index = int(row.file_index)
                key = (dataset_split, camera, chunk, file_index)
                task = tasks.get(key)
                if task is None:
                    task = tasks[key] = {
                        "dataset_split": dataset_split,
                        "camera": camera,
                        "chunk": chunk,
                        "file_index": file_index,
                        "input_height": args.input_height,
                        "input_width": args.input_width,
                        "intermediate_size": (
                            (args.intermediate_height, args.intermediate_width)
                            if args.intermediate_height and args.intermediate_width
                            else None
                        ),
                        "resize_mode": args.resize_mode,
                        "keep_videos": args.keep_videos,
                        "ranges": [],
                        "output_path": str(
                            cache_path(
                                dataset_split, camera, chunk, file_index,
                                args.input_height, args.input_width,
                            )
                        ),
                        "sidecar_path": str(
                            ranges_path(
                                dataset_split, camera, chunk, file_index,
                                args.input_height, args.input_width,
                            )
                        ),
                        "identity": {
                            "shard": f"{dataset_split}/{camera}/chunk{chunk:03d}/file{file_index:03d}"
                        },
                    }
                task["ranges"].append([int(row.start_frame), int(row.exclusive_end_frame)])
    for task in tasks.values():
        task["ranges"] = merge_ranges([tuple(block) for block in task["ranges"]])
    return [tasks[key] for key in sorted(tasks)]


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", action="append", help="CSV with dataset_split and episode_index (repeatable)")
    source.add_argument("--episodes-file", help="Legacy CSV with at least episode_index")
    source.add_argument("--num-episodes", type=int, help="Legacy sequential 0..N-1 mode")
    parser.add_argument("--dataset-split", choices=DATASET_SPLITS, default="success")
    parser.add_argument("--input-height", type=int, default=DEFAULT_INPUT_HEIGHT)
    parser.add_argument("--input-width", type=int, default=DEFAULT_INPUT_WIDTH)
    parser.add_argument(
        "--camera",
        choices=CAMERAS,
        default=None,
        help="restrict this invocation to one camera (default: all three), so callers can "
        "preprocess different cameras at different target sizes",
    )
    parser.add_argument(
        "--resize-mode",
        choices=("letterbox", "stretch"),
        default="letterbox",
        help="letterbox (default) is the frozen per-view production contract - aspect-"
        "preserving, black-padded. stretch is for the experimental composite-panel "
        "architecture, matching the real Cosmos client's plain F.interpolate resize with "
        "no padding (verified: a real Cosmos dream has no black bars at either panel seam)",
    )
    parser.add_argument(
        "--intermediate-height",
        type=int,
        default=None,
        help="resize through this size before --input-height/--input-width, so the final "
        "resize step matches what a real Cosmos-decoded tile would go through at inference "
        "(e.g. the exterior cameras' real 168x320 Cosmos tile) instead of resizing straight "
        "from the raw DROID source in one step",
    )
    parser.add_argument("--intermediate-width", type=int, default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--keep-videos", action="store_true", help="do not delete downloaded mp4 shards")
    args = parser.parse_args()
    if args.input_height <= 0 or args.input_width <= 0:
        raise ValueError("input dimensions must be positive")

    episodes = selected_episodes(args)
    if len({ref.key for ref in episodes}) != len(episodes):
        raise ValueError("episode selection contains duplicate split/index keys")
    print(f"preprocessing {len(episodes)} episodes at {args.input_width}x{args.input_height}", flush=True)

    tasks = build_tasks(args, episodes)
    stored = sum(end - start for task in tasks for start, end in task["ranges"])
    decoded = sum(max(end for _, end in task["ranges"]) for task in tasks)
    print(
        f"{len(tasks)} camera shards; decode {decoded:,} frames, store {stored:,} frames "
        f"({stored * args.input_height * args.input_width * 3 / 1e12:.3f} TB)",
        flush=True,
    )

    started = time.time()
    total = len(tasks)
    done = 0
    written_total = 0
    pending = tasks
    errors: dict[str, str] = {}
    for attempt in range(3):
        workers = args.workers if attempt == 0 else max(4, args.workers // (4 * attempt))
        if attempt:
            print(f"\nretry {attempt}: {len(pending)} shard(s) with {workers} workers", flush=True)
        retry: list[dict] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(decode_shard, task): task for task in pending}
            for future in as_completed(futures):
                task = futures[future]
                shard = task["identity"]["shard"]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - retry, then report
                    errors[shard] = f"{type(exc).__name__}: {exc}"
                    retry.append(task)
                    print(f"  FAILED {shard}: {exc}", file=sys.stderr, flush=True)
                    continue
                errors.pop(shard, None)
                if attempt == 0:
                    done += 1
                written_total += result["written"]
                if attempt == 0 and (done % 25 == 0 or done == total):
                    elapsed = time.time() - started
                    rate = done / max(elapsed, 1e-9)
                    print(
                        f"  {done}/{total} shards  {written_total:,} frames stored  "
                        f"{elapsed / 60:.1f} min elapsed  "
                        f"eta {(total - done) / max(rate, 1e-9) / 60:.1f} min",
                        flush=True,
                    )
        pending = retry
        if not pending:
            break
    if pending:
        print(f"\n{len(pending)} shard(s) failed after retries:", file=sys.stderr)
        for shard, message in list(errors.items())[:20]:
            print(f"  {shard}: {message}", file=sys.stderr)
        raise SystemExit(1)
    print(f"done in {(time.time() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
