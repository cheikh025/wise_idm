"""Build a small manifest from the episodes whose frames are already cached.

Used for short LR / stability probes that must not wait for the full 1.75 TB
preprocessing pass. An episode qualifies only when, for all three cameras, the
cache file exists and its ``.ranges.json`` sidecar covers the episode's whole
in-file span - so a probe reads exactly the same bytes the production run will.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from droid_dataset import CAMERAS, FPS, cache_path, ranges_path
from vision import DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH


def cached_shards(height: int, width: int) -> dict[tuple, list[list[int]]]:
    """Map (split, camera, chunk, file) -> written frame ranges, from sidecars."""
    directory = cache_path("success", CAMERAS[0], 0, 0, height, width).parent
    available: dict[tuple, list[list[int]]] = {}
    for sidecar in directory.glob(f"*_{width}x{height}_*.ranges.json"):
        name = sidecar.name
        for split in ("success", "failure"):
            if not name.startswith(split + "_"):
                continue
            for camera in CAMERAS:
                marker = f"{split}_{camera}_chunk"
                if not name.startswith(marker):
                    continue
                rest = name[len(marker):]
                chunk = int(rest[:3])
                file_index = int(rest[len("XXX_file"):][:3])
                if not cache_path(split, camera, chunk, file_index, height, width).exists():
                    continue
                available[(split, camera, chunk, file_index)] = json.loads(
                    sidecar.read_text(encoding="ascii")
                )["ranges"]
    return available


def covered(ranges: list[list[int]], start: int, stop: int) -> bool:
    return any(block[0] <= start and stop <= block[1] for block in ranges)


def filter_manifest(frame: pd.DataFrame, available: dict) -> pd.DataFrame:
    keep = pd.Series(True, index=frame.index)
    for camera in CAMERAS:
        prefix = f"videos/observation.image.{camera}"
        chunks = frame[f"{prefix}/chunk_index"].astype(int)
        files = frame[f"{prefix}/file_index"].astype(int)
        starts = (frame[f"{prefix}/from_timestamp"].astype(float) * FPS).round().astype(int)
        stops = (frame[f"{prefix}/to_timestamp"].astype(float) * FPS).round().astype(int)
        keep &= [
            covered(available.get((split, camera, chunk, file_index), []), start, stop)
            for split, chunk, file_index, start, stop in zip(
                frame["dataset_split"], chunks, files, starts, stops
            )
        ]
    return frame[keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", default="manifests/train_21k.csv")
    parser.add_argument("--val-manifest", default="manifests/val_1k.csv")
    parser.add_argument("--out-dir", default="manifests/probe")
    parser.add_argument("--max-train", type=int, default=3000)
    parser.add_argument("--max-val", type=int, default=300)
    parser.add_argument("--input-height", type=int, default=DEFAULT_INPUT_HEIGHT)
    parser.add_argument("--input-width", type=int, default=DEFAULT_INPUT_WIDTH)
    args = parser.parse_args()

    available = cached_shards(args.input_height, args.input_width)
    print(f"{len(available)} fully cached camera shards")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, path, limit in (
        ("train", args.train_manifest, args.max_train),
        ("val", args.val_manifest, args.max_val),
    ):
        frame = pd.read_csv(path)
        usable = filter_manifest(frame, available)
        print(f"{name}: {len(usable)}/{len(frame)} episodes fully cached")
        if usable.empty:
            raise SystemExit(f"no cached episodes for {name}; wait for preprocessing")
        # Deterministic thinning that keeps the lab mix of the parent manifest.
        selected = (
            usable.groupby("lab", group_keys=False, sort=True)
            .apply(lambda group: group.head(max(1, round(limit * len(group) / len(usable)))))
            .sort_values(["lab", "dataset_split", "episode_index"])
            .reset_index(drop=True)
        )
        target = out_dir / f"{name}_probe.csv"
        selected.to_csv(target, index=False)
        windows = None
        print(
            f"  wrote {len(selected)} episodes to {target} "
            f"(labs={selected['lab'].nunique()}, "
            f"outcomes={selected['dataset_split'].value_counts().to_dict()})"
        )
        del windows


if __name__ == "__main__":
    main()
