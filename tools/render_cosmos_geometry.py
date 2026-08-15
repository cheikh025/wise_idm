"""Show the training-time vs inference-time geometry for each camera.

Training frames come from DROID, where all three cameras are natively 640x360.
At inference the same model reads a decoded Cosmos dream panel of
33 x 528 x 640 x 3, split at row 360: the wrist keeps 640x360 but both exterior
views arrive as 320x168 - a third of the linear resolution and a slightly
different aspect (1.905 vs 1.778).

No real Cosmos dream exists on this machine, so the middle column *emulates*
the Cosmos transport by downsampling the DROID frame to the panel tile size
(640x360 -> 320x180 aspect-preserving, then the 180 -> 168 vertical squeeze the
VAE decode applies). It is a geometry illustration, not a generated dream.
"""
from __future__ import annotations

import argparse
import subprocess

import cv2
import numpy as np
import pandas as pd

from droid_dataset import CAMERAS, FPS, _local_or_download, load_episode_manifest
from vision import DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH, letterbox_rgb

LABELS = {
    "wrist_image_left": "wrist",
    "exterior_image_1_left": "exterior 1",
    "exterior_image_2_left": "exterior 2",
}
# Decoded Cosmos panel tile size per view (vision.PANEL_* contract).
COSMOS_TILE = {
    "wrist_image_left": (640, 360),
    "exterior_image_1_left": (320, 168),
    "exterior_image_2_left": (320, 168),
}


def decode_frame(video_path: str, index: int) -> np.ndarray:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", video_path],
        capture_output=True, text=True, check=True,
    )
    width, height = (int(value) for value in probe.stdout.strip().split(","))
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video_path,
         "-vf", f"select='eq(n\\,{index})'", "-vsync", "0", "-vframes", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


def annotate(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/smoke/val_smoke.csv")
    parser.add_argument("--row", type=int, default=3)
    parser.add_argument("--frame", type=int, default=80)
    parser.add_argument("--out", default="/workspace/idm_train_vs_cosmos_geometry.png")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    row = manifest.iloc[args.row]
    reference = load_episode_manifest(args.manifest)[args.row]
    height, width = DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH

    rows = []
    for camera in CAMERAS:
        prefix = f"videos/observation.image.{camera}"
        chunk = int(row[f"{prefix}/chunk_index"])
        file_index = int(row[f"{prefix}/file_index"])
        offset = round(float(row[f"{prefix}/from_timestamp"]) * FPS)
        relative = f"videos/observation.image.{camera}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
        source = decode_frame(
            _local_or_download(reference.dataset_split, relative), offset + args.frame
        )
        label = LABELS[camera]

        train_input = letterbox_rgb(source[None], height, width)[0]
        tile_width, tile_height = COSMOS_TILE[camera]
        cosmos_tile = cv2.resize(source, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        cosmos_input = letterbox_rgb(cosmos_tile[None], height, width)[0]

        def content_rows(source_shape) -> int:
            scale = min(width / source_shape[1], height / source_shape[0])
            return max(1, min(height, round(source_shape[0] * scale)))

        train_rows = content_rows(source.shape[:2])
        cosmos_rows = content_rows(cosmos_tile.shape[:2])
        display = (640, 360)
        rows.append(
            np.concatenate(
                [
                    annotate(
                        cv2.resize(source, display, interpolation=cv2.INTER_NEAREST),
                        f"{label} TRAIN source (DROID) {source.shape[1]}x{source.shape[0]}",
                    ),
                    annotate(
                        cv2.resize(train_input, display, interpolation=cv2.INTER_NEAREST),
                        f"{label} TRAIN model input {width}x{height} ({width}x{train_rows} content)",
                    ),
                    annotate(
                        cv2.resize(cosmos_tile, display, interpolation=cv2.INTER_NEAREST),
                        f"{label} INFER Cosmos tile {tile_width}x{tile_height} (emulated)",
                    ),
                    annotate(
                        cv2.resize(cosmos_input, display, interpolation=cv2.INTER_NEAREST),
                        f"{label} INFER model input {width}x{height} ({width}x{cosmos_rows} content)",
                    ),
                ],
                axis=1,
            )
        )
        print(
            f"{label:<11} train {source.shape[1]}x{source.shape[0]} -> {width}x{train_rows} content | "
            f"cosmos {tile_width}x{tile_height} -> {width}x{cosmos_rows} content"
        )

    cv2.imwrite(args.out, np.concatenate(rows, axis=0)[:, :, ::-1])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
