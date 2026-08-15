"""Render source DROID frames next to the exact 224x128 tensor the IDM sees.

Writes two PNGs so the resize can be judged by eye:

* ``idm_input_comparison.png`` - per camera: the native 640x360 source frame,
  the model input upscaled back to 640x360 with nearest-neighbour (so you see
  the real pixel grid, not an interpolated illusion of detail), and their
  absolute difference.
* ``idm_input_native.png`` - the three 224x128 model inputs at true size,
  which is what actually enters the network.
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
    cv2.putText(out, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="manifests/val_1k.csv")
    parser.add_argument("--row", type=int, default=0, help="which manifest row to render")
    parser.add_argument("--frame", type=int, default=60, help="frame offset inside the episode")
    parser.add_argument("--out-dir", default="/workspace")
    parser.add_argument("--height", type=int, default=DEFAULT_INPUT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_INPUT_WIDTH)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    row = manifest.iloc[args.row]
    refs = load_episode_manifest(args.manifest)
    reference = refs[args.row]
    print(f"episode {reference.key} ({reference.lab}), length {row['length']}, frame {args.frame}")

    comparison_rows = []
    native_rows = []
    for camera in CAMERAS:
        prefix = f"videos/observation.image.{camera}"
        chunk = int(row[f"{prefix}/chunk_index"])
        file_index = int(row[f"{prefix}/file_index"])
        offset = round(float(row[f"{prefix}/from_timestamp"]) * FPS)
        relative = f"videos/observation.image.{camera}/chunk-{chunk:03d}/file-{file_index:03d}.mp4"
        path = _local_or_download(reference.dataset_split, relative)
        source = decode_frame(path, offset + args.frame)

        model_input = letterbox_rgb(source[None], args.height, args.width)[0]
        source_height, source_width = source.shape[:2]
        upscaled = cv2.resize(
            model_input, (source_width, source_height), interpolation=cv2.INTER_NEAREST
        )
        difference = cv2.applyColorMap(
            np.abs(source.astype(np.int16) - upscaled.astype(np.int16))
            .max(axis=2).astype(np.uint8),
            cv2.COLORMAP_INFERNO,
        )[:, :, ::-1]

        # Match letterbox_rgb's own scale rule exactly: min() of both axis
        # scales, not just the width's. Using only the width scale (the old
        # bug here) silently mis-measured every case where height, not width,
        # is the binding constraint.
        scale = min(args.width / source_width, args.height / source_height)
        content_width = max(1, min(args.width, round(source_width * scale)))
        content_height = max(1, min(args.height, round(source_height * scale)))
        label = LABELS[camera]
        comparison_rows.append(
            np.concatenate(
                [
                    annotate(source, f"{label}: source {source_width}x{source_height}"),
                    annotate(
                        upscaled,
                        f"{label}: model input {args.width}x{args.height} "
                        f"({content_width}x{content_height} content), nearest-upscaled",
                    ),
                    annotate(difference, f"{label}: |difference| (inferno)"),
                ],
                axis=1,
            )
        )
        native_rows.append(annotate(model_input, f"{label} {args.width}x{args.height}"))
        print(
            f"  {label}: source {source_width}x{source_height} -> content "
            f"{content_width}x{content_height} inside {args.width}x{args.height} "
            f"({100 * content_width * content_height / (args.width * args.height):.1f}% of canvas used)"
        )

    comparison = np.concatenate(comparison_rows, axis=0)
    native = np.concatenate(native_rows, axis=0)
    for name, image in (("idm_input_comparison.png", comparison), ("idm_input_native.png", native)):
        target = f"{args.out_dir}/{name}"
        cv2.imwrite(target, image[:, :, ::-1])
        print(f"wrote {target}  {image.shape[1]}x{image.shape[0]}")


if __name__ == "__main__":
    main()
