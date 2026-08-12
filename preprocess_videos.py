"""Decode DROID videos into resized uint8 npy caches, one file per (camera, video-file).

Avoids repeated full-file ffmpeg re-decodes per window (AV1/general video
codecs decode sequentially; seeking to an arbitrary frame means decoding
everything before it). Decodes each needed (camera, chunk, file)'s required
frame prefix exactly once, resizes to IMAGE_SIZE, and caches to disk.

Generalized (vs. the M4 debug-subset version) to span multiple video files
per camera, since scaling past ~27-66 episodes (the debug-subset's video
chunking boundary) requires more than one file per camera.
"""
import argparse
import os
import subprocess

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

BASE = "/workspace/.hf_home/hub/datasets--nvidia--Cosmos3-DROID/snapshots/5c11a20accb11497270a5247a7f1e66ad04c956c/success"
CACHE_DIR = "/workspace/wise_idm/cache"
CAMERAS = ["wrist_image_left", "exterior_image_1_left", "exterior_image_2_left"]
IMAGE_SIZE = 128


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-episodes", type=int, default=500)
    a = p.parse_args()
    episodes = list(range(a.num_episodes))

    os.makedirs(CACHE_DIR, exist_ok=True)
    meta = pd.read_parquet(f"{BASE}/meta/episodes/chunk-000/file-000.parquet")
    meta = meta[meta["episode_index"].isin(episodes)].set_index("episode_index")

    for cam in CAMERAS:
        chunk_col = f"videos/observation.image.{cam}/chunk_index"
        file_col = f"videos/observation.image.{cam}/file_index"
        to_ts_col = f"videos/observation.image.{cam}/to_timestamp"

        needed = meta.groupby([chunk_col, file_col])[to_ts_col].max()
        print(f"{cam}: {len(needed)} video file(s) needed")

        for (chunk, file), max_to_ts in needed.items():
            chunk, file = int(chunk), int(file)
            out_path = os.path.join(CACHE_DIR, f"{cam}_chunk{chunk:03d}_file{file:03d}.npy")
            n_frames_needed = int(round(max_to_ts * 15.0)) + 2

            if os.path.exists(out_path):
                arr = np.load(out_path, mmap_mode="r")
                if arr.shape[0] >= n_frames_needed:
                    print(f"  {cam} chunk{chunk} file{file}: cached ({arr.shape[0]} frames), skipping")
                    continue

            rel = f"success/videos/observation.image.{cam}/chunk-{chunk:03d}/file-{file:03d}.mp4"
            print(f"  {cam} chunk{chunk} file{file}: downloading {rel} ...")
            local_path = hf_hub_download("nvidia/Cosmos3-DROID", rel, repo_type="dataset")

            print(f"  {cam} chunk{chunk} file{file}: decoding {n_frames_needed} frames ...")
            cmd = [
                "ffmpeg", "-v", "error", "-i", local_path,
                "-vframes", str(n_frames_needed),
                "-vf", f"scale={IMAGE_SIZE}:{IMAGE_SIZE}",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
            ]
            result = subprocess.run(cmd, capture_output=True, check=True)
            raw = result.stdout
            frame_bytes = IMAGE_SIZE * IMAGE_SIZE * 3
            n = len(raw) // frame_bytes
            frames = np.frombuffer(raw, dtype=np.uint8).reshape(n, IMAGE_SIZE, IMAGE_SIZE, 3)
            # Write to a temp file then atomically rename into place. np.save()
            # writing directly to out_path would truncate/overwrite the SAME
            # inode in place if the file already exists -- fatal if another
            # process (e.g. a concurrent training run) has that file mmap'd
            # open, since its pages would be invalidated mid-read (observed:
            # SIGBUS crash in train.py when this preprocessing script re-decoded
            # a wider frame range for the same cache file a training run had
            # open). os.replace() is atomic on the same filesystem and leaves
            # any existing mmap safely pointing at the old (unlinked) inode.
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "wb") as f:  # explicit handle: np.save() appends ".npy" if given a bare path string
                np.save(f, frames)
            os.replace(tmp_path, out_path)
            print(f"  {cam} chunk{chunk} file{file}: saved {frames.shape} ({frames.nbytes/1e6:.1f} MB)")

    print("done")


if __name__ == "__main__":
    main()
