"""Fetch raw DROID 1.0.1 per-episode metadata from the public GCS mirror.

Produces a single Parquet table with one row per raw episode:
``episode_id`` (``<lab>/<outcome>/<date>/<timestamp-dir>``) plus every field of
the raw ``metadata_*.json``. ``build_selection_catalog.py`` consumes this table
to attach scene identity (``building``, ``scene_id``, ``uuid``,
``robot_serial``) to the pinned Cosmos3-DROID episode metadata.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

BUCKET = "gresearch"
ROOT = "robotics/droid_raw/1.0.1"
LABS = (
    "AUTOLab",
    "CLVR",
    "GuptaLab",
    "ILIAD",
    "IPRL",
    "IRIS",
    "PennPAL",
    "RAD",
    "RAIL",
    "REAL",
    "RPL",
    "TRI",
    "WEIRD",
)
LIST_URL = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
OBJECT_URL = f"https://storage.googleapis.com/{BUCKET}/"


def _get(url: str, retries: int = 8) -> bytes:
    delay = 1.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return response.read()
        except Exception:  # noqa: BLE001 - transient GCS/network errors
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError("unreachable")


def list_lab(lab: str) -> list[str]:
    names: list[str] = []
    token = None
    while True:
        query = {
            "prefix": f"{ROOT}/{lab}/",
            "matchGlob": f"{ROOT}/*/*/*/*/metadata_*.json",
            "maxResults": "1000",
            "fields": "items(name),nextPageToken",
        }
        if token:
            query["pageToken"] = token
        payload = json.loads(_get(f"{LIST_URL}?{urllib.parse.urlencode(query)}"))
        names.extend(item["name"] for item in payload.get("items", []))
        token = payload.get("nextPageToken")
        if not token:
            break
    return names


def fetch_one(name: str) -> dict:
    parts = name.split("/")
    episode_id = "/".join(parts[3:7])
    payload = json.loads(_get(OBJECT_URL + urllib.parse.quote(name)))
    return {"episode_id": episode_id, "metadata_path": name, **payload}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--list-workers", type=int, default=13)
    parser.add_argument("--fetch-workers", type=int, default=128)
    args = parser.parse_args()

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.list_workers) as pool:
        listings = list(pool.map(list_lab, LABS))
    names = sorted({name for listing in listings for name in listing})
    print(f"listed {len(names)} metadata objects in {time.time() - started:.1f}s", flush=True)

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.fetch_workers) as pool:
        for index, row in enumerate(pool.map(fetch_one, names), start=1):
            rows.append(row)
            if index % 5000 == 0:
                print(f"  fetched {index}/{len(names)} ({time.time() - started:.0f}s)", flush=True)

    frame = pd.DataFrame(rows)
    for column in frame.columns:
        if frame[column].map(lambda value: isinstance(value, (list, dict))).any():
            frame[column] = frame[column].map(json.dumps)
    frame = frame.sort_values("metadata_path").reset_index(drop=True)
    frame.to_parquet(args.output.replace(".parquet", "_all.parquet"), index=False)

    # 264 raw episode directories ship the same metadata twice under different
    # lab-name capitalisations (e.g. ``TRI+...`` and ``tri+...``). Prefer the
    # row whose ``lab`` matches the official episode_id path component, then
    # fail loudly if any remaining duplicate disagrees about scene identity.
    official_lab = frame["episode_id"].str.split("/").str[0]
    frame["_lab_matches_path"] = (frame.get("lab", official_lab) == official_lab).astype(int)
    frame = frame.sort_values(
        ["episode_id", "_lab_matches_path", "metadata_path"], ascending=[True, False, True]
    ).reset_index(drop=True)
    identity = ["building", "scene_id", "success"]
    available = [column for column in identity if column in frame.columns]
    conflicts = (
        frame.groupby("episode_id")[available].nunique(dropna=False).max(axis=1) > 1
    )
    if conflicts.any():
        bad = conflicts[conflicts].index[:5].tolist()
        print(f"FATAL: conflicting raw identity for episodes: {bad}", file=sys.stderr)
        raise SystemExit(1)
    duplicate_count = int(frame["episode_id"].duplicated().sum())
    frame = (
        frame.drop_duplicates("episode_id", keep="first")
        .drop(columns="_lab_matches_path")
        .reset_index(drop=True)
    )
    print(f"dropped {duplicate_count} duplicate metadata files (identity agreed)", flush=True)
    frame.to_parquet(args.output, index=False)
    print(
        f"wrote {len(frame)} rows x {len(frame.columns)} columns to {args.output} "
        f"in {time.time() - started:.1f}s",
        flush=True,
    )
    print("columns:", sorted(frame.columns))


if __name__ == "__main__":
    main()
