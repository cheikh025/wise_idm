"""Recover raw DROID scene identity for episodes with no metadata_*.json.

31 of the 71,907 pinned Cosmos3-DROID episodes live in raw DROID directories
that ship no ``metadata_*.json``. Their ``trajectory.h5`` root attributes still
carry the authoritative ``building``/``scene_id``/``robot_serial_number``, so
this tool reads those (1-3 MB per file) and appends the recovered rows to the
raw identity table produced by ``fetch_droid_raw_metadata.py``.

Recovered rows are marked ``scene_identity_source="trajectory_h5_attrs"``; rows
that came from a metadata JSON are marked ``"metadata_json"``. ``uuid`` is not
present in the HDF5 attributes, so a deterministic ``h5:<episode_id>`` stand-in
is used - it stays unique and non-empty, and the source column makes the
substitution auditable.
"""
from __future__ import annotations

import argparse
import io
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import h5py
import pandas as pd

OBJECT_URL = "https://storage.googleapis.com/gresearch/"
ROOT = "robotics/droid_raw/1.0.1"


def read_h5_identity(episode_id: str) -> dict:
    url = OBJECT_URL + urllib.parse.quote(f"{ROOT}/{episode_id}/trajectory.h5")
    with urllib.request.urlopen(url, timeout=180) as response:
        payload = response.read()
    with h5py.File(io.BytesIO(payload), "r") as handle:
        attrs = dict(handle.attrs)
    lab = episode_id.split("/")[0]
    outcome = episode_id.split("/")[1]
    success = str(attrs["success"]).strip().lower() == "true"
    if success != (outcome == "success"):
        raise ValueError(f"{episode_id}: h5 success={attrs['success']} disagrees with {outcome}")
    return {
        "episode_id": episode_id,
        "metadata_path": f"{ROOT}/{episode_id}/trajectory.h5",
        "uuid": f"h5:{episode_id}",
        "lab": lab,
        "building": str(attrs["building"]).strip(),
        "scene_id": str(attrs["scene_id"]).strip(),
        "success": success,
        "robot_serial": str(attrs["robot_serial_number"]).strip().lower(),
        "current_task": str(attrs.get("current_task", "")).strip(),
        "scene_identity_source": "trajectory_h5_attrs",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-identity", required=True)
    parser.add_argument("--episode-meta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    raw = pd.read_parquet(args.raw_identity)
    official = pd.read_parquet(args.episode_meta)
    if "scene_identity_source" not in raw.columns:
        raw["scene_identity_source"] = "metadata_json"

    missing = sorted(set(official["episode_id"]) - set(raw["episode_id"]))
    print(f"{len(missing)} pinned episodes have no raw metadata JSON", flush=True)
    if missing:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            recovered = pd.DataFrame(list(pool.map(read_h5_identity, missing)))
        for column in ("building", "scene_id", "robot_serial"):
            blank = recovered[column].astype(str).str.strip() == ""
            if blank.any():
                raise ValueError(f"recovered rows have an empty {column}")
        # The JSON rows type scene_id as int64 while the HDF5 attributes are
        # strings; the scene key is textual either way, so normalise both.
        for column in ("lab", "building", "scene_id", "uuid", "robot_serial"):
            raw[column] = raw[column].astype(str).str.strip()
            recovered[column] = recovered[column].astype(str).str.strip()
        raw = pd.concat([raw, recovered], ignore_index=True)
        print(recovered[["episode_id", "building", "scene_id", "robot_serial"]].to_string(), flush=True)

    still_missing = set(official["episode_id"]) - set(raw["episode_id"])
    if still_missing:
        raise SystemExit(f"still missing raw identity for {len(still_missing)} episodes")
    if raw["episode_id"].duplicated().any():
        raise SystemExit("recovered raw identity table contains duplicate episode_id values")
    raw.to_parquet(args.output, index=False)
    counts = raw["scene_identity_source"].value_counts().to_dict()
    print(f"wrote {len(raw)} rows to {args.output}; sources={counts}", flush=True)


if __name__ == "__main__":
    main()
