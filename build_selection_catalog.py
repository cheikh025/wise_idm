"""Join pinned Cosmos3-DROID episode metadata to raw DROID scene identity."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from droid_dataset import CAMERAS, DATASET_SPLITS, HF_REPO, HF_REVISION, _metadata_files


RAW_IDENTITY_FIELDS = ("lab", "building", "scene_id", "success", "uuid", "robot_serial")


def load_raw_identity(path: Path) -> pd.DataFrame:
    """Load a table or an unpacked raw DROID tree of metadata JSON files."""
    if path.is_dir():
        rows = []
        for metadata_path in sorted(path.glob("*/*/*/*/metadata_*.json")):
            relative = metadata_path.relative_to(path)
            episode_id = "/".join(relative.parts[:4])
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            rows.append({"episode_id": episode_id, **payload})
        if not rows:
            raise FileNotFoundError(f"no <lab>/<outcome>/<date>/<episode>/metadata_*.json under {path}")
        return pd.DataFrame(rows)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return pd.DataFrame.from_dict(payload, orient="index").rename_axis("episode_id").reset_index()
        return pd.DataFrame(payload)
    return pd.read_csv(path)


def official_episode_frame() -> pd.DataFrame:
    columns = [
        "episode_index",
        "episode_id",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    for camera in CAMERAS:
        columns.extend(
            f"videos/observation.image.{camera}/{field}"
            for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
        )
    frames = []
    for dataset_split in DATASET_SPLITS:
        frame = pd.concat(
            [pd.read_parquet(path, columns=columns) for path in _metadata_files(dataset_split)],
            ignore_index=True,
        )
        frame.insert(0, "dataset_split", dataset_split)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def normalize_success(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "success"):
        return True
    if normalized in ("false", "0", "failure"):
        return False
    raise ValueError(f"invalid raw success value: {value!r}")


def build_catalog(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"episode_id", *RAW_IDENTITY_FIELDS}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"raw identity table is missing columns: {sorted(missing)}")
    if raw["episode_id"].duplicated().any():
        raise ValueError("raw identity table contains duplicate episode_id values")

    raw = raw[["episode_id", *RAW_IDENTITY_FIELDS]].copy()
    raw = raw.rename(columns={"success": "raw_success", "lab": "raw_lab"})
    official = official_episode_frame()
    if official.duplicated(["dataset_split", "episode_index"]).any():
        raise ValueError("official metadata contains duplicate split-qualified episode keys")
    joined = official.merge(raw, on="episode_id", how="left", validate="one_to_one")
    if joined["raw_lab"].isna().any():
        missing_ids = joined.loc[joined["raw_lab"].isna(), "episode_id"].head().tolist()
        raise ValueError(f"raw identity join is missing official episodes: {missing_ids}")

    parsed = joined["episode_id"].str.split("/", expand=True)
    if parsed.shape[1] != 4:
        raise ValueError("official episode_id values do not have exactly four path components")
    joined["lab"] = parsed[0]
    if not (parsed[1] == joined["dataset_split"]).all():
        raise ValueError("official episode_id outcome disagrees with dataset root")
    if not (joined["raw_lab"].astype(str) == joined["lab"]).all():
        raise ValueError("raw lab disagrees with official episode_id")
    raw_success = joined["raw_success"].map(normalize_success)
    if not (raw_success == (joined["dataset_split"] == "success")).all():
        raise ValueError("raw success flag disagrees with official dataset root")

    for column in ("building", "scene_id", "uuid", "robot_serial"):
        if joined[column].isna().any() or (joined[column].astype(str).str.strip() == "").any():
            raise ValueError(f"raw identity contains empty {column}")
    invalid_scene = joined["scene_id"].astype(str).str.strip().isin(("-1", "unknown", "N/A"))
    if invalid_scene.any():
        raise ValueError("raw identity contains missing/sentinel scene_id values")
    joined["scene_key"] = (
        joined["lab"].astype(str)
        + "|"
        + joined["building"].astype(str)
        + "|"
        + joined["scene_id"].astype(str)
    )
    joined["source_repo"] = HF_REPO
    joined["source_revision"] = HF_REVISION
    return joined.drop(columns=("raw_lab", "raw_success"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-metadata", required=True, help="raw identity table or unpacked DROID root")
    parser.add_argument("--output", required=True, help="output .parquet catalog")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    catalog = build_catalog(load_raw_identity(Path(args.raw_metadata)))
    temporary = output.with_suffix(output.suffix + ".tmp")
    catalog.to_parquet(temporary, index=False)
    os.replace(temporary, output)
    print(f"wrote {len(catalog)} episodes to {output} at revision {HF_REVISION}")


if __name__ == "__main__":
    main()
