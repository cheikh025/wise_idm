"""Validation and audit helpers for the frozen 21K/1K DROID selection."""
from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
import hashlib
import json

import pandas as pd

from droid_dataset import EpisodeRef, FPS, HF_REPO, HF_REVISION, window_starts


TRAIN_EPISODES = 21_000
VAL_EPISODES = 1_000
MIN_EPISODE_FRAMES = 33
SELECTION_ALGORITHM_VERSION = "scene_block_v1"
OFFICIAL_CAMERA_KEYS = (
    "observation.image.wrist_image_left",
    "observation.image.exterior_image_1_left",
    "observation.image.exterior_image_2_left",
)

# Official Cosmos3-DROID population at revision
# 5c11a20accb11497270a5247a7f1e66ad04c956c after the unavoidable
# length >= 33 eligibility constraint. Values are (success, failure).
ELIGIBLE_LAB_SPLIT_COUNTS = {
    "AUTOLab": (7_012, 3_274),
    "CLVR": (4_701, 372),
    "GuptaLab": (1_114, 159),
    "ILIAD": (2_071, 1_136),
    "IPRL": (4_416, 1_235),
    "IRIS": (3_169, 169),
    "PennPAL": (2_791, 89),
    "RAD": (999, 461),
    "RAIL": (5_403, 2_498),
    "REAL": (3_497, 1_470),
    "RPL": (1_947, 533),
    "TRI": (18_091, 1_674),
    "WEIRD": (2_373, 599),
}
ELIGIBLE_LAB_OUTCOME_COUNTS = {
    (lab, outcome): counts[outcome_index]
    for lab, counts in ELIGIBLE_LAB_SPLIT_COUNTS.items()
    for outcome_index, outcome in enumerate(("success", "failure"))
}


def largest_remainder_quotas(
    total: int, population: Mapping[Hashable, int]
) -> dict[Hashable, int]:
    """Apportion ``total`` deterministically over arbitrary population keys."""
    if total < 0 or not population or any(count < 0 for count in population.values()):
        raise ValueError("total and population counts must be non-negative")
    population_total = sum(population.values())
    if population_total == 0 or total > population_total:
        raise ValueError("quota total must not exceed a non-empty population")
    exact = {key: total * count / population_total for key, count in population.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(quotas.values())
    priority = sorted(
        population,
        key=lambda key: (exact[key] - quotas[key], population[key], str(key)),
        reverse=True,
    )
    for key in priority[:remainder]:
        quotas[key] += 1
    return quotas


TRAIN_LAB_OUTCOME_QUOTAS = largest_remainder_quotas(
    TRAIN_EPISODES, ELIGIBLE_LAB_OUTCOME_COUNTS
)
VAL_LAB_OUTCOME_QUOTAS = largest_remainder_quotas(
    VAL_EPISODES, ELIGIBLE_LAB_OUTCOME_COUNTS
)


def _marginal_lab_quotas(quotas: Mapping[tuple[str, str], int]) -> dict[str, int]:
    return {
        lab: sum(quotas[(lab, outcome)] for outcome in ("success", "failure"))
        for lab in ELIGIBLE_LAB_SPLIT_COUNTS
    }


TRAIN_LAB_QUOTAS = _marginal_lab_quotas(TRAIN_LAB_OUTCOME_QUOTAS)
VAL_LAB_QUOTAS = _marginal_lab_quotas(VAL_LAB_OUTCOME_QUOTAS)


def _strict_episode_id(episode_id: str, expected_lab: str, expected_split: str) -> None:
    parts = episode_id.split("/")
    if len(parts) != 4 or any(not part for part in parts):
        raise ValueError(
            "episode_id must be <lab>/<success|failure>/<date>/<timestamp-directory>"
        )
    if parts[0] != expected_lab or parts[1] != expected_split:
        raise ValueError(
            f"episode_id {episode_id!r} disagrees with lab={expected_lab!r} "
            f"or dataset_split={expected_split!r}"
        )


def _manifest_frame(refs: Sequence[EpisodeRef], path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "dataset_split",
        "episode_index",
        "episode_id",
        "lab",
        "building",
        "scene_id",
        "scene_key",
        "uuid",
        "robot_serial",
        "length",
        "source_repo",
        "source_revision",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"production manifest {path} is missing columns: {sorted(missing)}")
    expected_keys = [(ref.dataset_split, ref.episode_index) for ref in refs]
    actual_keys = list(zip(frame["dataset_split"].astype(str), frame["episode_index"].astype(int)))
    if expected_keys != actual_keys:
        raise ValueError(f"production manifest {path} changed while loading")
    for column in (
        "lab",
        "building",
        "scene_id",
        "scene_key",
        "episode_id",
        "uuid",
        "robot_serial",
    ):
        if frame[column].isna().any() or (frame[column].astype(str).str.strip() == "").any():
            raise ValueError(f"production manifest {path} contains an empty {column}")
    if set(frame["source_repo"].astype(str)) != {HF_REPO}:
        raise ValueError(f"production manifest {path} does not identify {HF_REPO}")
    if set(frame["source_revision"].astype(str)) != {HF_REVISION}:
        raise ValueError(
            f"production manifest {path} is not pinned to revision {HF_REVISION}"
        )
    expected_scene_keys = (
        frame["lab"].astype(str).str.strip()
        + "|"
        + frame["building"].astype(str).str.strip()
        + "|"
        + frame["scene_id"].astype(str).str.strip()
    )
    if not (frame["scene_key"].astype(str).str.strip() == expected_scene_keys).all():
        raise ValueError(f"production manifest {path} contains an inconsistent scene_key")
    if frame["uuid"].astype(str).duplicated().any():
        raise ValueError(f"production manifest {path} contains duplicate raw UUIDs")
    numeric_length = pd.to_numeric(frame["length"], errors="coerce")
    if numeric_length.isna().any() or (numeric_length < MIN_EPISODE_FRAMES).any():
        raise ValueError(
            f"production manifest {path} contains an episode shorter than "
            f"{MIN_EPISODE_FRAMES} frames"
        )

    for ref, row in zip(refs, frame.itertuples(index=False), strict=True):
        _strict_episode_id(str(row.episode_id), str(row.lab), str(row.dataset_split))
        if (
            ref.scene_id != str(row.scene_key).strip()
            or ref.lab != str(row.lab).strip()
            or ref.episode_id != str(row.episode_id).strip()
            or ref.length != int(row.length)
        ):
            raise ValueError(f"production manifest {path} changed while loading metadata")
    return frame


def validate_production_manifest(
    refs: Sequence[EpisodeRef],
    path: str,
    *,
    expected_count: int,
    lab_outcome_quotas: Mapping[tuple[str, str], int],
) -> dict:
    """Enforce the frozen eligible-population lab x outcome allocation."""
    if len(refs) != expected_count:
        raise ValueError(
            f"production manifest {path} must contain exactly {expected_count} episodes, got {len(refs)}"
        )
    frame = _manifest_frame(refs, path)
    lab_counts = Counter(frame["lab"].astype(str))
    lab_outcome_counts = Counter(
        zip(frame["lab"].astype(str), frame["dataset_split"].astype(str))
    )
    if dict(lab_outcome_counts) != dict(lab_outcome_quotas):
        got = {f"{lab}/{outcome}": count for (lab, outcome), count in lab_outcome_counts.items()}
        expected = {
            f"{lab}/{outcome}": count for (lab, outcome), count in lab_outcome_quotas.items()
        }
        raise ValueError(
            f"production manifest {path} lab/outcome quotas differ from the frozen "
            f"eligible-population quotas: got {dict(sorted(got.items()))}, "
            f"expected {dict(sorted(expected.items()))}"
        )

    split_counts = Counter(frame["dataset_split"].astype(str))
    return {
        "episode_count": len(frame),
        "eligible_min_frames": MIN_EPISODE_FRAMES,
        "lab_counts": dict(sorted(lab_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "lab_outcome_counts": {
            f"{lab}/{outcome}": int(count)
            for (lab, outcome), count in sorted(lab_outcome_counts.items())
        },
    }


def _scene_key(row: pd.Series) -> str:
    if "scene_key" in row and not pd.isna(row["scene_key"]):
        value = str(row["scene_key"]).strip()
        if value:
            return value
    values = (row.get("lab"), row.get("building"), row.get("scene_id"))
    if any(pd.isna(value) for value in values):
        raise ValueError("catalog rows need scene_key or non-empty lab/building/scene_id")
    return "|".join(str(value).strip() for value in values)


def _stable_random_key(seed: int, *parts: object) -> str:
    payload = json.dumps([seed, *parts], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _catalog_shard_columns(frame: pd.DataFrame) -> list[str]:
    official = [
        f"videos/{camera}/{field}"
        for camera in OFFICIAL_CAMERA_KEYS
        for field in ("chunk_index", "file_index")
    ]
    if all(column in frame.columns for column in official):
        return official
    aliases = [
        f"{camera}_{field}"
        for camera in ("wrist", "left", "right")
        for field in ("video_chunk_index", "video_file_index")
    ]
    if all(column in frame.columns for column in aliases):
        return aliases
    raise ValueError(
        "selection catalog needs all three cameras' chunk_index/file_index fields "
        "for shard-block selection"
    )


def _block_key(row: pd.Series, shard_columns: Sequence[str]) -> tuple[int, ...]:
    return tuple(int(row[column]) for column in shard_columns)


def _selection_shard_audit(frame: pd.DataFrame, *, stride: int) -> dict:
    """Summarize selected episodes, frames, windows, and decode prefixes per shard."""
    total_windows = sum(
        len(window_starts(int(length), stride=stride)) for length in frame["length"]
    )
    records = []
    for camera in OFFICIAL_CAMERA_KEYS:
        prefix = f"videos/{camera}"
        required = [
            f"{prefix}/chunk_index",
            f"{prefix}/file_index",
            f"{prefix}/from_timestamp",
            f"{prefix}/to_timestamp",
        ]
        if not all(column in frame.columns for column in required):
            raise ValueError(f"selection catalog lacks timestamp shard fields for {camera}")
        working = frame[
            ["dataset_split", "length", *required]
        ].copy()
        working["start_frame"] = (
            working[f"{prefix}/from_timestamp"].astype(float).mul(FPS).round().astype(int)
        )
        working["exclusive_end_frame"] = (
            working[f"{prefix}/to_timestamp"].astype(float).mul(FPS).round().astype(int)
        )
        if (
            working["exclusive_end_frame"] - working["start_frame"]
            != working["length"].astype(int)
        ).any():
            raise ValueError(f"selection catalog has invalid timestamp duration for {camera}")
        working["window_count"] = [
            len(window_starts(int(length), stride=stride)) for length in working["length"]
        ]
        group_columns = [
            "dataset_split",
            f"{prefix}/chunk_index",
            f"{prefix}/file_index",
        ]
        for key, rows in working.groupby(group_columns, sort=True):
            dataset_split, chunk, file_index = key
            episode_count = len(rows)
            window_count = int(rows["window_count"].sum())
            records.append(
                {
                    "camera": camera,
                    "dataset_split": str(dataset_split),
                    "chunk_index": int(chunk),
                    "file_index": int(file_index),
                    "episode_count": episode_count,
                    "episode_share_within_camera": episode_count / len(frame),
                    "selected_frame_count": int(rows["length"].sum()),
                    "window_count": window_count,
                    "window_share_within_camera": window_count / total_windows,
                    "min_start_frame": int(rows["start_frame"].min()),
                    "max_exclusive_end_frame": int(rows["exclusive_end_frame"].max()),
                    "decode_frame_requirement": int(rows["exclusive_end_frame"].max()),
                }
            )
    return {
        "stride": stride,
        "window_count": total_windows,
        "unique_camera_shards": len(records),
        "max_camera_shard_episode_share": max(
            record["episode_share_within_camera"] for record in records
        ),
        "max_camera_shard_window_share": max(
            record["window_share_within_camera"] for record in records
        ),
        "camera_shards": records,
    }


def select_scene_disjoint_manifests(
    catalog: pd.DataFrame,
    *,
    seed: int = 0,
    population: Mapping[tuple[str, str], int] = ELIGIBLE_LAB_OUTCOME_COUNTS,
    train_quotas: Mapping[tuple[str, str], int] = TRAIN_LAB_OUTCOME_QUOTAS,
    val_quotas: Mapping[tuple[str, str], int] = VAL_LAB_OUTCOME_QUOTAS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Select deterministic scene-disjoint manifests with frozen joint quotas.

    Scenes are allocated to validation first. Within each lab/outcome cell,
    candidate scenes and episodes use seeded stable hashes. Video-shard fields
    are used as secondary locality keys when the catalog provides them.
    """
    required = {"dataset_split", "episode_index", "episode_id", "lab", "length"}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f"selection catalog is missing columns: {sorted(missing)}")
    frame = catalog.copy()
    frame["dataset_split"] = frame["dataset_split"].astype(str).str.strip()
    frame["lab"] = frame["lab"].astype(str).str.strip()
    if set(frame["dataset_split"]) != {"success", "failure"}:
        raise ValueError("selection catalog must contain exactly success and failure roots")
    if set(frame["lab"]) != set(ELIGIBLE_LAB_SPLIT_COUNTS):
        raise ValueError("selection catalog lab set differs from the pinned population")
    frame["scene_key"] = frame.apply(_scene_key, axis=1)
    if "source_repo" not in frame.columns or "source_revision" not in frame.columns:
        raise ValueError("selection catalog must record source_repo and source_revision")
    repositories = set(frame["source_repo"].astype(str))
    revisions = set(frame["source_revision"].astype(str))
    if repositories != {HF_REPO}:
        raise ValueError(f"selection catalog source {sorted(repositories)} does not match {HF_REPO}")
    if revisions != {HF_REVISION}:
        raise ValueError(
            f"selection catalog revision {sorted(revisions)} does not match pinned {HF_REVISION}"
        )
    frame = frame[frame["length"].astype(int) >= MIN_EPISODE_FRAMES].copy()
    if frame.duplicated(["dataset_split", "episode_index"]).any():
        raise ValueError("selection catalog contains duplicate split-qualified episode keys")
    if frame["episode_id"].astype(str).duplicated().any():
        raise ValueError("selection catalog contains duplicate episode_id values")
    for row in frame.itertuples(index=False):
        _strict_episode_id(str(row.episode_id), str(row.lab), str(row.dataset_split))
    shard_columns = _catalog_shard_columns(frame)
    frame["_block_key"] = pd.Series(
        [_block_key(row, shard_columns) for _, row in frame.iterrows()],
        index=frame.index,
        dtype=object,
    )

    selected_val = []
    selected_train = []
    val_scenes: set[str] = set()
    for lab in ELIGIBLE_LAB_SPLIT_COUNTS:
        for outcome in ("success", "failure"):
            cell = frame[(frame["lab"] == lab) & (frame["dataset_split"] == outcome)].copy()
            available = len(cell)
            expected_available = population[(lab, outcome)]
            if available != expected_available:
                raise ValueError(
                    f"eligible catalog count for {lab}/{outcome} is {available}; "
                    f"expected {expected_available} at the pinned dataset revision"
                )

            blocks = list(cell.groupby("_block_key", sort=False))
            blocks.sort(key=lambda item: _stable_random_key(seed, "val-block", *item[0]))
            val_quota = val_quotas[(lab, outcome)]
            cell_val = []
            selected_count = 0
            for _, block_rows in blocks:
                scenes = list(block_rows.groupby("scene_key", sort=False))
                scenes.sort(
                    key=lambda item: _stable_random_key(
                        seed, "val-scene", lab, outcome, item[0]
                    )
                )
                for scene, scene_rows in scenes:
                    if scene in val_scenes:
                        continue
                    rows = scene_rows.copy()
                    rows["_selection_key"] = [
                        _stable_random_key(seed, "val-episode", outcome, int(index))
                        for index in rows["episode_index"]
                    ]
                    rows = rows.sort_values("_selection_key")
                    take = min(val_quota - selected_count, len(rows))
                    if take:
                        cell_val.append(rows.iloc[:take])
                        selected_count += take
                        val_scenes.add(scene)
                    if selected_count == val_quota:
                        break
                if selected_count == val_quota:
                    break
            if sum(len(part) for part in cell_val) != val_quota:
                raise ValueError(f"cannot fill validation quota for {lab}/{outcome}")
            selected_val.extend(cell_val)

    val = pd.concat(selected_val, ignore_index=True).drop(columns="_selection_key")
    train_pool = frame[~frame["scene_key"].isin(val_scenes)].copy()
    for lab in ELIGIBLE_LAB_SPLIT_COUNTS:
        for outcome in ("success", "failure"):
            cell = train_pool[
                (train_pool["lab"] == lab) & (train_pool["dataset_split"] == outcome)
            ].copy()
            train_quota = train_quotas[(lab, outcome)]
            if len(cell) < train_quota:
                raise ValueError(
                    f"scene-disjoint train pool cannot fill {lab}/{outcome}: "
                    f"need {train_quota}, have {len(cell)}"
                )
            blocks = list(cell.groupby("_block_key", sort=False))
            blocks.sort(key=lambda item: _stable_random_key(seed, "train-block", *item[0]))
            cell_train = []
            selected_count = 0
            for _, block_rows in blocks:
                rows = block_rows.copy()
                rows["_selection_key"] = [
                    _stable_random_key(seed, "train-episode", outcome, int(index))
                    for index in rows["episode_index"]
                ]
                rows = rows.sort_values("_selection_key")
                take = min(train_quota - selected_count, len(rows))
                if take:
                    cell_train.append(rows.iloc[:take])
                    selected_count += take
                if selected_count == train_quota:
                    break
            selected_train.extend(cell_train)

    train = pd.concat(selected_train, ignore_index=True).drop(columns="_selection_key")
    train = train.drop(columns="_block_key")
    val = val.drop(columns="_block_key")
    sort_columns = ["lab", "dataset_split", "episode_index"]
    train = train.sort_values(sort_columns).reset_index(drop=True)
    val = val.sort_values(sort_columns).reset_index(drop=True)
    audit = {
        "algorithm": SELECTION_ALGORITHM_VERSION,
        "seed": seed,
        "eligible_episode_count": len(frame),
        "train_episode_count": len(train),
        "val_episode_count": len(val),
        "train_scene_count": train["scene_key"].nunique(),
        "val_scene_count": val["scene_key"].nunique(),
        "scene_overlap_count": len(set(train["scene_key"]) & set(val["scene_key"])),
        "train_shards": _selection_shard_audit(train, stride=16),
        "val_shards": _selection_shard_audit(val, stride=32),
    }
    for camera in OFFICIAL_CAMERA_KEYS:
        columns = [f"videos/{camera}/{field}" for field in ("chunk_index", "file_index")]
        if all(column in frame.columns for column in columns):
            train_shards = set(map(tuple, train[columns].astype(int).to_numpy()))
            val_shards = set(map(tuple, val[columns].astype(int).to_numpy()))
            audit[f"{camera}/train_shards"] = len(train_shards)
            audit[f"{camera}/val_shards"] = len(val_shards)
            audit[f"{camera}/shared_shards"] = len(train_shards & val_shards)
    return train, val, audit


def window_audit(dataset, manifest_path: str | None = None) -> dict:
    """Report window concentration by episode, outcome, lab, and video shard."""
    episode_window_counts = Counter(window.episode_key for window in dataset.windows)
    split_window_counts = Counter(window.dataset_split for window in dataset.windows)
    audit = {
        "window_count": len(dataset.windows),
        "window_counts_by_split": dict(sorted(split_window_counts.items())),
        "min_windows_per_episode": min(episode_window_counts.values()),
        "max_windows_per_episode": max(episode_window_counts.values()),
        "max_episode_window_share": max(episode_window_counts.values()) / len(dataset.windows),
    }
    if manifest_path is not None:
        frame = pd.read_csv(manifest_path)
        lab_by_key = {
            (str(row.dataset_split), int(row.episode_index)): str(row.lab)
            for row in frame.itertuples(index=False)
        }
        lab_windows = Counter(
            lab_by_key[window.episode_key] for window in dataset.windows
        )
        audit["window_counts_by_lab"] = dict(sorted(lab_windows.items()))
        audit["window_share_by_lab"] = {
            lab: count / len(dataset.windows) for lab, count in sorted(lab_windows.items())
        }

    shard_windows = Counter()
    shard_episodes = Counter()
    shard_frames = Counter()
    shard_min_start = {}
    shard_max_end = {}
    for ref in dataset.episodes:
        row = dataset.meta[ref.key]
        length = int(row["length"])
        for camera in dataset.cameras:
            shard = (
                ref.dataset_split,
                camera,
                int(row[f"videos/observation.image.{camera}/chunk_index"]),
                int(row[f"videos/observation.image.{camera}/file_index"]),
            )
            start = round(float(row[f"videos/observation.image.{camera}/from_timestamp"]) * FPS)
            end = round(float(row[f"videos/observation.image.{camera}/to_timestamp"]) * FPS)
            shard_episodes[shard] += 1
            shard_frames[shard] += length
            shard_windows[shard] += episode_window_counts[ref.key]
            shard_min_start[shard] = min(start, shard_min_start.get(shard, start))
            shard_max_end[shard] = max(end, shard_max_end.get(shard, end))
    audit["unique_camera_shards"] = len(shard_windows)
    audit["max_camera_shard_window_share"] = max(shard_windows.values()) / len(dataset.windows)
    audit["max_camera_shard_episode_share"] = max(shard_episodes.values()) / len(dataset.episodes)
    audit["camera_shards"] = {
        "/".join(map(str, shard)): {
            "episode_count": shard_episodes[shard],
            "episode_share_within_camera": shard_episodes[shard] / len(dataset.episodes),
            "selected_frame_count": shard_frames[shard],
            "window_count": shard_windows[shard],
            "window_share_within_camera": shard_windows[shard] / len(dataset.windows),
            "min_start_frame": shard_min_start[shard],
            "max_exclusive_end_frame": shard_max_end[shard],
            "decode_frame_requirement": shard_max_end[shard],
        }
        for shard in sorted(shard_windows)
    }
    return audit
