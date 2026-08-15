
"""Create the frozen 21K train and 1K validation episode manifests.

The input catalog is the official Cosmos3-DROID episode metadata joined to
raw DROID identity fields. It must contain split-qualified episode identity,
scene identity, and all three camera shard locators.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from droid_dataset import assert_scene_disjoint, load_episode_manifest
from selection import (
    TRAIN_EPISODES,
    TRAIN_LAB_OUTCOME_QUOTAS,
    VAL_EPISODES,
    VAL_LAB_OUTCOME_QUOTAS,
    select_scene_disjoint_manifests,
    validate_production_manifest,
)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, help="joined Parquet or CSV episode catalog")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.catalog)
    catalog = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    train, val, audit = select_scene_disjoint_manifests(catalog, seed=args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train_21k.csv"
    val_path = out_dir / "val_1k.csv"
    atomic_csv(train, train_path)
    atomic_csv(val, val_path)

    train_refs = load_episode_manifest(train_path, require_scene_id=True)
    val_refs = load_episode_manifest(val_path, require_scene_id=True)
    assert_scene_disjoint(train_refs, val_refs)
    audit["train_contract"] = validate_production_manifest(
        train_refs,
        str(train_path),
        expected_count=TRAIN_EPISODES,
        lab_outcome_quotas=TRAIN_LAB_OUTCOME_QUOTAS,
    )
    audit["val_contract"] = validate_production_manifest(
        val_refs,
        str(val_path),
        expected_count=VAL_EPISODES,
        lab_outcome_quotas=VAL_LAB_OUTCOME_QUOTAS,
    )
    audit_path = out_dir / "selection_audit.json"
    temporary = audit_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="ascii")
    os.replace(temporary, audit_path)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
