import tempfile
import unittest
from pathlib import Path

import pandas as pd

from droid_dataset import EpisodeRef, assert_scene_disjoint, load_episode_manifest, window_starts
from droid_dataset import HF_REPO, HF_REVISION
from selection import (
    ELIGIBLE_LAB_OUTCOME_COUNTS,
    TRAIN_LAB_OUTCOME_QUOTAS,
    TRAIN_LAB_QUOTAS,
    VAL_LAB_OUTCOME_QUOTAS,
    VAL_LAB_QUOTAS,
    select_scene_disjoint_manifests,
)


class DatasetContractTest(unittest.TestCase):
    def test_frozen_joint_quotas_have_exact_split_sizes(self):
        self.assertEqual(sum(ELIGIBLE_LAB_OUTCOME_COUNTS.values()), 71_253)
        self.assertEqual(sum(TRAIN_LAB_OUTCOME_QUOTAS.values()), 21_000)
        self.assertEqual(sum(VAL_LAB_OUTCOME_QUOTAS.values()), 1_000)
        self.assertEqual(TRAIN_LAB_OUTCOME_QUOTAS[("TRI", "success")], 5_332)
        self.assertEqual(TRAIN_LAB_OUTCOME_QUOTAS[("TRI", "failure")], 493)
        self.assertEqual(VAL_LAB_OUTCOME_QUOTAS[("TRI", "success")], 254)
        self.assertEqual(VAL_LAB_OUTCOME_QUOTAS[("TRI", "failure")], 24)
        self.assertEqual(TRAIN_LAB_QUOTAS["TRI"], 5_825)
        self.assertEqual(VAL_LAB_QUOTAS["TRI"], 278)

    def test_window_starts_preserve_stride_and_end_align_tail(self):
        self.assertEqual(window_starts(33, stride=16), [0])
        self.assertEqual(window_starts(66, stride=16), [0, 16, 32, 33])
        self.assertEqual(window_starts(32, stride=16), [])
        self.assertEqual(window_starts(66, stride=32), [0, 32, 33])

    def test_manifest_episode_identity_includes_dataset_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.csv"
            pd.DataFrame(
                [
                    {
                        "dataset_split": "success",
                        "episode_index": 7,
                        "scene_key": "lab|building|scene-a",
                        "lab": "lab",
                        "episode_id": "lab/success/date/time-a",
                        "length": 33,
                    },
                    {
                        "dataset_split": "failure",
                        "episode_index": 7,
                        "scene_key": "lab|building|scene-b",
                        "lab": "lab",
                        "episode_id": "lab/failure/date/time-b",
                        "length": 34,
                    },
                ]
            ).to_csv(path, index=False)

            refs = load_episode_manifest(path, require_scene_id=True)

        self.assertEqual([ref.key for ref in refs], [("success", 7), ("failure", 7)])
        self.assertEqual(refs[0].scene_id, "lab|building|scene-a")
        self.assertEqual(refs[0].length, 33)

    def test_scene_overlap_is_rejected(self):
        train = [EpisodeRef("success", 1, "same-scene")]
        validation = [EpisodeRef("failure", 2, "same-scene")]
        with self.assertRaisesRegex(ValueError, "scene overlap"):
            assert_scene_disjoint(train, validation)

    def test_episode_overlap_is_rejected_even_when_scene_ids_differ(self):
        train = [EpisodeRef("success", 1, "scene-a")]
        validation = [EpisodeRef("success", 1, "scene-b")]
        with self.assertRaisesRegex(ValueError, "episode overlap"):
            assert_scene_disjoint(train, validation)

    def test_selection_is_deterministic_and_scene_disjoint(self):
        rows = []
        population = {}
        train_quotas = {}
        val_quotas = {}
        episode_index = 0
        for lab in sorted({key[0] for key in ELIGIBLE_LAB_OUTCOME_COUNTS}):
            for outcome in ("success", "failure"):
                population[(lab, outcome)] = 4
                train_quotas[(lab, outcome)] = 2
                val_quotas[(lab, outcome)] = 1
                for item in range(4):
                    rows.append(
                        {
                            "dataset_split": outcome,
                            "episode_index": episode_index,
                            "episode_id": f"{lab}/{outcome}/date/time-{item}",
                            "lab": lab,
                            "length": 64,
                            "scene_key": f"{lab}|building|scene-{item}",
                            "source_repo": HF_REPO,
                            "source_revision": HF_REVISION,
                            "videos/observation.image.wrist_image_left/chunk_index": 0,
                            "videos/observation.image.wrist_image_left/file_index": item // 2,
                            "videos/observation.image.wrist_image_left/from_timestamp": item * 64 / 15,
                            "videos/observation.image.wrist_image_left/to_timestamp": (item + 1) * 64 / 15,
                            "videos/observation.image.exterior_image_1_left/chunk_index": 0,
                            "videos/observation.image.exterior_image_1_left/file_index": item // 2,
                            "videos/observation.image.exterior_image_1_left/from_timestamp": item * 64 / 15,
                            "videos/observation.image.exterior_image_1_left/to_timestamp": (item + 1) * 64 / 15,
                            "videos/observation.image.exterior_image_2_left/chunk_index": 0,
                            "videos/observation.image.exterior_image_2_left/file_index": item // 2,
                            "videos/observation.image.exterior_image_2_left/from_timestamp": item * 64 / 15,
                            "videos/observation.image.exterior_image_2_left/to_timestamp": (item + 1) * 64 / 15,
                        }
                    )
                    episode_index += 1
        catalog = pd.DataFrame(rows)

        train_a, val_a, audit_a = select_scene_disjoint_manifests(
            catalog,
            seed=9,
            population=population,
            train_quotas=train_quotas,
            val_quotas=val_quotas,
        )
        train_b, val_b, audit_b = select_scene_disjoint_manifests(
            catalog,
            seed=9,
            population=population,
            train_quotas=train_quotas,
            val_quotas=val_quotas,
        )

        pd.testing.assert_frame_equal(train_a, train_b)
        pd.testing.assert_frame_equal(val_a, val_b)
        self.assertEqual(audit_a, audit_b)
        self.assertEqual(len(train_a), 52)
        self.assertEqual(len(val_a), 26)
        self.assertFalse(set(train_a.scene_key) & set(val_a.scene_key))
        self.assertEqual(audit_a["train_shards"]["window_count"], 156)
        self.assertEqual(audit_a["val_shards"]["window_count"], 52)
        self.assertGreater(audit_a["train_shards"]["unique_camera_shards"], 0)
        self.assertGreater(audit_a["val_shards"]["unique_camera_shards"], 0)


if __name__ == "__main__":
    unittest.main()
