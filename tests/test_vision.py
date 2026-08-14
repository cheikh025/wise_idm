import unittest

import numpy as np

from vision import letterbox_rgb, resize_rgb_stretch, split_cosmos_panel


class VisionGeometryTest(unittest.TestCase):
    def test_split_cosmos_panel_uses_fixed_geometry(self):
        dream = np.zeros((33, 528, 640, 3), dtype=np.uint8)
        dream[:, :360] = 1
        dream[:, 360:, :320] = 2
        dream[:, 360:, 320:] = 3

        views = split_cosmos_panel(dream)

        self.assertEqual(views["wrist"].shape, (33, 360, 640, 3))
        self.assertEqual(views["left"].shape, (33, 168, 320, 3))
        self.assertEqual(views["right"].shape, (33, 168, 320, 3))
        self.assertTrue(np.all(views["wrist"] == 1))
        self.assertTrue(np.all(views["left"] == 2))
        self.assertTrue(np.all(views["right"] == 3))

    def test_split_cosmos_panel_rejects_wrong_frame_count(self):
        with self.assertRaisesRegex(ValueError, "expected 33 frames"):
            split_cosmos_panel(np.zeros((32, 54, 64, 3), dtype=np.uint8))

    def test_split_cosmos_panel_rejects_wrong_geometry(self):
        with self.assertRaisesRegex(ValueError, "must be 528x640"):
            split_cosmos_panel(np.zeros((33, 60, 60, 3), dtype=np.uint8))

    def test_letterbox_preserves_aspect_ratio_without_crop(self):
        source = np.full((2, 36, 64, 3), 255, dtype=np.uint8)

        resized = letterbox_rgb(source, height=128, width=224)

        self.assertEqual(resized.shape, (2, 128, 224, 3))
        self.assertTrue(np.all(resized[:, 0] == 0))
        self.assertTrue(np.all(resized[:, 1:-1] == 255))
        self.assertTrue(np.all(resized[:, -1] == 0))

    def test_legacy_stretch_has_no_letterbox_padding(self):
        frames = np.full((1, 8, 16, 3), 255, dtype=np.uint8)
        resized = resize_rgb_stretch(frames, 8, 8)

        self.assertEqual(resized.shape, (1, 8, 8, 3))
        self.assertTrue((resized == 255).all())


if __name__ == "__main__":
    unittest.main()
