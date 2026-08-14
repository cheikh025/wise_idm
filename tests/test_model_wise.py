import unittest

import torch

from model_wise import ResNet50Layer3, SpatialSoftmax, WiseIDM


class WiseIDMTest(unittest.TestCase):
    def test_spatial_softmax_uses_every_input_channel(self):
        module = SpatialSoftmax(channels=2, feature_height=2, feature_width=3)
        features = torch.full((1, 2, 2, 3), -20.0)
        features[0, 0, 1, 2] = 20.0
        features[0, 1, 0, 0] = 20.0

        coordinates = module(features)

        self.assertEqual(coordinates.shape, (1, 4))
        self.assertTrue(
            torch.allclose(coordinates, torch.tensor([[1.0, 1.0, -1.0, -1.0]]), atol=1e-5)
        )

    def test_resnet_pair_stem_repeats_and_averages_rgb_weights(self):
        backbone = ResNet50Layer3(pretrained=False)
        weights = backbone.stem[0].weight.detach()

        self.assertEqual(weights.shape[1], 6)
        self.assertTrue(torch.equal(weights[:, :3], weights[:, 3:]))

    def test_wise_idm_predicts_one_action_per_transition(self):
        model = WiseIDM(
            input_height=32,
            input_width=48,
            num_frames=3,
            action_horizon=2,
            d_model=32,
            n_heads=4,
            cross_view_layers=1,
            temporal_layers=1,
            ffn_dim=64,
            dropout=0.0,
            pretrained_backbone=False,
        )
        views = [torch.rand(1, 3, 3, 32, 48) for _ in range(3)]

        output = model(views)
        loss = output["joints"].square().mean() + output["gripper_logit"].square().mean()
        loss.backward()

        self.assertEqual(output["joints"].shape, (1, 2, 7))
        self.assertEqual(output["gripper_logit"].shape, (1, 2, 1))
        self.assertIsNotNone(model.joint_head.weight.grad)
        self.assertFalse(hasattr(model, "action_queries"))
        self.assertFalse(hasattr(model, "decoder"))

    def test_wise_idm_rejects_wrong_frame_count(self):
        model = WiseIDM(
            input_height=32,
            input_width=48,
            num_frames=3,
            action_horizon=2,
            d_model=32,
            n_heads=4,
            cross_view_layers=1,
            temporal_layers=1,
            ffn_dim=64,
            pretrained_backbone=False,
        )
        invalid_views = [torch.rand(1, 2, 3, 32, 48) for _ in range(3)]

        with self.assertRaisesRegex(ValueError, "must have shape"):
            model(invalid_views)


if __name__ == "__main__":
    unittest.main()
