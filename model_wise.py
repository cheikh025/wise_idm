"""Three-view, vision-only inverse dynamics model for WISE.

The model consumes three synchronized RGB clips in the fixed DROID camera
order. Each of the 32 adjacent-frame transitions is encoded independently per
view, fused across views, and decoded at the aligned action step. No robot
state, language, task metadata, or Cosmos features enter the model.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


ARCH_ID = "wise_resnet50_transformer_v1"
CAMERA_ORDER = ("wrist", "left", "right")
BACKBONE_WEIGHTS = "ResNet50_Weights.IMAGENET1K_V2"


class SpatialSoftmax(nn.Module):
    """Return the expected (x, y) location of every feature channel."""

    def __init__(self, channels: int, feature_height: int, feature_width: int):
        super().__init__()
        if min(channels, feature_height, feature_width) <= 0:
            raise ValueError("spatial-softmax dimensions must be positive")
        self.channels = channels
        self.feature_height = feature_height
        self.feature_width = feature_width

        ys = torch.linspace(-1.0, 1.0, feature_height)
        xs = torch.linspace(-1.0, 1.0, feature_width)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer("grid_x", grid_x.reshape(1, 1, -1), persistent=False)
        self.register_buffer("grid_y", grid_y.reshape(1, 1, -1), persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"expected BCHW features, got shape {tuple(features.shape)}")
        batch, channels, height, width = features.shape
        expected = (self.channels, self.feature_height, self.feature_width)
        if (channels, height, width) != expected:
            raise ValueError(
                "unexpected backbone feature geometry: "
                f"got {(channels, height, width)}, expected {expected}"
            )

        weights = F.softmax(features.flatten(2).float(), dim=-1)
        x = (weights * self.grid_x.float()).sum(dim=-1)
        y = (weights * self.grid_y.float()).sum(dim=-1)
        coords = torch.stack((x, y), dim=-1).flatten(1)
        return coords.to(dtype=features.dtype).reshape(batch, channels * 2)


class ResNet50Layer3(nn.Module):
    """ImageNet ResNet-50 truncated after layer3 with a six-channel stem."""

    out_channels = 1024

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        network = resnet50(weights=weights)

        rgb_conv = network.conv1
        pair_conv = nn.Conv2d(
            6,
            rgb_conv.out_channels,
            kernel_size=rgb_conv.kernel_size,
            stride=rgb_conv.stride,
            padding=rgb_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            pair_conv.weight.copy_(rgb_conv.weight.repeat(1, 2, 1, 1) / 2.0)

        self.stem = nn.Sequential(pair_conv, network.bn1, network.relu, network.maxpool)
        self.layer1 = network.layer1
        self.layer2 = network.layer2
        self.layer3 = network.layer3

    def forward(self, pairs: torch.Tensor) -> torch.Tensor:
        x = self.stem(pairs)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


def _transformer_encoder(
    *, d_model: int, n_heads: int, ffn_dim: int, dropout: float, layers: int
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(d_model))


class WiseIDM(nn.Module):
    """EVA-style multi-view temporal IDM with aligned 32-step predictions."""

    def __init__(
        self,
        *,
        input_height: int = 128,
        input_width: int = 224,
        num_frames: int = 33,
        num_cameras: int = 3,
        action_horizon: int = 32,
        d_model: int = 512,
        n_heads: int = 8,
        cross_view_layers: int = 2,
        temporal_layers: int = 6,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        if num_cameras != len(CAMERA_ORDER):
            raise ValueError(f"WISE-IDM requires exactly {len(CAMERA_ORDER)} cameras")
        if num_frames < 2:
            raise ValueError("num_frames must be at least 2")
        if action_horizon != num_frames - 1:
            raise ValueError("action_horizon must equal num_frames - 1 for aligned prediction")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")

        self.input_height = input_height
        self.input_width = input_width
        self.num_frames = num_frames
        self.num_pairs = num_frames - 1
        self.num_cameras = num_cameras
        self.action_horizon = action_horizon
        self.d_model = d_model

        self.register_buffer(
            "rgb_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "rgb_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 1, 1, 3, 1, 1),
            persistent=False,
        )

        self.backbone = ResNet50Layer3(pretrained=pretrained_backbone)
        feature_height = (input_height + 15) // 16
        feature_width = (input_width + 15) // 16
        self.feature_geometry = (self.backbone.out_channels, feature_height, feature_width)
        self.spatial_softmax = SpatialSoftmax(*self.feature_geometry)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(self.backbone.out_channels * 2),
            nn.Linear(self.backbone.out_channels * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.camera_embedding = nn.Embedding(num_cameras, d_model)
        self.fusion_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.fusion_token, std=0.02)
        self.cross_view_encoder = _transformer_encoder(
            d_model=d_model,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            layers=cross_view_layers,
        )

        self.time_embedding = nn.Embedding(self.num_pairs, d_model)
        self.temporal_encoder = _transformer_encoder(
            d_model=d_model,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            layers=temporal_layers,
        )
        self.joint_head = nn.Linear(d_model, 7)
        self.gripper_head = nn.Linear(d_model, 1)

    def _validate_views(self, views: Sequence[torch.Tensor]) -> tuple[int, torch.device]:
        if not isinstance(views, (list, tuple)):
            raise TypeError("views must be a list or tuple in wrist, left, right order")
        if len(views) != self.num_cameras:
            raise ValueError(f"expected {self.num_cameras} views, got {len(views)}")

        expected_tail = (self.num_frames, 3, self.input_height, self.input_width)
        batch = views[0].shape[0] if views[0].ndim == 5 else -1
        device = views[0].device
        for camera_id, view in enumerate(views):
            if view.ndim != 5 or tuple(view.shape[1:]) != expected_tail:
                raise ValueError(
                    f"view {camera_id} must have shape (B, {expected_tail}), got {tuple(view.shape)}"
                )
            if view.shape[0] != batch:
                raise ValueError("all views must have the same batch size")
            if view.device != device:
                raise ValueError("all views must be on the same device")
            if not view.is_floating_point():
                raise TypeError("views must be floating-point tensors scaled to [0, 1]")
        return batch, device

    def _encode_transitions(self, views: Sequence[torch.Tensor]) -> torch.Tensor:
        batch, _ = self._validate_views(views)
        stacked = torch.stack(tuple(views), dim=1)
        normalized = (stacked - self.rgb_mean) / self.rgb_std
        pairs = torch.cat((normalized[:, :, :-1], normalized[:, :, 1:]), dim=3)
        pairs = pairs.reshape(
            batch * self.num_cameras * self.num_pairs,
            6,
            self.input_height,
            self.input_width,
        )
        features = self.backbone(pairs)
        coordinates = self.spatial_softmax(features)
        tokens = self.token_projection(coordinates)
        return tokens.reshape(batch, self.num_cameras, self.num_pairs, self.d_model).transpose(1, 2)

    def forward(self, views: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
        view_tokens = self._encode_transitions(views)
        batch = view_tokens.shape[0]
        camera_ids = torch.arange(self.num_cameras, device=view_tokens.device)
        view_tokens = view_tokens + self.camera_embedding(camera_ids).view(1, 1, self.num_cameras, -1)

        view_tokens = view_tokens.reshape(batch * self.num_pairs, self.num_cameras, self.d_model)
        fusion = self.fusion_token.expand(batch * self.num_pairs, -1, -1)
        fused = self.cross_view_encoder(torch.cat((fusion, view_tokens), dim=1))[:, 0]
        fused = fused.reshape(batch, self.num_pairs, self.d_model)

        time_ids = torch.arange(self.num_pairs, device=fused.device)
        temporal = self.temporal_encoder(fused + self.time_embedding(time_ids).unsqueeze(0))
        return {
            "joints": self.joint_head(temporal),
            "gripper_logit": self.gripper_head(temporal),
        }
