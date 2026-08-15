"""Experimental single-panel IDM variant.

Not the frozen production architecture (`model_wise.py`, `ARCH_ID =
wise_resnet50_transformer_v1`). This is an isolated architecture experiment:
instead of three independently-cropped camera clips fused by a cross-view
Transformer, the wrist/left/right views are assembled into one composite
panel that exactly matches the geometry Cosmos itself decodes
(`vision.PANEL_HEIGHT/PANEL_WIDTH/PANEL_SEAM`: 528x640, split at row 360), and
a single shared 6-channel ResNet50-through-layer3 pass reads the whole panel
at once. There is consequently no camera-identity embedding, no fusion token,
and no cross-view Transformer - camera identity is carried by pixel position
in the panel instead of by a learned tag.

SpatialSoftmax is deliberately not reused here: it collapses every channel to
a single (x, y) coordinate, discarding magnitude/appearance information (e.g.
the gripper's visible state indicator) that a coarse pooled feature grid
retains. A 1x1 conv compresses channels before pooling so the flatten+project
step stays a sane parameter count (~10M) instead of the ~700M a naive flatten
of the full 1024x33x40 feature map would need.

Known, unresolved architectural risk (not fixed by anything in this file):
mid/late-layer convolutions have a receptive field of roughly 200px by
layer3, so features within about that distance of the wrist/exterior or
left/right seams are computed from pixels belonging to two physically
unrelated cameras. Validate with a small pilot before trusting this at scale.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model_wise import ResNet50Layer3, _transformer_encoder


ARCH_ID = "wise_composite_panel_resnet50_v1"
BACKBONE_WEIGHTS = "ResNet50_Weights.IMAGENET1K_V2"


class CompositeIDM(nn.Module):
    """Single-panel IDM: one shared backbone pass over the whole composite."""

    def __init__(
        self,
        *,
        panel_height: int = 528,
        panel_width: int = 640,
        num_frames: int = 33,
        action_horizon: int = 32,
        compressed_channels: int = 256,
        pool_height: int = 8,
        pool_width: int = 10,
        d_model: int = 512,
        n_heads: int = 8,
        temporal_layers: int = 6,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        if num_frames < 2:
            raise ValueError("num_frames must be at least 2")
        if action_horizon != num_frames - 1:
            raise ValueError("action_horizon must equal num_frames - 1 for aligned prediction")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")

        self.panel_height = panel_height
        self.panel_width = panel_width
        self.num_frames = num_frames
        self.num_pairs = num_frames - 1
        self.action_horizon = action_horizon
        self.d_model = d_model

        # Rank 3 (3,1,1), not the naive rank-5 (1,1,3,1,1): nn.Module._apply
        # applies memory_format conversion to any 4-D or 5-D buffer/parameter
        # it finds, and channels_last strictly requires 4-D NCHW, so a 5-D
        # buffer here crashes model.to(memory_format=torch.channels_last). A
        # 6-D buffer (matching model_wise.py's own dodge) avoids the crash but
        # silently changes the broadcast result against a 5-D (B,T,3,H,W)
        # panel (verified: produces a spurious extra leading dim). Rank 3
        # sidesteps the (4,5) conversion check entirely while still
        # broadcasting identically to the original (1,1,3,1,1) shape, because
        # broadcasting aligns missing leading dims as size 1 regardless.
        self.register_buffer("rgb_mean", torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1), persistent=False)
        self.register_buffer("rgb_std", torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1), persistent=False)

        self.backbone = ResNet50Layer3(pretrained=pretrained_backbone)
        feature_height = (panel_height + 15) // 16
        feature_width = (panel_width + 15) // 16
        self.feature_geometry = (self.backbone.out_channels, feature_height, feature_width)

        self.channel_compress = nn.Conv2d(self.backbone.out_channels, compressed_channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((pool_height, pool_width))
        pooled_dim = compressed_channels * pool_height * pool_width
        self.token_projection = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        self.time_embedding = nn.Embedding(self.num_pairs, d_model)
        self.temporal_encoder = _transformer_encoder(
            d_model=d_model, n_heads=n_heads, ffn_dim=ffn_dim, dropout=dropout, layers=temporal_layers
        )
        self.joint_head = nn.Linear(d_model, 7)
        self.gripper_head = nn.Linear(d_model, 1)

    def _validate_panel(self, panel: torch.Tensor) -> int:
        expected_tail = (self.num_frames, 3, self.panel_height, self.panel_width)
        if panel.ndim != 5 or tuple(panel.shape[1:]) != expected_tail:
            raise ValueError(f"panel must have shape (B, {expected_tail}), got {tuple(panel.shape)}")
        if not panel.is_floating_point():
            raise TypeError("panel must be a floating-point tensor scaled to [0, 1]")
        return panel.shape[0]

    def forward(self, views) -> dict[str, torch.Tensor]:
        if not isinstance(views, (list, tuple)) or len(views) != 1:
            raise ValueError("CompositeIDM takes a single-element sequence: [panel]")
        panel = views[0]
        batch = self._validate_panel(panel)

        normalized = (panel - self.rgb_mean) / self.rgb_std
        pairs = torch.cat((normalized[:, :-1], normalized[:, 1:]), dim=2)
        pairs = pairs.reshape(batch * self.num_pairs, 6, self.panel_height, self.panel_width)

        features = self.backbone(pairs)
        compressed = self.channel_compress(features)
        pooled = self.pool(compressed).flatten(1)
        tokens = self.token_projection(pooled).reshape(batch, self.num_pairs, self.d_model)

        time_ids = torch.arange(self.num_pairs, device=tokens.device)
        temporal = self.temporal_encoder(tokens + self.time_embedding(time_ids).unsqueeze(0))
        return {
            "joints": self.joint_head(temporal),
            "gripper_logit": self.gripper_head(temporal),
        }
