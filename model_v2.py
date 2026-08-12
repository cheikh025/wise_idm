"""DROID IDM v2 -- spatial-softmax token compression, vision-only (M4 redesign).

Motivated by RUN_0015: the v1 architecture (model.py) flattens each camera-
pair's spatial feature grid into one token per grid cell (32 pairs x 3
cameras x 16 cells = 1537 tokens), which makes the transformer encoder's
self-attention the dominant, expensive cost. At 5000-episode scale, even a
properly matched 30-epoch/DDP training run plateaued worse than the
500-episode v1 checkpoint (0.117 vs 0.069) -- a capacity/token-efficiency
problem, not an under-training one.

v2 keeps the same overall skeleton (CNN backbone, adjacent-pair fusion,
global non-causal transformer encoder, learned-query decoder, joint+gripper
heads) but compresses each camera-pair's spatial feature grid down to ONE
token via spatial softmax (Finn et al. 2016 deep spatial autoencoders;
adopted by EVA, arXiv:2603.17808, whose own ablation shows spatial softmax
clearly beats flatten/global-pooling for this exact kind of visuomotor
keypoint task -- 98.6% vs 77.4% test accuracy). Token count drops from 1537
to 96 (32 pairs x 3 cameras), an ~256x reduction in attention cost, while
keeping full non-causal cross-camera/cross-time attention (VPT,
arXiv:2206.11795, validates that this whole-clip global-attention paradigm
works well at scale, provided each frame/pair is compressed to a small
number of tokens before attention runs -- VPT compresses to one token per
frame; we compress to one token per camera-pair).

Deliberately NOT added: VPT's wider local-temporal-conv (5-frame window)
before the CNN. Reasoning: VPT needs it because its global stage only ever
sees one token per whole frame, with no other path to recover local motion.
Our pair-tokens are already time-indexed and mutually attendable, so
pair-to-pair attention gives a comparable path to cross-step motion info
that a wider local window would otherwise provide -- revisit only if a
specific acceleration/deceleration-sensitive failure mode shows up.

Deliberately NOT included: the proprioception token (per explicit project
direction) -- this is purely vision-only, matching both EVA and VPT, which
also predict actions/keys from pixels alone with no privileged state input.

No VAE roundtrip anywhere (unchanged hard constraint) -- SmallCNNBackbone
operates on raw RGB pixels exactly as in v1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SmallCNNBackbone


class SpatialSoftmax(nn.Module):
    """Reduce a (B, C, H, W) feature map to (B, C, 2) (x, y) keypoint
    coordinates per channel, via a 1x1 conv to `num_keypoints` channels
    followed by a softmax-weighted spatial expectation. Coordinates are
    normalized to [-1, 1] in both axes.
    """

    def __init__(self, in_channels: int, num_keypoints: int = 48):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)
        self.num_keypoints = num_keypoints
        self._grid_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _grid(self, h: int, w: int, device) -> torch.Tensor:
        key = (h, w)
        if key not in self._grid_cache or self._grid_cache[key].device != device:
            ys = torch.linspace(-1.0, 1.0, h, device=device)
            xs = torch.linspace(-1.0, 1.0, w, device=device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            self._grid_cache[key] = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (H*W, 2)
        return self._grid_cache[key]

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, _, H, W = feat.shape
        x = self.reduce(feat)  # (B, K, H, W)
        x = x.reshape(B, self.num_keypoints, H * W)
        weights = F.softmax(x, dim=-1)  # (B, K, H*W)
        grid = self._grid(H, W, feat.device)  # (H*W, 2)
        coords = torch.einsum("bkn,nd->bkd", weights, grid)  # (B, K, 2)
        return coords.reshape(B, self.num_keypoints * 2)


class DroidIDMv2(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        num_frames: int = 33,
        num_cameras: int = 3,
        cnn_width: int = 64,
        num_keypoints: int = 48,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        action_horizon: int = 32,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_pairs = num_frames - 1
        self.num_cameras = num_cameras
        self.action_horizon = action_horizon

        self.backbone = SmallCNNBackbone(in_channels=6, width=cnn_width)
        with torch.no_grad():
            dummy = torch.zeros(1, 6, image_size, image_size)
            feat = self.backbone(dummy)
        self.spatial_softmax = SpatialSoftmax(feat.shape[1], num_keypoints=num_keypoints)

        self.token_proj = nn.Linear(num_keypoints * 2, d_model)
        self.camera_embed = nn.Embedding(num_cameras, d_model)
        self.time_embed = nn.Embedding(self.num_pairs, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_encoder_layers)

        self.action_queries = nn.Parameter(torch.randn(action_horizon, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, ffn_dim, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, n_decoder_layers)

        self.joint_head = nn.Linear(d_model, 7)
        self.gripper_head = nn.Linear(d_model, 1)

    def encode_view(self, frames: torch.Tensor, camera_id: int) -> torch.Tensor:
        """frames: (B, T, 3, H, W) -> tokens: (B, num_pairs, d_model), one
        compressed token per adjacent-frame pair (vs v1's num_pairs*16)."""
        B, T, C, H, W = frames.shape
        pairs = torch.cat([frames[:, :-1], frames[:, 1:]], dim=2)  # (B, T-1, 6, H, W)
        pairs = pairs.reshape(B * self.num_pairs, 6, H, W)
        feat = self.backbone(pairs)  # (B*num_pairs, feat_c, fh, fw)
        coords = self.spatial_softmax(feat)  # (B*num_pairs, num_keypoints*2)
        tokens = self.token_proj(coords).reshape(B, self.num_pairs, -1)  # (B, num_pairs, d_model)

        device = tokens.device
        cam_e = self.camera_embed(torch.tensor(camera_id, device=device)).view(1, 1, -1)
        time_e = self.time_embed(torch.arange(self.num_pairs, device=device)).view(1, self.num_pairs, -1)
        return tokens + cam_e + time_e

    def forward(self, wrist: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> dict:
        """No proprio argument -- pure vision-only, per explicit project direction."""
        B = wrist.shape[0]
        tokens = torch.cat([
            self.encode_view(wrist, 0),
            self.encode_view(left, 1),
            self.encode_view(right, 2),
        ], dim=1)  # (B, 3*num_pairs, d_model)  -- e.g. (B, 96, d_model)

        context = self.encoder(tokens)

        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(queries, context)  # (B, action_horizon, d_model)

        joints = self.joint_head(decoded)
        gripper_logit = self.gripper_head(decoded)
        return {"joints": joints, "gripper_logit": gripper_logit}
