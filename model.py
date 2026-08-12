"""DreamZero-inspired inverse-dynamics model for DROID/Cosmos (M4).

Design (per research/IDM_DESIGN.md, adapted from DreamZero, NOT LIBERO):
  1. adjacent-frame pairwise early fusion as the primary motion encoder;
  2. shared CNN visual backbone operating on RAW RGB pixels -- NO VAE
     roundtrip anywhere in this model, by explicit project constraint;
  3. spatial feature grid preserved (no global average pooling);
  4. spatial features projected to Transformer tokens;
  5. camera / time / spatial-position identity embeddings added to each token;
  6. Transformer encoder over the full multi-view/multi-time token set;
  7. learned action queries (one per target action step) + Transformer decoder;
  8. one action predicted per query, in parallel;
  9. separate 7-D joint regression head and 1-D gripper head.

Spatial token count is derived from the actual backbone output geometry at
the configured image size -- not hard-coded to LIBERO's 25.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SmallCNNBackbone(nn.Module):
    """Plain conv backbone, raw pixels in, spatial feature grid out.

    No pretrained weights (kept dependency-free / no external weight
    download), no VAE, no latent-space anything -- just strided convs. Input
    channels = 6 (two adjacent RGB frames concatenated), matching the
    adjacent-frame early-fusion design.
    """

    def __init__(self, in_channels: int = 6, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width, 7, stride=2, padding=3), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1), nn.GroupNorm(8, width * 2), nn.GELU(),
            nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1), nn.GroupNorm(8, width * 4), nn.GELU(),
            nn.Conv2d(width * 4, width * 4, 3, stride=2, padding=1), nn.GroupNorm(8, width * 4), nn.GELU(),
            # Extra spatial-compression stride: IDM_DESIGN.md prefers trimming
            # compute via spatial pooling over temporal subsampling, since the
            # dataset now feeds the IDM every frame in the chunk (no temporal
            # subsampling at all) -- this keeps the total token budget in check.
            nn.Conv2d(width * 4, width * 4, 3, stride=2, padding=1), nn.GroupNorm(8, width * 4), nn.GELU(),
        )
        self.out_channels = width * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, C, H', W')


class DroidIDM(nn.Module):
    """Pure vision -- no proprioception input, ever (explicit, permanent
    project direction). Predicts actions purely from the imagined/observed
    video, matching EVA (arXiv:2603.17808) and VPT (arXiv:2206.11795), both
    of which also use no privileged state input for exactly this class of
    problem. Note: RUN_0009/RUN_0015's saved checkpoints predate this rule
    and have proprio_proj weights in their state_dict -- they no longer
    load through this class; the pre-removal model.py is preserved in git
    history (commit 045c0e3 and earlier) if ever needed again.
    """

    def __init__(
        self,
        image_size: int = 128,
        num_frames: int = 9,      # per view, adjacent pairs -> num_frames-1 pairs
        num_cameras: int = 3,
        cnn_width: int = 64,
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
        _, feat_c, feat_h, feat_w = feat.shape
        self.num_spatial_tokens = feat_h * feat_w
        self.feat_hw = (feat_h, feat_w)

        self.token_proj = nn.Linear(feat_c, d_model)
        self.camera_embed = nn.Embedding(num_cameras, d_model)
        self.time_embed = nn.Embedding(self.num_pairs, d_model)
        self.spatial_embed = nn.Embedding(self.num_spatial_tokens, d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_encoder_layers)

        self.action_queries = nn.Parameter(torch.randn(action_horizon, d_model) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, ffn_dim, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, n_decoder_layers)

        self.joint_head = nn.Linear(d_model, 7)
        self.gripper_head = nn.Linear(d_model, 1)

    def encode_view(self, frames: torch.Tensor, camera_id: int) -> torch.Tensor:
        """frames: (B, T, 3, H, W) -> tokens: (B, num_pairs*num_spatial_tokens, d_model)"""
        B, T, C, H, W = frames.shape
        pairs = torch.cat([frames[:, :-1], frames[:, 1:]], dim=2)  # (B, T-1, 6, H, W)
        pairs = pairs.reshape(B * self.num_pairs, 6, H, W)
        feat = self.backbone(pairs)  # (B*num_pairs, feat_c, fh, fw)
        _, feat_c, fh, fw = feat.shape
        feat = feat.flatten(2).transpose(1, 2)  # (B*num_pairs, fh*fw, feat_c)
        tokens = self.token_proj(feat)  # (B*num_pairs, num_spatial, d_model)
        tokens = tokens.reshape(B, self.num_pairs, self.num_spatial_tokens, -1)

        device = tokens.device
        cam_e = self.camera_embed(torch.tensor(camera_id, device=device)).view(1, 1, 1, -1)
        time_e = self.time_embed(torch.arange(self.num_pairs, device=device)).view(1, self.num_pairs, 1, -1)
        spat_e = self.spatial_embed(torch.arange(self.num_spatial_tokens, device=device)).view(1, 1, -1, tokens.shape[-1])
        tokens = tokens + cam_e + time_e + spat_e
        return tokens.reshape(B, self.num_pairs * self.num_spatial_tokens, -1)

    def forward(self, views: list[torch.Tensor]) -> dict:
        """views: list of (B, T, 3, H, W) tensors, one per camera, length must match num_cameras."""
        assert len(views) == self.num_cameras, f"expected {self.num_cameras} views, got {len(views)}"
        B = views[0].shape[0]
        view_tokens = [self.encode_view(v, cam_id) for cam_id, v in enumerate(views)]
        tokens = torch.cat(view_tokens, dim=1)

        context = self.encoder(tokens)

        queries = self.action_queries.unsqueeze(0).expand(B, -1, -1)
        decoded = self.decoder(queries, context)  # (B, action_horizon, d_model)

        joints = self.joint_head(decoded)     # (B, action_horizon, 7), standardized joint-position regression
        gripper_logit = self.gripper_head(decoded)  # (B, action_horizon, 1), RAW LOGIT -- gripper_position is
        # heavily bimodal (open/closed with rare transitions, confirmed empirically on the DROID debug subset:
        # 59% near 0, 29% near 1, 12% mid), so it is trained with BCEWithLogitsLoss against the raw [0,1] target,
        # not SmoothL1 regression in standardized space. Apply sigmoid to get a probability-like value in [0,1].
        return {"joints": joints, "gripper_logit": gripper_logit}
