#!/usr/bin/env python3
"""Neural corner detector models.

Two architectures are provided — choose based on your deployment target:

TinyCornerCNN  (recommended for most uses)
--------------------------------------------
  ~48K parameters, 0.2 MB fp32, ~50 KB int8.
  Pure depthwise-separable CNN — no attention, no external dependencies.
  Input: 448×448.  Encoder output: 28×28 spatial maps.
  Corners extracted via soft-argmax over 4-channel heatmaps — one channel
  per corner (TL, TR, BR, BL).  Spatial precision: 448/16 = 28px grid,
  i.e. ~16px resolution at the input scale.

  Heatmap output naturally handles occlusion: an occluded corner produces
  a low-confidence, diffuse peak rather than a confident wrong prediction.
  Per-corner confidence is available from heatmap peak values.

MobileViTCornerDetector  (ablation / accuracy ceiling)
--------------------------------------------------------
  951K parameters, 3.6 MB fp32.  Uses the same MobileViT-XXS backbone as the
  card-ID model, so backbone weights can be seeded from a pretrained ArcFace
  checkpoint.  Still uses direct global regression (not updated to heatmap).

Both models return corners in canonical TL→TR→BR→BL order (sorted by
sort_corners_canonical convention — CW winding, shortest edge at 0→1).

Loss (TinyCornerCNN):
  L = BCEWithLogitsLoss(presence)
    + λ_c * SmoothL1Loss(corners[positives], direct canonical order)
    + λ_h * BCEWithLogitsLoss(heatmaps[positives], gaussian_targets)
  λ_c = 5.0, λ_h = 10.0 by default.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE    = 448
HEATMAP_SIZE  = 28   # INPUT_SIZE / 16  (4× stride-2 layers in encoder)


# ---------------------------------------------------------------------------
# Soft-argmax and Gaussian heatmap helpers
# ---------------------------------------------------------------------------

def _soft_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    """Convert spatial heatmaps to normalised corner coordinates.

    Args:
        heatmaps: (B, 4, H, W) raw logits — one channel per corner.

    Returns:
        (B, 4, 2) normalised (x, y) coordinates in [0, 1] (pixel-centre
        convention: the centre of the top-left pixel maps to (0.5/W, 0.5/H)).
    """
    B, C, H, W = heatmaps.shape
    flat    = heatmaps.view(B, C, -1)                         # (B, 4, H*W)
    weights = torch.softmax(flat, dim=2)                      # (B, 4, H*W)

    # Pixel-centre grid in [0, 1]
    xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=heatmaps.device)
    ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=heatmaps.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')   # (H, W)
    grid_x = grid_x.reshape(1, 1, -1)                        # (1, 1, H*W)
    grid_y = grid_y.reshape(1, 1, -1)

    cx = (weights * grid_x).sum(dim=2)                        # (B, 4)
    cy = (weights * grid_y).sum(dim=2)
    return torch.stack([cx, cy], dim=2)                       # (B, 4, 2)


def make_gaussian_heatmaps(
    corners_norm: torch.Tensor,
    H: int = HEATMAP_SIZE,
    W: int = HEATMAP_SIZE,
    sigma: float = 2.0,
) -> torch.Tensor:
    """Generate Gaussian blob heatmaps for corner supervision.

    Args:
        corners_norm: (N, 4, 2) GT corners in normalised [0, 1] (x, y).
        H, W:         Heatmap spatial dimensions (default HEATMAP_SIZE×HEATMAP_SIZE).
        sigma:        Gaussian std-dev in heatmap pixels (default 2.0).

    Returns:
        (N, 4, H, W) float32 heatmaps in [0, 1] with Gaussian peaks at
        each corner position.  Used as targets for BCEWithLogitsLoss on
        the raw heatmap logits.
    """
    N, C, _ = corners_norm.shape
    device  = corners_norm.device

    ys = torch.arange(H, device=device).float()
    xs = torch.arange(W, device=device).float()
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')   # (H, W)

    # Normalised [0,1] → heatmap pixel position
    px = corners_norm[:, :, 0] * W   # (N, 4)
    py = corners_norm[:, :, 1] * H   # (N, 4)

    # Broadcast to (N, 4, H, W)
    dx = grid_x.unsqueeze(0).unsqueeze(0) - px.unsqueeze(2).unsqueeze(3)
    dy = grid_y.unsqueeze(0).unsqueeze(0) - py.unsqueeze(2).unsqueeze(3)

    return torch.exp(-(dx ** 2 + dy ** 2) / (2.0 * sigma ** 2))


# ---------------------------------------------------------------------------
# TinyCornerCNN — purpose-built, edge-deployable
# ---------------------------------------------------------------------------

def _dw_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Depthwise-separable conv block: DW → PW, with BN + ReLU6."""
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch), nn.ReLU6(inplace=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU6(inplace=True),
    )


class TinyCornerCNN(nn.Module):
    """~48K-parameter depthwise-separable CNN for card corner detection.

    Input:  (B, 3, 448, 448) float32, ImageNet-normalised.
    Output:
        corners   (B, 8)          — flat (x0,y0,…,x3,y3) TL/TR/BR/BL,
                                    derived via soft-argmax, clamped [0,1]
                                    at inference.
        presence  (B,)            — raw presence logit.
        heatmaps  (B, 4, 28, 28)  — per-corner spatial heatmaps (raw logits,
                                    before softmax).  Channel 0=TL, 1=TR,
                                    2=BR, 3=BL.  Peak confidence available
                                    as sigmoid(heatmaps).amax(dim=(-2,-1)).

    The encoder (4× stride-2, total stride 16) produces 28×28 feature maps
    at 448×448 input, giving ~16px spatial precision per grid cell.
    Soft-argmax makes corner extraction fully differentiable.
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # 448→224, 3→16  (standard conv for first layer)
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            # 224→112, 16→32
            _dw_block(16, 32, stride=2),
            # 112→56, 32→64
            _dw_block(32, 64, stride=2),
            _dw_block(64, 64, stride=1),
            # 56→28, 64→128
            _dw_block(64, 128, stride=2),
            _dw_block(128, 128, stride=1),
        )
        # → (B, 128, 28, 28) at 448×448 input

        # Corner heatmap head: one output channel per corner
        self.corner_head = nn.Sequential(
            nn.Conv2d(128, 32, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU6(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(32, 4, 1),
        )

        # Presence head: global pooling → scalar logit
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats    = self.encoder(x)                                     # (B, 128, 28, 28)
        heatmaps = self.corner_head(feats)                             # (B, 4, 28, 28)
        presence = self.presence_head(feats).squeeze(1)                # (B,)
        corners  = _soft_argmax(heatmaps).reshape(x.shape[0], 8)      # (B, 8)
        return corners, presence, heatmaps


# ---------------------------------------------------------------------------
# MobileViTCornerDetector — accuracy ceiling / transfer-learning ablation
# ---------------------------------------------------------------------------

class MobileViTCornerDetector(nn.Module):
    """MobileViT-XXS backbone + corner regression head.

    951K parameters, 3.6 MB fp32.  Same backbone as the card-ID model —
    backbone weights can be seeded from a pretrained ArcFace checkpoint via
    load_card_id_checkpoint().  Still uses global direct regression (not
    updated to heatmap architecture).

    Returns 2-tuple (corners (B,8), presence (B,)) for compatibility.

    Requires: pip install timm
    """

    def __init__(self, pretrained_backbone: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError("timm is required: pip install timm")
        self.backbone = timm.create_model(
            "mobilevit_xxs", pretrained=pretrained_backbone, num_classes=0
        )
        feat_dim = self.backbone.num_features  # 384

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 9),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        out   = self.head(feats)
        return out[:, :8], out[:, 8]

    def load_card_id_checkpoint(self, ckpt_path: Path | str) -> None:
        """Seed backbone from a card-ID ArcFace checkpoint (backbone.* keys only)."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)
        backbone_state = {
            k[len("backbone."):]: v
            for k, v in state.items()
            if k.startswith("backbone.")
        }
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        print(f"  Loaded backbone from {Path(ckpt_path).name}")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
