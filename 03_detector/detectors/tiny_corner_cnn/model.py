#!/usr/bin/env python3
"""Neural corner detector models.

Three architectures are provided:

TinyCornerCNN  (soft-argmax heatmap head — CONCLUSIVELY STUCK)
--------------------------------------------------------------
  ~48K parameters, 0.2 MB fp32.  Heatmap head + soft-argmax.
  RESULT: val_cpe permanently stuck at ~0.63 (image-center baseline) over
  40+ epochs. Root cause: BCE loss and soft-argmax pull in opposite directions
  (see v0.3 experiment notes in README). Do not use for new training runs.
  Kept for reference and for the quick diagnostic below.

TinyCornerCNNDirect  (direct regression — diagnostic run)
----------------------------------------------------------
  ~49K parameters, 0.2 MB fp32.  Same encoder as TinyCornerCNN, but replaces
  the heatmap/soft-argmax head with AdaptiveAvgPool2d(2) → LayerNorm → MLP → 9.
  Purpose: isolate whether soft-argmax was the *only* problem, or whether the
  48K backbone also lacks capacity. If this breaks the center baseline, the
  architecture was fine and soft-argmax was the sole culprit.

MobileViTCornerDetector  (primary path forward)
------------------------------------------------
  951K parameters, 3.6 MB fp32.  MobileViT-XXS backbone (ImageNet pretrained)
  + spatial-aware direct regression head (AdaptiveAvgPool2d(4) → MLP → 9).
  Seed: ImageNet pretrained weights only — do NOT seed from ArcFace card-ID
  checkpoints, which are trained on aligned card internals and focus on fine
  artwork/typography details rather than card borders.
  Pool is 4×4 (5120-d) to preserve more spatial information than the earlier
  2×2 design (1280-d) which capped localization at ±25% of image per quadrant.

All models return corners in canonical TL→TR→BR→BL order.

Loss (TinyCornerCNN / TinyCornerCNNDirect):
  TinyCornerCNN:      L = BCE(presence) + λ_c * SmoothL1(corners) + λ_h * BCE(heatmaps)
  TinyCornerCNNDirect: L = BCE(presence) + λ_c * SmoothL1(corners)  [no heatmap term]
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
# TinyCornerCNNDirect — same encoder, direct regression head (diagnostic)
# ---------------------------------------------------------------------------

class TinyCornerCNNDirect(nn.Module):
    """~49K-parameter direct-regression corner detector.

    Same depthwise-separable CNN encoder as TinyCornerCNN, but replaces the
    soft-argmax heatmap head with a direct spatial regression head.

    Purpose: diagnostic run to isolate whether soft-argmax gradient collapse
    was the *only* failure mode, or whether the 48K encoder also lacks capacity.
    If this breaks the center-prediction baseline, the soft-argmax was the
    sole culprit. If it also gets stuck, backbone capacity is also a factor.

    Input:  (B, 3, 448, 448) float32, ImageNet-normalised.
    Output: (corners (B, 8), presence (B,))
      corners:  flat (x0,y0,…,x3,y3) TL/TR/BR/BL in [0, 1]
      presence: raw logit (B,)
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            _dw_block(16, 32, stride=2),
            _dw_block(32, 64, stride=2),
            _dw_block(64, 64, stride=1),
            _dw_block(64, 128, stride=2),
            _dw_block(128, 128, stride=1),
        )
        # → (B, 128, 28, 28) at 448×448 input

        # Pool to 2×2 to preserve coarse quadrant structure (same design as
        # MobileViTCornerDetector) before flattening to a fixed-size vector.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),   # (B, 128, 2, 2)
            nn.Flatten(),              # (B, 512)
            nn.LayerNorm(512),
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 9),          # 8 corner coords + 1 presence logit
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.encoder(x)
        out   = self.head(feats)       # (B, 9)
        return out[:, :8], out[:, 8]


# ---------------------------------------------------------------------------
# MobileViTCornerDetector — primary path forward
# ---------------------------------------------------------------------------

class MobileViTCornerDetector(nn.Module):
    """MobileViT-XXS backbone + spatial-aware direct regression head.

    951K parameters backbone, 3.6 MB fp32.  ImageNet pretrained weights only
    — do NOT seed from ArcFace card-ID checkpoints, which are trained on
    aligned card internals and encode fine artwork/typography features rather
    than the card-border features needed here.

    Uses forward_features() to obtain the final spatial feature map
    (B, 320, H/32, W/32) before global pooling, then pools to 4×4 to
    retain coarse spatial information before regressing corner coordinates.
    The 4×4 pool (5120-d) preserves more spatial resolution than the earlier
    2×2 design (1280-d), which capped localization precision at roughly ±25%
    of the image dimension per quadrant.

    At 448×448 input: backbone outputs (B, 320, 14, 14) → pool to (B, 320, 4, 4)
    → flatten to (B, 5120) → head → (B, 9).

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
        # forward_features() returns (B, 320, H/32, W/32).
        # Pool to 4×4 — 14×14 (448/32) divides evenly, captures 16-region
        # spatial structure for improved localization resolution.
        # Note: MPS requires input size divisible by output size for adaptive pool.
        spatial_feat_dim = 320 * 4 * 4  # 5120

        self.spatial_pool = nn.AdaptiveAvgPool2d(4)
        self.head = nn.Sequential(
            nn.Flatten(),                             # (B, 5120)
            nn.LayerNorm(spatial_feat_dim),
            nn.Linear(spatial_feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 9),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone.forward_features(x)  # (B, 320, H/32, W/32)
        feats = self.spatial_pool(feats)            # (B, 320, 4, 4)
        out   = self.head(feats)                    # (B, 9)
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
