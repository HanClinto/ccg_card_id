#!/usr/bin/env python3
"""Neural corner detector models.

TinyCornerCNN  (soft-argmax heatmap head — CONCLUSIVELY STUCK)
--------------------------------------------------------------
  ~48K parameters.  RESULT: val_cpe stuck at ~0.63 (center baseline).
  Root cause: BCE heatmap loss and soft-argmax pull in opposite directions.
  Kept for reference only.

TinyCornerCNNDirect  (direct regression — diagnostic)
------------------------------------------------------
  ~49K parameters.  Same encoder, direct regression head instead of soft-argmax.
  Confirmed model can break the center baseline; used to rule out backbone
  capacity as a limiting factor.

MobileViTCornerDetector  (direct regression — current baseline)
---------------------------------------------------------------
  ~2.3M parameters.  MobileViT-XXS backbone (ImageNet pretrained) + spatial
  pool (configurable, default 4×4) → LayerNorm → MLP → 9.
  Plateaus at val_cpe ~0.48–0.57 after 20+ epochs. pHash never moves.
  Still the best result so far; kept as the comparison baseline.

SimCCCornerDetector  (1D coordinate classification — next experiment)
---------------------------------------------------------------------
  ~2.2M parameters.  Same MobileViT-XXS backbone + SimCC head.
  Replaces the 8-float regression with 8 independent 1D softmax classifiers
  (one per corner axis), each over NUM_BINS bins at 1px resolution.
  Loss: KL divergence against Gaussian soft targets (σ ≈ 6 bins).
  Motivation: avoids the ill-conditioned L1/L2 regression landscape; each
  classifier has well-conditioned cross-entropy gradients from random init.
  Recommended by ChatGPT ET, Claude Opus, and Grok as the next step if
  direct regression plateaus.

All models return (corners (B,8), presence (B,)) — corners in canonical
TL→TR→BR→BL order, normalised [0,1].
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
    retain richer quadrant-level spatial information before regressing
    corner coordinates.

    Input must be sized so that H/32 is divisible by 4 (i.e. input
    divisible by 128).  Recommended: 384×384 (feature map 12×12 → pool
    to 4×4, 12÷4=3 ✓).  448×448 does NOT work (14÷4=3.5 ✗ on MPS).

    At 384×384 input: backbone outputs (B, 320, 12, 12) → pool to (B, 320, 4, 4)
    → flatten to (B, 5120) → head → (B, 9).

    Returns 2-tuple (corners (B,8), presence (B,)) for compatibility.

    Requires: pip install timm
    """

    def __init__(
        self,
        pretrained_backbone: bool = True,
        dropout: float = 0.3,
        pool_size: int = 4,
    ) -> None:
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError("timm is required: pip install timm")
        self.backbone = timm.create_model(
            "mobilevit_xxs", pretrained=pretrained_backbone, num_classes=0
        )
        # forward_features() returns (B, 320, H/32, W/32).
        # At 384×384 input the feature map is 12×12.
        # Valid pool_size values (must divide 12 evenly for MPS): 1,2,3,4,6,12.
        # Default 4×4 → 5120-d.  6×6 → 11520-d (finer spatial resolution).
        self._pool_size    = pool_size
        spatial_feat_dim   = 320 * pool_size * pool_size

        self.spatial_pool = nn.AdaptiveAvgPool2d(pool_size)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(spatial_feat_dim),
            nn.Linear(spatial_feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 9),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone.forward_features(x)   # (B, 320, H/32, W/32)
        feats = self.spatial_pool(feats)             # (B, 320, P, P)
        out   = self.head(feats)                     # (B, 9)
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


# ---------------------------------------------------------------------------
# SimCCCornerDetector — 1D coordinate classification
# ---------------------------------------------------------------------------

class SimCCCornerDetector(nn.Module):
    """MobileViT-XXS backbone + SimCC head for corner coordinate prediction.

    SimCC (Simple Coordinate Classification) replaces corner regression with
    two independent 1D softmax classifiers per corner — one for x, one for y —
    each distributing probability mass over NUM_BINS equally-spaced bins that
    span the normalised [0, 1] coordinate range.

    Advantages over direct regression (MobileViTCornerDetector):
      - Well-conditioned cross-entropy / KL gradients from random init
      - No ill-conditioned L1/L2 regression landscape where all outputs start near 0.5
      - Gaussian soft targets provide smooth label density over neighbouring bins
      - Argmax at inference gives ~1px resolution (at 384 bins for 384×384 input)

    Architecture:
      backbone.forward_features()  →  (B, 320, 12, 12)   [at 384×384 input]
      AdaptiveAvgPool2d(1)         →  (B, 320)
      LayerNorm + Linear(320→256) + GELU + Dropout
      ├─ 8 × Linear(256→NUM_BINS)  →  coord logits per corner-axis
      └─ Linear(256→1)             →  presence logit

    Loss (computed externally in train.py via simcc_coord_loss()):
      KL divergence between log-softmax(logits) and Gaussian soft targets
      centered at the GT bin, σ = SIGMA_BINS bins.  Only applied on
      card-present frames.

    Returns (corners (B,8), presence (B,), coord_logits list[8×(B,NUM_BINS)]).
    coord_logits are needed for the KL loss during training; at inference only
    corners and presence are used.
    """

    NUM_BINS: int = 384        # one bin per pixel at 384×384 input
    SIGMA_BINS: float = 6.0    # Gaussian soft-target width in bins (~1.5% of image)

    def __init__(
        self,
        pretrained_backbone: bool = True,
        dropout: float = 0.3,
        num_bins: int = NUM_BINS,
    ) -> None:
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError("timm is required: pip install timm")
        self.num_bins = num_bins
        self.backbone = timm.create_model(
            "mobilevit_xxs", pretrained=pretrained_backbone, num_classes=0
        )
        feat_dim = 320   # MobileViT-XXS final feature channels

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # 8 classifiers: corners 0-3, each with x and y axis
        # Order: x0, y0, x1, y1, x2, y2, x3, y3
        self.coord_heads = nn.ModuleList([nn.Linear(256, num_bins) for _ in range(8)])
        self.presence_head = nn.Linear(256, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        feats = self.backbone.forward_features(x)   # (B, 320, H/32, W/32)
        proj  = self.proj(self.pool(feats))          # (B, 256)

        coord_logits = [head(proj) for head in self.coord_heads]  # 8 × (B, num_bins)

        # Soft-argmax over bins → normalised [0, 1] coordinate
        bins   = torch.linspace(0.0, 1.0, self.num_bins, device=x.device)  # (num_bins,)
        coords = [(torch.softmax(l, dim=-1) * bins).sum(dim=-1) for l in coord_logits]
        corners  = torch.stack(coords, dim=-1)           # (B, 8)
        presence = self.presence_head(proj).squeeze(-1)  # (B,)

        return corners, presence, coord_logits


def simcc_coord_loss(
    coord_logits: list[torch.Tensor],
    gt_corners: torch.Tensor,
    mask: torch.Tensor,
    sigma_bins: float = SimCCCornerDetector.SIGMA_BINS,
) -> torch.Tensor:
    """KL divergence loss for SimCC coordinate heads.

    Args:
        coord_logits : list of 8 tensors, each (B, num_bins) — raw logits.
        gt_corners   : (B, 8) normalised [0, 1] GT corner coordinates,
                       order x0,y0,x1,y1,x2,y2,x3,y3.
        mask         : (B,) bool — True for card-present frames.
        sigma_bins   : Gaussian soft-target std-dev in bins.

    Returns scalar loss (mean over axes and present frames).
    """
    import torch.nn.functional as F

    if mask.sum() == 0:
        return torch.tensor(0.0, device=gt_corners.device, requires_grad=True)

    num_bins = coord_logits[0].shape[-1]
    bins     = torch.arange(num_bins, device=gt_corners.device).float()  # (num_bins,)

    total = torch.tensor(0.0, device=gt_corners.device)
    for i, logits in enumerate(coord_logits):
        gt_bin = gt_corners[mask, i] * (num_bins - 1)      # (N,)
        diff   = bins.unsqueeze(0) - gt_bin.unsqueeze(1)    # (N, num_bins)
        target = torch.exp(-0.5 * (diff / sigma_bins) ** 2)
        target = target / target.sum(dim=-1, keepdim=True)  # normalise → probability dist

        log_pred = F.log_softmax(logits[mask], dim=-1)       # (N, num_bins)
        total    = total + F.kl_div(log_pred, target, reduction="batchmean", log_target=False)

    return total / len(coord_logits)
