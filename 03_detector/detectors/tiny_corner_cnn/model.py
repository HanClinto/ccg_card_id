#!/usr/bin/env python3
"""Neural corner detector models.

Two architectures are provided — choose based on your deployment target:

TinyCornerCNN  (recommended for most uses)
--------------------------------------------
  46K parameters, 0.2 MB fp32, ~40 KB int8.
  Pure depthwise-separable CNN — no attention, no external dependencies.
  Runs at ~15–20 FPS on a Raspberry Pi 4; small enough to embed in a webpage.

  Corner detection is a *geometric* task (find 4 edge/perspective-defined
  points), not a *semantic* one. A CNN that learns edge detectors and spatial
  gradients is architecturally well-matched. Attention over semantic tokens —
  as in MobileViT — adds overhead without benefit for this task.

MobileViTCornerDetector  (ablation / accuracy ceiling)
--------------------------------------------------------
  951K parameters, 3.6 MB fp32.  Uses the same MobileViT-XXS backbone as the
  card-ID model, so backbone weights can be seeded from a pretrained ArcFace
  checkpoint.  Useful to verify that TinyCornerCNN isn't leaving accuracy on
  the table, but overkill for production deployment.

Both models share the same interface:
  forward(x) → (corners: (B,8) sigmoid, presence_logit: (B,) raw)
  corners are TL, TR, BR, BL in normalized [0,1] (x,y) order.

Loss (same for both):
  L = BCEWithLogitsLoss(presence) + λ * SmoothL1Loss(corners[positives only])
  λ = 5.0 by default.
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
INPUT_SIZE    = 224


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
    """46K-parameter depthwise-separable CNN for card corner regression.

    Input : (B, 3, 224, 224) float32, ImageNet-normalized.
    Output: corners  (B, 8)  — sigmoid, TL/TR/BR/BL (x,y) in [0, 1]
            presence (B,)    — raw logit (apply sigmoid for probability)

    ~0.2 MB fp32 / ~40 KB int8. Runs at ≈3 ms on a modern laptop CPU;
    estimated 50–70 ms on Raspberry Pi 4 (≈15–20 FPS).
    No external dependencies beyond PyTorch.
    """

    def __init__(self, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # 224 → 112, 3 → 16 (standard conv for first layer)
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU6(inplace=True),
            # 112 → 56, 16 → 32
            _dw_block(16, 32, stride=2),
            # 56 → 28, 32 → 64
            _dw_block(32, 64, stride=2),
            _dw_block(64, 64, stride=1),
            # 28 → 14, 64 → 128
            _dw_block(64, 128, stride=2),
            _dw_block(128, 128, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 9),  # 8 corner coords + 1 presence logit
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.head(self.pool(self.encoder(x)))
        return torch.sigmoid(out[:, :8]), out[:, 8]


# ---------------------------------------------------------------------------
# MobileViTCornerDetector — accuracy ceiling / transfer-learning ablation
# ---------------------------------------------------------------------------

class MobileViTCornerDetector(nn.Module):
    """MobileViT-XXS backbone + corner regression head.

    951K parameters, 3.6 MB fp32.  Same backbone as the card-ID model —
    backbone weights can be seeded from a pretrained ArcFace checkpoint via
    load_card_id_checkpoint().  Use this to check whether TinyCornerCNN is
    leaving accuracy on the table before committing to the smaller model.

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
        return torch.sigmoid(out[:, :8]), out[:, 8]

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
