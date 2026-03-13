#!/usr/bin/env python3
"""Neural corner detector: MobileViT-XXS backbone + regression head.

Architecture
------------
  Backbone : MobileViT-XXS (from timm), pretrained on ImageNet-1k by default.
             Can be seeded from a card-ID ArcFace checkpoint (loads only the
             backbone weights, ignoring the projection/ArcFace head).
  Head     : Global average pool → LayerNorm → Linear(384, 256) → GELU →
             Dropout(0.3) → Linear(256, 9)
             Output slice [:8] → sigmoid → 4 corner (x,y) pairs (TL,TR,BR,BL)
             Output slice [8]  → sigmoid → card_presence probability

Input: 224×224 RGB, normalized with ImageNet mean/std.

The two-output design (corners + presence) lets the network learn to say
"I see no card" rather than hallucinating corners for empty frames.

Loss
----
  L_presence = BCEWithLogitsLoss(pred_presence, label_presence)
  L_corners  = SmoothL1Loss(pred_corners, true_corners)  # only on positives
  L_total    = L_presence + lambda_corners * L_corners    # lambda_corners = 5.0
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:
    _TIMM_AVAILABLE = False


class NeuralCornerDetector(nn.Module):
    """MobileViT-XXS backbone + corner regression head."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    INPUT_SIZE    = 224

    def __init__(self, pretrained_backbone: bool = True, dropout: float = 0.3) -> None:
        super().__init__()
        if not _TIMM_AVAILABLE:
            raise ImportError("timm is required: pip install timm")

        self.backbone = timm.create_model(
            "mobilevit_xxs", pretrained=pretrained_backbone, num_classes=0
        )
        feat_dim = self.backbone.num_features  # 384 for mobilevit_xxs

        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 9),   # 8 corner coords + 1 presence logit
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, 224, 224) float32, ImageNet-normalized.

        Returns:
            corners:  (B, 8) — sigmoid-activated corner coordinates in [0, 1]
            presence: (B,)   — raw logit (apply sigmoid for probability)
        """
        feats = self.backbone(x)          # (B, feat_dim)
        out   = self.head(feats)          # (B, 9)
        corners  = torch.sigmoid(out[:, :8])
        presence = out[:, 8]              # raw logit
        return corners, presence

    def load_card_id_checkpoint(self, ckpt_path: Path | str) -> None:
        """Seed backbone weights from a card-ID ArcFace checkpoint.

        The checkpoint format is the one produced by 04_build/mobilevit_xxs/02_train.py:
          {"model": state_dict, "optimizer": ..., "epoch": ...}

        Only keys starting with "backbone." are loaded; projection/ArcFace
        head weights are intentionally ignored.
        """
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
