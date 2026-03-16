from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    name = name.lower()
    if name == "tinyvit":
        try:
            import timm
        except Exception as e:
            raise RuntimeError("tinyvit backbone requires timm. Install with: pip install timm") from e
        model = timm.create_model("tiny_vit_5m_224", pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = model.num_features
        return model, int(feat_dim)

    if name == "mobilevit_xxs":
        try:
            import timm
        except Exception as e:
            raise RuntimeError("mobilevit_xxs backbone requires timm. Install with: pip install timm") from e
        model = timm.create_model("mobilevit_xxs", pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = model.num_features
        return model, int(feat_dim)

    if name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        return base, int(feat_dim)

    raise ValueError(f"Unknown backbone: {name}")


class EmbeddingNet(nn.Module):
    def __init__(self, backbone_name: str, embedding_dim: int, pretrained: bool = True):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone_name, pretrained=pretrained)
        self.proj = nn.Linear(feat_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        z = self.proj(f)
        return F.normalize(z, dim=1)


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int, margin: float = 0.3, scale: float = 30.0):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(emb), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale
        return F.cross_entropy(logits, labels)
