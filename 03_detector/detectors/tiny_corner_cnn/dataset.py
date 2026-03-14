#!/usr/bin/env python3
"""PyTorch Dataset for card corner detection training.

Two training data sources are supported:

  packopening (default, recommended)
    Loads positive examples directly from the packopening SQLite DB
    (datasets/packopening/packopening.db).  Every row in the `frames` table
    is a SIFT-verified match with normalized corner coordinates — 380k+
    labeled frames across 768 pack-opening videos.

    Negatives are sampled from videos in 'frames_extracted' status: these
    videos had frames extracted but no SIFT matches, so random frames from
    them are a natural source of "card not visible / not identifiable" examples.

  clint (legacy / ablation)
    Reads corners.csv from the clint_cards_with_backgrounds dataset (~1267
    positives) plus the bad/ directory (62 negatives).  Useful for ablation
    studies but much smaller than the packopening source.

In both cases clint_cards_with_backgrounds is used as the TEST set only —
it is never included in training.  This gives a proper domain-generalization
evaluation: train on pack-opening video frames, test on handheld card scans.

Corner column order (both sources): TL, TR, BR, BL — normalized (x, y) in
[0, 1] relative to image dimensions, produced by the standard sum/diff sort.

Augmentation (training only):
  Spatial : horizontal flip (with corner reorder), rotation ±30°
            → corner coordinates are transformed accordingly.
  Color   : ColorJitter, RandomGrayscale, GaussianBlur.
"""
from __future__ import annotations

import csv
import random
import sqlite3
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE     = 224


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_from_packopening_db(
    db_path: Path,
    data_dir: Path,
    neg_sample_n: int = 10_000,
    val_frac: float = 0.05,
    seed: int = 42,
    max_phash_dist: int = 20,
) -> tuple[list[dict], list[dict]]:
    """Load SIFT-verified frames from the packopening DB.

    Positive examples: rows in the `frames` table filtered by pHash distance.
    pHash distance is the Hamming distance between the dewarped frame's pHash
    and the Scryfall reference image pHash — a low value confirms the SIFT
    homography matched the right card and the corners are trustworthy.

    Distribution (380k total):
      0–5:  11%  near-perfect      16–20: 21%  acceptable
      6–10: 34%  good              21–30: 10%  suspect
      11–15: 23% decent            31+:  0.6%  likely spurious

    Default max_phash_dist=20 keeps ~89% of frames (~340k) while excluding
    the clearly suspect tail. Frames with NULL phash_dist are also excluded
    (228 frames where pHash was not computed).

    Negative examples: random frames sampled from videos in 'frames_extracted'
    status (frames extracted but no SIFT match found).

    Returns (train_rows, val_rows).  Each row dict:
        img_path     : str, relative to data_dir
        card_present : bool
        corners      : np.ndarray (4, 2) float32, or None
    """
    rng = random.Random(seed)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    # --- positives ---
    rows_db = con.execute(
        "SELECT frame_path, corner0_x, corner0_y, corner1_x, corner1_y, "
        "       corner2_x, corner2_y, corner3_x, corner3_y "
        "FROM frames "
        "WHERE corner0_x IS NOT NULL AND corner1_x IS NOT NULL "
        "  AND corner2_x IS NOT NULL AND corner3_x IS NOT NULL "
        "  AND phash_dist IS NOT NULL AND phash_dist <= ?",
        (max_phash_dist,),
    ).fetchall()

    positives = []
    for r in rows_db:
        corners = np.array(
            [[r["corner0_x"], r["corner0_y"]],
             [r["corner1_x"], r["corner1_y"]],
             [r["corner2_x"], r["corner2_y"]],
             [r["corner3_x"], r["corner3_y"]]],
            dtype=np.float32,
        )
        positives.append({
            "img_path":     r["frame_path"],   # already relative to data_dir
            "card_present": True,
            "corners":      corners,
        })

    # --- negatives: sample random frames from unmatched videos ---
    unmatched_videos = con.execute(
        "SELECT slug FROM videos WHERE status = 'frames_extracted'"
    ).fetchall()
    con.close()

    frames_root = data_dir / "datasets" / "packopening" / "frames"
    neg_candidates: list[str] = []
    for v in unmatched_videos:
        slug_dir = frames_root / v["slug"]
        if slug_dir.exists():
            neg_candidates.extend(
                str((slug_dir / f.name).relative_to(data_dir))
                for f in slug_dir.iterdir()
                if f.suffix == ".jpg"
            )

    rng.shuffle(neg_candidates)
    negatives = [
        {"img_path": p, "card_present": False, "corners": None}
        for p in neg_candidates[:neg_sample_n]
    ]

    all_rows = positives + negatives
    rng.shuffle(all_rows)
    n_val = max(1, int(len(all_rows) * val_frac))
    print(
        f"packopening: {len(positives):,} positives + {len(negatives):,} negatives "
        f"→ {len(all_rows) - n_val:,} train / {n_val:,} val"
    )
    return all_rows[n_val:], all_rows[:n_val]


def load_from_clint_csv(
    corners_csv: Path,
    neg_dir: Path,
    data_dir: Path,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Load the clint_cards_with_backgrounds corners.csv (train/val split).

    Intended for ablation studies.  For normal training use
    load_from_packopening_db() and keep clint as the test set.

    Returns (train_rows, val_rows).
    """
    rng = random.Random(seed)

    positives: list[dict] = []
    with corners_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            corners = np.array(
                [[float(row[f"corner{i}_x"]), float(row[f"corner{i}_y"])]
                 for i in range(4)],
                dtype=np.float32,
            )
            positives.append({
                "img_path":     row["img_path"],
                "card_id":      row.get("card_id"),
                "card_present": True,
                "corners":      corners,
            })

    negatives: list[dict] = []
    if neg_dir.exists():
        for p in neg_dir.glob("*.jpg"):
            negatives.append({
                "img_path":     str(p.relative_to(data_dir)),
                "card_id":      None,
                "card_present": False,
                "corners":      None,
            })

    all_rows = positives + negatives
    rng.shuffle(all_rows)
    n_val = max(1, int(len(all_rows) * val_frac))
    print(
        f"clint: {len(positives)} positives + {len(negatives)} negatives "
        f"→ {len(all_rows) - n_val} train / {n_val} val"
    )
    return all_rows[n_val:], all_rows[:n_val]


def load_dataset(
    corners_csv: Path,
    neg_dir: Path,
    data_dir: Path,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Alias for load_from_clint_csv — convenience entry-point for the benchmark."""
    return load_from_clint_csv(corners_csv, neg_dir, data_dir, val_frac=val_frac, seed=seed)


def load_clint_as_test(
    corners_csv: Path,
    neg_dir: Path,
    data_dir: Path,
) -> list[dict]:
    """Return ALL clint frames as a flat test list (no train/val split).

    Use this when packopening is the training source, to evaluate
    domain generalization on clint.
    """
    rows: list[dict] = []
    with corners_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            corners = np.array(
                [[float(row[f"corner{i}_x"]), float(row[f"corner{i}_y"])]
                 for i in range(4)],
                dtype=np.float32,
            )
            rows.append({
                "img_path":     row["img_path"],
                "card_id":      row.get("card_id"),
                "card_present": True,
                "corners":      corners,
            })
    if neg_dir.exists():
        for p in neg_dir.glob("*.jpg"):
            rows.append({
                "img_path":     str(p.relative_to(data_dir)),
                "card_id":      None,
                "card_present": False,
                "corners":      None,
            })
    return rows


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CornerDataset(Dataset):
    """PyTorch Dataset for corner regression + card-presence detection."""

    def __init__(
        self,
        rows: list[dict],
        data_dir: Path,
        augment: bool = False,
        fast_data_dir: Path | None = None,
    ) -> None:
        self.rows          = rows
        self.data_dir      = data_dir
        self.fast_data_dir = fast_data_dir
        self.augment       = augment

        # If cached images are already 224×224, skip the Resize step
        if fast_data_dir is not None:
            self._normalize = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ])
        else:
            self._normalize = transforms.Compose([
                transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ])
        self._color_jitter = transforms.Compose([
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
        ])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        # Loop to skip missing files (rare: DB entries whose frames were never extracted)
        for attempt in range(len(self.rows)):
            row = self.rows[(idx + attempt) % len(self.rows)]
            rel = row["img_path"]
            # Use fast cache if available, fall back to original
            from_cache = False
            if self.fast_data_dir is not None:
                cached = self.fast_data_dir / rel
                if cached.exists():
                    img_path = cached
                    from_cache = True
                else:
                    img_path = self.data_dir / rel
            else:
                img_path = self.data_dir / rel
            try:
                img = Image.open(img_path).convert("RGB")
                break
            except (FileNotFoundError, OSError):
                continue
        else:
            # All rows missing — return a black image (should never happen)
            img = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE))
            row = self.rows[idx]
            from_cache = True
        # If we fell back to the original (not pre-resized), resize now
        if not from_cache and img.size != (INPUT_SIZE, INPUT_SIZE):
            img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)

        corners = row["corners"].copy() if row["corners"] is not None else None

        if self.augment:
            img, corners = self._apply_augmentation(img, corners)
            img = self._color_jitter(img)

        tensor = self._normalize(img)

        return {
            "image":        tensor,
            "card_present": torch.tensor(float(row["card_present"])),
            "corners":      (torch.tensor(corners.flatten(), dtype=torch.float32)
                             if corners is not None
                             else torch.zeros(8, dtype=torch.float32)),
        }

    def _apply_augmentation(
        self, img: Image.Image, corners: np.ndarray | None
    ) -> tuple[Image.Image, np.ndarray | None]:
        import torchvision.transforms.functional as TF

        # Random horizontal flip (corners must be reordered: TL↔TR, BL↔BR)
        if random.random() < 0.5:
            img = TF.hflip(img)
            if corners is not None:
                corners = corners.copy()
                corners[:, 0] = 1.0 - corners[:, 0]
                corners = corners[[1, 0, 3, 2]]

        # Random rotation ±30°
        if random.random() < 0.5:
            angle = random.uniform(-30, 30)
            img = TF.rotate(img, angle, expand=False)
            if corners is not None:
                corners = _rotate_corners(corners, angle, cx=0.5, cy=0.5)

        return img, corners


def _rotate_corners(corners: np.ndarray, angle_deg: float, cx: float, cy: float) -> np.ndarray:
    """Rotate normalized corner coordinates around centre (cx, cy)."""
    rad = np.deg2rad(-angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    shifted = corners - np.array([cx, cy])
    rotated = shifted @ np.array([[cos_a, -sin_a], [sin_a, cos_a]]).T
    return np.clip(rotated + np.array([cx, cy]), 0.0, 1.0).astype(np.float32)
