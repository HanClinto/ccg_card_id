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
  Spatial : Full 0–360° random rotation (uniform)
            → corner coordinates are transformed accordingly.
            Black fill is used for out-of-frame pixels after rotation.
  Color   : ColorJitter (brightness/contrast/saturation/hue).
"""
from __future__ import annotations

import csv
import random
import sqlite3
import sys
from pathlib import Path as _Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_DETECTOR_DIR = _Path(__file__).resolve().parents[2]
if str(_DETECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(_DETECTOR_DIR))
from base import sort_corners_canonical

from pathlib import Path

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
INPUT_SIZE     = 384


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

# Slugs whose first character falls in this set are assigned to val.
# YouTube slugs are base64url so each character covers ~1.6% of slugs.
# X, Y, Z → ~4.8% val — stable, additive, and visually identifiable from
# the folder name without any computation.
_VAL_FIRST_CHARS: frozenset[str] = frozenset("XYZ")

# Cached split overrides: loaded once from split_overrides.json.
# Keys: 'val' (slugs pinned to val), 'force_train' (slugs pinned to train).
# Overrides take priority over the first-character rule, preserving the
# train/val assignment of all videos that existed before the hash rule
# was introduced — so adding new videos never contaminates existing splits.
_split_overrides: dict[str, frozenset[str]] | None = None


def _load_split_overrides(db_path: "Path") -> dict[str, frozenset[str]]:
    global _split_overrides
    if _split_overrides is not None:
        return _split_overrides
    import json
    override_path = db_path.parent / "split_overrides.json"
    if override_path.exists():
        data = json.loads(override_path.read_text())
        _split_overrides = {
            "val":         frozenset(data.get("val", [])),
            "force_train": frozenset(data.get("force_train", [])),
        }
    else:
        _split_overrides = {"val": frozenset(), "force_train": frozenset()}
    return _split_overrides


def _slug_is_val(slug: str, db_path: "Path | None" = None) -> bool:
    """Return True if this slug belongs to the val split.

    Priority:
      1. split_overrides.json 'val' list       → always val
      2. split_overrides.json 'force_train'    → always train
      3. first character in XYZ               → val
      4. everything else                       → train
    """
    if db_path is not None:
        overrides = _load_split_overrides(db_path)
        if slug in overrides["val"]:
            return True
        if slug in overrides["force_train"]:
            return False
    return bool(slug) and slug[0] in _VAL_FIRST_CHARS


def load_from_packopening_db(
    db_path: Path,
    data_dir: Path,
    neg_sample_n: int = 10_000,
    val_frac: float = 0.05,
    seed: int = 42,
    max_phash_dist: int = 20,
    min_phash_dist: int = 0,
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

    min_phash_dist (default 0): exclude frames with pHash distance below this
    threshold. Filters reference-image overlays where SIFT matched a card image
    displayed on-screen rather than a physical card being opened — those frames
    have near-zero pHash distance (nearly identical to the Scryfall reference)
    but corners that describe the overlay position, not a real card.

    Train/val split: assigned per video slug via a stable hash, so adding new
    videos never changes the assignment of existing ones.

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
        "SELECT frame_path, card_id, corner0_x, corner0_y, corner1_x, corner1_y, "
        "       corner2_x, corner2_y, corner3_x, corner3_y "
        "FROM frames "
        "WHERE corner0_x IS NOT NULL AND corner1_x IS NOT NULL "
        "  AND corner2_x IS NOT NULL AND corner3_x IS NOT NULL "
        "  AND phash_dist IS NOT NULL AND phash_dist <= ? AND phash_dist >= ?",
        (max_phash_dist, min_phash_dist),
    ).fetchall()

    # Group positive frames by video slug (extracted from frame_path:
    # "datasets/packopening/frames/{slug}/frame_XXXXX.jpg")
    video_to_frames: dict[str, list[dict]] = {}
    for r in rows_db:
        corners = np.array(
            [[r["corner0_x"], r["corner0_y"]],
             [r["corner1_x"], r["corner1_y"]],
             [r["corner2_x"], r["corner2_y"]],
             [r["corner3_x"], r["corner3_y"]]],
            dtype=np.float32,
        )
        # frame_path: "datasets/packopening/frames/{slug}/filename.jpg"
        slug = r["frame_path"].split("/")[3]
        video_to_frames.setdefault(slug, []).append({
            "img_path":     r["frame_path"],
            "card_id":      r["card_id"] or "",
            "card_present": True,
            "corners":      corners,
        })

    # Stable per-slug assignment — overrides freeze existing split, XYZ rule handles new slugs
    val_slugs   = {s for s in video_to_frames if _slug_is_val(s, db_path)}
    train_slugs = {s for s in video_to_frames if s not in val_slugs}

    train_positives = [f for s in train_slugs for f in video_to_frames[s]]
    val_positives   = [f for s in val_slugs   for f in video_to_frames[s]]

    # --- negatives: sample random frames from unmatched videos ---
    # These videos have no SIFT matches so are inherently disjoint from positives.
    # All negatives go to train; val quality is measured on positives only.
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

    train_rows = train_positives + negatives
    val_rows   = val_positives
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    print(
        f"packopening: {len(train_positives)+len(val_positives):,} positives "
        f"({len(train_slugs)} train videos / {len(val_slugs)} val videos) "
        f"+ {len(negatives):,} negatives "
        f"→ {len(train_rows):,} train / {len(val_rows):,} val"
    )
    return train_rows, val_rows


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
        self._color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1,
        )

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
                # Cache stores frames at frames_448/ (size-explicit) rather than frames/
                fast_rel = rel.replace(
                    "datasets/packopening/frames/", "datasets/packopening/frames_384/", 1
                )
                cached = self.fast_data_dir / fast_rel
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
            "card_id":      row.get("card_id") or "",
        }

    def _apply_augmentation(
        self, img: Image.Image, corners: np.ndarray | None
    ) -> tuple[Image.Image, np.ndarray | None]:
        import torchvision.transforms.functional as TF

        # Full random rotation: uniform 0–360° so the model sees cards at all
        # orientations, including the 45°-diagonal dead zone that the previous
        # 90°-increment-only augmentation never covered.
        angle = random.uniform(0.0, 360.0)
        if angle > 0.5:
            img = TF.rotate(img, angle=angle, expand=False, fill=0)
            if corners is not None:
                corners = _rotate_corners(corners, angle, cx=0.5, cy=0.5)

        # Restore canonical corner order (TL→TR→BR→BL) after augmentation.
        # This ensures the direct per-channel loss in training always sees
        # consistently ordered corners regardless of flip/rotation applied.
        if corners is not None:
            w, h = img.size
            corners = sort_corners_canonical(corners, img_w=w, img_h=h)

        return img, corners


def _rotate_corners(corners: np.ndarray, angle_deg: float, cx: float, cy: float) -> np.ndarray:
    """Rotate normalized corner coordinates around centre (cx, cy)."""
    rad = np.deg2rad(-angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    shifted = corners - np.array([cx, cy])
    rotated = shifted @ np.array([[cos_a, -sin_a], [sin_a, cos_a]]).T
    return np.clip(rotated + np.array([cx, cy]), 0.0, 1.0).astype(np.float32)


