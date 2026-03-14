#!/usr/bin/env python3
"""Gallery search — pHash (Hamming) and ArcFace (cosine similarity).

Both search classes lazy-load their gallery on first use.  Construct them
early and call .find() per request.

Gallery NPZ format
------------------
pHash:   embeddings (n, bytes_per_hash) uint8
ArcFace: embeddings (n, 128)            float32, already L2-normalised

Both are indexed in manifest CSV row order.  The manifest card_id column
provides the label for each gallery row.

Example
-------
    manager = GallerySearchManager()
    searcher = manager.get("phash_16x16")
    result = searcher.find(bgr_image, corners)
    if result:
        print(result.card_id, result.score)
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imagehash
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Path bootstrap — add project root so we can import ccg_card_id
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ccg_card_id.config import cfg  # noqa: E402

# ---------------------------------------------------------------------------
# Card output size for dewarping (standard card aspect ratio: 63mm × 88mm)
# ---------------------------------------------------------------------------

_DEWARP_W = 745
_DEWARP_H = 1040

# 256-entry lookup table for counting set bits in a byte (popcount)
_POPCOUNT_U8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Result of a single gallery search."""
    card_id: str
    """Scryfall UUID of the best matching gallery card."""
    score: float
    """Match quality score. For pHash: Hamming distance (lower = better).
    For ArcFace: cosine similarity (higher = better, max 1.0)."""
    score_type: str
    """'hamming' or 'cosine'"""


# ---------------------------------------------------------------------------
# Dewarping helper
# ---------------------------------------------------------------------------

def _dewarp(bgr: np.ndarray, corners_norm: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Perspective-warp a card region to a flat rectangle.

    Args:
        bgr:          HxWx3 uint8 BGR image.
        corners_norm: (4, 2) float32 normalised [0,1] corners, TL TR BR BL.
        out_w:        Output width in pixels.
        out_h:        Output height in pixels.

    Returns:
        out_w x out_h BGR image.
    """
    h, w = bgr.shape[:2]
    src = corners_norm * np.array([w, h], dtype=np.float32)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(bgr, M, (out_w, out_h))


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------

def _load_manifest_card_ids(csv_path: Path) -> list[str]:
    """Read card_id column from manifest CSV in row order.

    This is intentionally a fast, minimal read — no file-existence checks
    are performed.  The order must match the gallery NPZ row order.
    """
    ids: list[str] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids.append(row["card_id"].lower())
    return ids


# ---------------------------------------------------------------------------
# PHashSearch
# ---------------------------------------------------------------------------

class PHashSearch:
    """Identifies cards by perceptual hash Hamming distance.

    Gallery is loaded lazily on first call to find().

    Args:
        gallery_npz:  Path to pre-computed gallery NPZ (uint8 embeddings).
        manifest_csv: Path to artwork_id_manifest.csv (for card_id labels).
        hash_size:    Hash grid dimension (e.g. 8, 16, 32).
    """

    def __init__(self, gallery_npz: Path, manifest_csv: Path, hash_size: int) -> None:
        self._gallery_npz = gallery_npz
        self._manifest_csv = manifest_csv
        self._hash_size = hash_size
        self._gallery: np.ndarray | None = None   # (n, bytes) uint8
        self._card_ids: list[str] | None = None

    def _ensure_loaded(self) -> None:
        if self._gallery is not None:
            return
        data = np.load(self._gallery_npz, allow_pickle=False)
        self._gallery = data["embeddings"]           # (n, bytes) uint8
        self._card_ids = _load_manifest_card_ids(self._manifest_csv)
        assert len(self._gallery) == len(self._card_ids), (
            f"PHashSearch: gallery has {len(self._gallery)} rows but manifest has "
            f"{len(self._card_ids)} rows — they must match"
        )

    def find(self, bgr: np.ndarray, corners: np.ndarray) -> SearchResult | None:
        """Dewarp image, compute pHash, find closest gallery entry.

        Args:
            bgr:     HxWx3 uint8 BGR image.
            corners: (4, 2) float32 normalised corners from detector.

        Returns:
            SearchResult or None if gallery is empty / error.
        """
        self._ensure_loaded()
        assert self._gallery is not None and self._card_ids is not None

        try:
            warped = _dewarp(bgr, corners, _DEWARP_W, _DEWARP_H)
            pil = Image.fromarray(warped[:, :, ::-1])  # BGR → RGB
            h = imagehash.phash(pil, hash_size=self._hash_size)
            q = np.packbits(h.hash.flatten().astype(np.uint8))  # (bytes,) uint8
        except Exception as exc:
            raise RuntimeError(f"pHash computation failed: {exc}") from exc

        # Hamming distance via XOR + popcount lookup table
        xor = np.bitwise_xor(self._gallery, q)       # (n, bytes) uint8
        distances = _POPCOUNT_U8[xor].sum(axis=1)    # (n,) int counts

        best_idx = int(np.argmin(distances))
        best_dist = int(distances[best_idx])

        return SearchResult(
            card_id=self._card_ids[best_idx],
            score=float(best_dist),
            score_type="hamming",
        )

    @property
    def loaded(self) -> bool:
        return self._gallery is not None


# ---------------------------------------------------------------------------
# ArcFaceSearch
# ---------------------------------------------------------------------------

class ArcFaceSearch:
    """Identifies cards by cosine similarity of ArcFace embeddings.

    Gallery is loaded lazily on first call to find().

    Args:
        gallery_npz:  Path to pre-computed gallery NPZ (float32 embeddings).
        manifest_csv: Path to artwork_id_manifest.csv (for card_id labels).
        checkpoint:   Path to model .pt checkpoint.
    """

    def __init__(self, gallery_npz: Path, manifest_csv: Path, checkpoint: Path) -> None:
        self._gallery_npz = gallery_npz
        self._manifest_csv = manifest_csv
        self._checkpoint = checkpoint
        self._gallery: np.ndarray | None = None   # (n, 128) float32
        self._card_ids: list[str] | None = None
        self._model: Any = None
        self._device: Any = None
        self._transform: Any = None

    def _ensure_loaded(self) -> None:
        if self._gallery is not None:
            return

        import torch
        from torchvision import transforms

        # Add build path so we can import retrieval.py
        build_path = str(ROOT / "04_build" / "mobilevit_xxs")
        if build_path not in sys.path:
            sys.path.insert(0, build_path)
        from retrieval import load_finetuned_model  # noqa: PLC0415

        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self._device = device
        self._model, _ = load_finetuned_model(self._checkpoint, device)

        self._transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        data = np.load(self._gallery_npz, allow_pickle=False)
        self._gallery = data["embeddings"].astype(np.float32)   # (n, 128)
        self._card_ids = _load_manifest_card_ids(self._manifest_csv)

        assert len(self._gallery) == len(self._card_ids), (
            f"ArcFaceSearch: gallery has {len(self._gallery)} rows but manifest has "
            f"{len(self._card_ids)} rows — they must match"
        )

    def find(self, bgr: np.ndarray, corners: np.ndarray) -> SearchResult | None:
        """Dewarp image, embed with ArcFace model, find closest gallery entry.

        Args:
            bgr:     HxWx3 uint8 BGR image.
            corners: (4, 2) float32 normalised corners from detector.

        Returns:
            SearchResult or None on error.
        """
        self._ensure_loaded()
        assert self._gallery is not None and self._card_ids is not None and self._model is not None

        import torch

        try:
            warped = _dewarp(bgr, corners, _DEWARP_W, _DEWARP_H)
            pil = Image.fromarray(warped[:, :, ::-1])  # BGR → RGB
            tensor = self._transform(pil).unsqueeze(0).to(self._device)  # (1, 3, 224, 224)

            with torch.no_grad():
                emb = self._model(tensor).squeeze(0).cpu().numpy()  # (128,) float32
        except Exception as exc:
            raise RuntimeError(f"ArcFace embedding failed: {exc}") from exc

        # Cosine similarity: gallery is already L2-normalised, emb comes out of
        # the model L2-normalised.  Dot product = cosine similarity.
        sims = self._gallery @ emb   # (n,) float32

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        return SearchResult(
            card_id=self._card_ids[best_idx],
            score=best_sim,
            score_type="cosine",
        )

    @property
    def loaded(self) -> bool:
        return self._gallery is not None


# ---------------------------------------------------------------------------
# GallerySearchManager
# ---------------------------------------------------------------------------

# Matches e.g. "phash_16x16_256bit_gallery.npz"
_PHASH_RE = re.compile(r"phash_(\d+)x(\d+)_(\d+)bit_gallery\.npz$")

# Matches e.g. "mobilevit_xxs_ft_illustration_id_e75_128d_gallery.npz"
_ARCFACE_RE = re.compile(r"(mobilevit_xxs_ft_illustration_id_e(\d+)_(\d+)d)_gallery\.npz$")


class GallerySearchManager:
    """Discovers available gallery NPZs and manages lazy-loaded searcher instances.

    One instance of this class is created at server startup.  Callers ask for
    a searcher by name; the searcher is created and cached on first request.
    """

    def __init__(
        self,
        phash_gallery_dir: Path | None = None,
        arcface_gallery_dir: Path | None = None,
        manifest_csv: Path | None = None,
        arcface_checkpoint_dir: Path | None = None,
    ) -> None:
        self._phash_dir = phash_gallery_dir or (
            cfg.data_dir / "vectors" / "phash" / "gallery_manifest_artwork_id_manifest"
        )
        self._arcface_dir = arcface_gallery_dir or (
            cfg.data_dir / "vectors" / "mobilevit_xxs" / "img224"
            / "gallery_manifest_artwork_id_manifest"
        )
        self._manifest = manifest_csv or (
            cfg.data_dir / "mobilevit_xxs" / "artwork_id_manifest.csv"
        )
        self._ckpt_dir = arcface_checkpoint_dir or (
            cfg.data_dir / "results" / "mobilevit_xxs" / "mobilevit_xxs_illustration_id_128d"
        )

        # Cache of name → PHashSearch | ArcFaceSearch instances
        self._cache: dict[str, PHashSearch | ArcFaceSearch] = {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_phash_identifiers(self) -> list[dict]:
        """Scan for available pHash gallery NPZs.

        Returns list of {name, label, bytes_per_card}.
        """
        if not self._phash_dir.exists():
            return []
        results = []
        for path in sorted(self._phash_dir.glob("phash_*_gallery.npz")):
            m = _PHASH_RE.search(path.name)
            if not m:
                continue
            n = int(m.group(1))
            bits = int(m.group(3))
            bytes_per = (bits + 7) // 8
            name = f"phash_{n}x{n}"
            results.append({
                "name": name,
                "label": f"pHash {n}x{n} ({bits}-bit, {bytes_per} B/card)",
                "bytes_per_card": bytes_per,
                "_path": path,
                "_hash_size": n,
            })
        return results

    def list_arcface_identifiers(self) -> list[dict]:
        """Scan for available ArcFace gallery NPZs.

        Returns list of {name, label, bytes_per_card}.
        """
        if not self._arcface_dir.exists():
            return []
        results = []
        for path in sorted(self._arcface_dir.glob("*_gallery.npz")):
            m = _ARCFACE_RE.search(path.name)
            if not m:
                continue
            stem = m.group(1)           # e.g. "mobilevit_xxs_ft_illustration_id_e75_128d"
            epoch = int(m.group(2))
            dim = int(m.group(3))
            bytes_per = dim * 4         # float32
            name = f"arcface_illustration_id_e{epoch}"
            results.append({
                "name": name,
                "label": f"ArcFace epoch {epoch} ({dim}-d, {bytes_per} B/card)",
                "bytes_per_card": bytes_per,
                "_path": path,
                "_stem": stem,
                "_epoch": epoch,
            })
        return results

    def list_all_identifiers(self) -> list[dict]:
        """Return combined list of all available identifiers (public fields only)."""
        public_keys = {"name", "label", "bytes_per_card"}
        items = self.list_phash_identifiers() + self.list_arcface_identifiers()
        return [{k: v for k, v in item.items() if k in public_keys} for item in items]

    # ------------------------------------------------------------------
    # Searcher access
    # ------------------------------------------------------------------

    def get(self, name: str) -> PHashSearch | ArcFaceSearch:
        """Return (lazily created) searcher for the given identifier name.

        Raises KeyError if the identifier is not available.
        """
        if name in self._cache:
            return self._cache[name]

        # Check pHash
        for item in self.list_phash_identifiers():
            if item["name"] == name:
                searcher = PHashSearch(
                    gallery_npz=item["_path"],
                    manifest_csv=self._manifest,
                    hash_size=item["_hash_size"],
                )
                self._cache[name] = searcher
                return searcher

        # Check ArcFace
        for item in self.list_arcface_identifiers():
            if item["name"] == name:
                # Find corresponding checkpoint
                epoch = item["_epoch"]
                ckpt = self._find_arcface_checkpoint(epoch)
                if ckpt is None:
                    raise KeyError(
                        f"ArcFace gallery '{name}' found but no checkpoint for epoch {epoch} "
                        f"in {self._ckpt_dir}"
                    )
                searcher = ArcFaceSearch(
                    gallery_npz=item["_path"],
                    manifest_csv=self._manifest,
                    checkpoint=ckpt,
                )
                self._cache[name] = searcher
                return searcher

        raise KeyError(f"Identifier '{name}' not available. "
                       f"Available: {[i['name'] for i in self.list_all_identifiers()]}")

    def _find_arcface_checkpoint(self, epoch: int) -> Path | None:
        """Find the checkpoint .pt file for a given epoch number."""
        if not self._ckpt_dir.exists():
            return None

        # Try exact epoch file first
        exact = self._ckpt_dir / f"epoch_{epoch:04d}.pt"
        if exact.exists():
            return exact

        # Try last.pt as fallback (used when epoch matches the last saved epoch)
        last = self._ckpt_dir / "last.pt"
        if last.exists():
            return last

        return None
