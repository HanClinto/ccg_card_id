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

    def __init__(
        self,
        gallery_npz: Path,
        manifest_csv: Path,
        checkpoint: Path,
        image_size: int = 224,
    ) -> None:
        self._gallery_npz = gallery_npz
        self._manifest_csv = manifest_csv
        self._checkpoint = checkpoint
        self._image_size = image_size
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
        build_path = str(ROOT / "04_vectorize" / "mobilevit_xxs")
        if build_path not in sys.path:
            sys.path.insert(0, build_path)
        from retrieval import load_finetuned_model  # noqa: PLC0415

        device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self._device = device
        self._model, ckpt = load_finetuned_model(self._checkpoint, device)
        # Prefer image_size from checkpoint args if available; fall back to constructor arg
        cargs = ckpt.get("args", {})
        img_size = cargs.get("image_size", self._image_size)

        self._transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
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
# and           "mobilevit_xxs_ft_illustration_id+set_code_e15_128d_gallery.npz"
_ARCFACE_RE = re.compile(r"(mobilevit_xxs_ft_(.+?)_e(\d+)_(\d+)d)_gallery\.npz$")


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
        # Each entry: (gallery_dir, manifest_csv, ckpt_dir_hint)
        # ckpt_dir_hint: if set, checkpoint search is restricted to that specific subdirectory
        _mt_ckpt_dir = (
            cfg.data_dir / "results" / "mobilevit_xxs"
            / "mobilevit_xxs_multitask_illustration_id+set_code_shared_128d+128d"
              "_mvitxxs_shared2h_arcface_v2light_img448_ph10"
        )
        self._arcface_scan: list[tuple[Path, Path, Path | None]] = [
            (
                cfg.data_dir / "vectors" / "mobilevit_xxs" / "img224"
                / "gallery_manifest_artwork_id_manifest",
                cfg.data_dir / "mobilevit_xxs" / "artwork_id_manifest.csv",
                None,  # search all results/mobilevit_xxs/ subdirs
            ),
            (
                cfg.data_dir / "vectors" / "mobilevit_xxs" / "img448"
                / "gallery_manifest_manifest",
                cfg.data_dir / "mobilevit_xxs" / "manifest.csv",
                _mt_ckpt_dir,  # use the v2light run specifically
            ),
        ]
        # Legacy single-dir override (passed explicitly)
        if arcface_gallery_dir is not None:
            self._arcface_scan = [(arcface_gallery_dir, manifest_csv or self._arcface_scan[0][1], None)]

        self._manifest = manifest_csv or (
            cfg.data_dir / "mobilevit_xxs" / "artwork_id_manifest.csv"
        )
        self._results_dir = arcface_checkpoint_dir or (
            cfg.data_dir / "results" / "mobilevit_xxs"
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
        """Scan for available ArcFace gallery NPZs across all gallery directories.

        Returns list of {name, label, bytes_per_card}.
        """
        seen_names: set[str] = set()
        results = []
        for gallery_dir, manifest_csv, ckpt_dir_hint in self._arcface_scan:
            if not gallery_dir.exists():
                continue
            for path in sorted(gallery_dir.glob("*_gallery.npz")):
                m = _ARCFACE_RE.search(path.name)
                if not m:
                    continue
                label_field = m.group(2)   # e.g. "illustration_id" or "illustration_id+set_code"
                epoch = int(m.group(3))
                dim = int(m.group(4))
                bytes_per = dim * 4        # float32
                name = f"arcface_{label_field}_e{epoch}"
                if name in seen_names:
                    continue  # prefer first (lower-res) dir if duplicate epoch
                seen_names.add(name)
                results.append({
                    "name": name,
                    "label": f"ArcFace {label_field} e{epoch} ({dim}-d, {bytes_per} B/card)",
                    "bytes_per_card": bytes_per,
                    "_path": path,
                    "_manifest": manifest_csv,
                    "_label_field": label_field,
                    "_epoch": epoch,
                    "_ckpt_dir_hint": ckpt_dir_hint,
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
                epoch = item["_epoch"]
                label_field = item["_label_field"]
                ckpt_dir_hint = item.get("_ckpt_dir_hint")
                ckpt = self._find_arcface_checkpoint(epoch, label_field, ckpt_dir_hint)
                if ckpt is None:
                    raise KeyError(
                        f"ArcFace gallery '{name}' found but no matching checkpoint "
                        f"(epoch {epoch}, label '{label_field}') under {self._results_dir}"
                    )
                searcher = ArcFaceSearch(
                    gallery_npz=item["_path"],
                    manifest_csv=item["_manifest"],
                    checkpoint=ckpt,
                )
                self._cache[name] = searcher
                return searcher

        raise KeyError(f"Identifier '{name}' not available. "
                       f"Available: {[i['name'] for i in self.list_all_identifiers()]}")

    def _find_arcface_checkpoint(
        self, epoch: int, label_field: str, ckpt_dir_hint: Path | None = None
    ) -> Path | None:
        """Find checkpoint .pt file for a given epoch + label field.

        If ckpt_dir_hint is given, look only there (used to pin a specific run).
        Otherwise scan all results/mobilevit_xxs/ subdirs and prefer dirs whose
        name contains the full label_field string.
        """
        target = f"epoch_{epoch:04d}.pt"
        if ckpt_dir_hint is not None:
            ckpt = ckpt_dir_hint / target
            if ckpt.exists():
                return ckpt
            last = ckpt_dir_hint / "last.pt"
            return last if last.exists() else None

        if not self._results_dir.exists():
            return None
        candidates = []
        for run_dir in sorted(self._results_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            ckpt = run_dir / target
            if ckpt.exists():
                priority = 0 if label_field in run_dir.name else 1
                candidates.append((priority, run_dir.name, ckpt))
        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]))
            return candidates[0][2]
        return None
