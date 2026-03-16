#!/usr/bin/env python3
"""Match extracted frames to reference cards using SIFT homography.

For each frame:
  1. Extract SIFT features.
  2. Quick match count against all set cards (Lowe's ratio test, no homography yet).
  3. Best card must beat 2nd-best by MARGIN matches (avoids ambiguous frames).
  4. Full homography on best candidate: all 4 corners must fall within the frame.
  5. Accepted frames are dewarped and written to aligned/{slug}/{card_id}_{pos}.jpg,
     with corner coordinates stored in the DB.

Quality thresholds are conservative: precision over recall.

Usage (run from project root):
    python 02_data_sets/packopening/code/pipeline/04_match_frames.py --slug <slug>
    python 02_data_sets/packopening/code/pipeline/04_match_frames.py --all

Requires: pip install opencv-contrib-python-headless
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import re
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import imagehash
    _PHASH_AVAILABLE = True
except ImportError:
    _PHASH_AVAILABLE = False
    print("WARNING: imagehash not installed — pHash verification disabled. "
          "Run: pip install imagehash pillow", file=__import__("sys").stderr)

ROOT = Path(__file__).resolve().parents[4]
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_DIR))

from ccg_card_id.config import cfg
from db import open_db, set_video_status, claim_next_video

# Load helpers from sibling scripts (filenames start with digits)
_spec = _ilu.spec_from_file_location(
    "precompute_sift", Path(__file__).parent / "02_precompute_sift.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_sift_features = _mod.load_sift_features

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
MIN_MATCHES = 20
MARGIN = 5
MIN_AREA_PCT = 0.05
MAX_AREA_PCT = 0.95
LOWE_RATIO = 0.75
REF_W, REF_H = 745, 1040
MIN_QUAD_ANGLE = 15.0   # degrees; quads with sharper corners are near-edge-on / degenerate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kp_array_to_kps(kp_array: np.ndarray) -> list[cv2.KeyPoint]:
    return [
        cv2.KeyPoint(x=float(r[0]), y=float(r[1]), size=float(r[2]),
                     angle=float(r[3]), response=float(r[4]),
                     octave=int(r[5]), class_id=int(r[6]))
        for r in kp_array
    ]


def _kps_to_array(kps: list[cv2.KeyPoint]) -> np.ndarray:
    if not kps:
        return np.empty((0, 7), dtype=np.float32)
    return np.array(
        [[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id]
         for k in kps],
        dtype=np.float32,
    )


def build_flann() -> cv2.FlannBasedMatcher:
    return cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})


def build_global_flann(all_descs: np.ndarray) -> cv2.FlannBasedMatcher:
    """Build a single FLANN index from the pre-concatenated descriptor matrix."""
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 50})
    flann.add([all_descs])
    flann.train()
    return flann


# ---------------------------------------------------------------------------
# Gallery cache (keyed by sorted set_codes + lang)
# ---------------------------------------------------------------------------

_GALLERY_CACHE_VERSION = 2


def _gallery_cache_key(set_codes: list[str], lang: str) -> str:
    lang = (lang or "en").strip().lower()
    return "_".join(sorted(set_codes)) + f"_{lang}"


def _gallery_cache_stale(cache_path: Path, source_npz_paths: list[Path]) -> bool:
    """True if cache is missing or any source file is newer than the cache."""
    if not cache_path.exists():
        return True
    cache_mtime = cache_path.stat().st_mtime
    return any(p.stat().st_mtime > cache_mtime for p in source_npz_paths)


def _save_gallery_cache(
    cache_path: Path,
    gallery: list[dict],
    all_descs: np.ndarray,
    desc_to_card: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    K = len(gallery)

    card_ids      = np.array([g["card_id"].encode()  for g in gallery], dtype="S36")
    card_set_codes = np.array([g["set_code"].encode() for g in gallery], dtype="S16")

    kp_arrays  = [_kps_to_array(g["kps"]) for g in gallery]
    kp_lengths = np.array([len(a) for a in kp_arrays], dtype=np.int32)
    kp_data    = np.vstack(kp_arrays) if K else np.empty((0, 7), dtype=np.float32)

    desc_lengths = np.array(
        [len(g["descs"]) if g["descs"] is not None else 0 for g in gallery],
        dtype=np.int32,
    )

    # pHash: store as 64-element uint8 (the 8×8 bool hash flattened)
    phash_data  = np.zeros((K, 64), dtype=np.uint8)
    phash_valid = np.zeros(K, dtype=np.int8)
    for i, g in enumerate(gallery):
        ph = g.get("ref_phash")
        if ph is not None:
            phash_data[i]  = ph.hash.flatten().astype(np.uint8)
            phash_valid[i] = 1

    np.savez_compressed(
        str(cache_path),
        version=np.array([_GALLERY_CACHE_VERSION], dtype=np.int32),
        all_descs=all_descs,
        desc_to_card=desc_to_card,
        desc_lengths=desc_lengths,
        card_ids=card_ids,
        card_set_codes=card_set_codes,
        kp_data=kp_data,
        kp_lengths=kp_lengths,
        phash_data=phash_data,
        phash_valid=phash_valid,
    )


def _load_gallery_cache(
    cache_path: Path,
) -> tuple[list[dict], np.ndarray, np.ndarray] | None:
    """Return (gallery, all_descs, desc_to_card) from cache, or None on failure."""
    if not cache_path.exists():
        return None
    try:
        data = np.load(str(cache_path), allow_pickle=False)
        if int(data["version"][0]) != _GALLERY_CACHE_VERSION:
            return None

        all_descs    = data["all_descs"]
        desc_to_card = data["desc_to_card"]
        desc_lengths = data["desc_lengths"]
        card_ids     = data["card_ids"]
        card_scs     = data["card_set_codes"]
        kp_data      = data["kp_data"]
        kp_lengths   = data["kp_lengths"]
        phash_data   = data["phash_data"]
        phash_valid  = data["phash_valid"]

        gallery: list[dict] = []
        kp_off = desc_off = 0
        for i in range(len(card_ids)):
            kl = int(kp_lengths[i])
            dl = int(desc_lengths[i])
            kp_arr = kp_data[kp_off : kp_off + kl]
            descs  = all_descs[desc_off : desc_off + dl] if dl else None
            ref_phash = None
            if phash_valid[i] and _PHASH_AVAILABLE:
                ref_phash = imagehash.ImageHash(phash_data[i].reshape(8, 8).astype(bool))
            gallery.append({
                "card_id":   card_ids[i].decode(),
                "set_code":  card_scs[i].decode(),
                "kps":       _kp_array_to_kps(kp_arr),
                "descs":     descs,
                "ref_phash": ref_phash,
            })
            kp_off   += kl
            desc_off += dl

        return gallery, all_descs, desc_to_card
    except Exception as e:
        print(f"  WARNING: gallery cache invalid ({e}) — will rebuild", file=sys.stderr)
        return None


def homography_corners(frame_kps, ref_kps, frame_descs, ref_descs,
                       frame_h, frame_w, flann):
    """Returns (scene_corners 4×1×2, inlier_count) or (None, 0)."""
    try:
        matches = flann.knnMatch(frame_descs, ref_descs, k=2)
    except Exception:
        return None, 0
    good = [m for m, n in matches if m.distance < LOWE_RATIO * n.distance]
    if len(good) < MIN_MATCHES:
        return None, 0

    src = np.float32([ref_kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([frame_kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if M is None:
        return None, 0

    ref_corners = np.float32([[0,0],[REF_W,0],[REF_W,REF_H],[0,REF_H]]).reshape(-1,1,2)
    scene_corners = cv2.perspectiveTransform(ref_corners, M)

    margin = 5
    for pt in scene_corners:
        x, y = pt[0]
        if x < -margin or x > frame_w + margin or y < -margin or y > frame_h + margin:
            return None, 0

    xs = [pt[0][0] for pt in scene_corners]
    ys = [pt[0][1] for pt in scene_corners]
    area_pct = ((max(xs) - min(xs)) * (max(ys) - min(ys))) / (frame_w * frame_h)
    if not (MIN_AREA_PCT <= area_pct <= MAX_AREA_PCT):
        return None, 0

    inliers = int(mask.sum()) if mask is not None else 0
    return scene_corners, inliers


def is_valid_quad(scene_corners) -> bool:
    """Return False if the source quad is non-convex or has a near-degenerate angle."""
    pts = scene_corners.reshape(4, 2).astype(np.float32)
    if not cv2.isContourConvex(pts.reshape(-1, 1, 2).astype(np.int32)):
        return False
    for i in range(4):
        v1 = pts[(i - 1) % 4] - pts[i]
        v2 = pts[(i + 1) % 4] - pts[i]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-6:
            return False
        cos_a = np.dot(v1, v2) / denom
        angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        if angle < MIN_QUAD_ANGLE:
            return False
    return True


def compute_phash(img_bgr):
    """Compute a pHash for an OpenCV BGR image. Returns None if imagehash unavailable."""
    if not _PHASH_AVAILABLE:
        return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


def dewarp(frame, scene_corners):
    dst = np.float32([[0,0],[REF_W,0],[REF_W,REF_H],[0,REF_H]])
    src = scene_corners.reshape(4, 2).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (REF_W, REF_H))


def _cache_dir_name(set_code: str, lang: str) -> str:
    lang = lang.strip().lower() or "en"
    return set_code if lang == "en" else f"{set_code}_{lang}"


def load_gallery(
    cache_root: Path,
    set_codes: list[str],
    lang: str = "en",
    gallery_cache_dir: Path | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Load per-card SIFT features and build the global descriptor matrix.

    Returns (gallery, all_descs, desc_to_card).
    If gallery_cache_dir is given, the result is cached and reused when
    no source .npz file has changed since the last cache write.
    """
    all_npz: list[tuple[str, Path]] = []
    for sc in set_codes:
        dir_name = _cache_dir_name(sc, lang)
        sc_dir = cache_root / dir_name
        if not sc_dir.exists():
            print(f"  WARNING: SIFT cache missing for '{dir_name}' — run 02_precompute_sift.py first")
            continue
        all_npz.extend((sc, p) for p in sorted(sc_dir.glob("*.npz")))

    # --- Try gallery cache ---
    gcache_path: Path | None = None
    if gallery_cache_dir is not None:
        gcache_path = gallery_cache_dir / f"{_gallery_cache_key(set_codes, lang)}.npz"
        source_paths = [p for _, p in all_npz]
        if not _gallery_cache_stale(gcache_path, source_paths):
            result = _load_gallery_cache(gcache_path)
            if result is not None:
                gallery, all_descs, desc_to_card = result
                print(f"  Gallery: {len(gallery)} cards (loaded from cache)")
                return gallery, all_descs, desc_to_card

    # --- Build from individual .npz files ---
    gallery: list[dict] = []
    with tqdm(all_npz, unit="card", desc="  loading gallery", leave=False) as pbar:
        for sc, npz_path in pbar:
            stem = npz_path.stem
            if stem.endswith("_back"):
                card_id = stem[:-5]
                face = "back"
            else:
                card_id = stem
                face = "front"
            kp_array, descs = load_sift_features(npz_path)
            ref_phash = None
            if _PHASH_AVAILABLE:
                ref_img_path = (cfg.scryfall_images_dir / face
                                / card_id[0] / card_id[1] / f"{card_id}.png")
                ref_img = cv2.imread(str(ref_img_path))
                if ref_img is not None:
                    ref_phash = compute_phash(ref_img)
            gallery.append({
                "card_id":   card_id,
                "set_code":  sc,
                "kps":       _kp_array_to_kps(kp_array),
                "descs":     descs,
                "ref_phash": ref_phash,
            })

    # --- Concatenate descriptor matrix ---
    descs_list: list[np.ndarray] = []
    mapping: list[int] = []
    for i, card in enumerate(gallery):
        d = card["descs"]
        if d is None or len(d) == 0:
            continue
        descs_list.append(d)
        mapping.extend([i] * len(d))

    if not descs_list:
        return gallery, np.empty((0, 128), dtype=np.float32), np.empty(0, dtype=np.int32)

    all_descs    = np.vstack(descs_list).astype(np.float32)
    desc_to_card = np.array(mapping, dtype=np.int32)

    # --- Save gallery cache ---
    if gcache_path is not None:
        try:
            _save_gallery_cache(gcache_path, gallery, all_descs, desc_to_card)
            print(f"  gallery cache saved → {gcache_path.name}")
        except Exception as e:
            print(f"  WARNING: could not save gallery cache: {e}", file=sys.stderr)

    return gallery, all_descs, desc_to_card


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def _match_batch(
    frame_paths: list[Path],
    gallery: list[dict],
    global_flann,
    desc_to_card: np.ndarray,
    homography_flann,
    sift,
    already_done: set[str],
    aligned_dir: Path,
    video_id: str,
    data_dir: Path,
    con,
    card_id_to_illust: dict[str, str] | None = None,
    desc: str = "  matching",
) -> tuple[int, int, int]:
    """Run the SIFT matching loop over frame_paths.

    Returns (matched, discarded, skipped).
    """
    matched = discarded = skipped = 0

    with tqdm(frame_paths, unit="frame", desc=desc) as pbar:
        for frame_path in pbar:
            pbar.set_postfix(matched=matched, discarded=discarded, skipped=skipped)
            rel_path = str(frame_path.relative_to(data_dir))
            if rel_path in already_done:
                skipped += 1
                continue

            frame = cv2.imread(str(frame_path))
            if frame is None:
                discarded += 1
                continue

            frame_h, frame_w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_kps, frame_descs = sift.detectAndCompute(gray, None)
            if frame_kps is None or frame_descs is None or len(frame_kps) < MIN_MATCHES:
                discarded += 1
                continue

            # Quick pre-filter: single knnMatch against the global index
            try:
                raw_matches = global_flann.knnMatch(frame_descs.astype(np.float32), k=2)
            except Exception:
                discarded += 1
                continue

            card_counts = np.zeros(len(gallery), dtype=np.int32)
            for pair in raw_matches:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < LOWE_RATIO * n.distance:
                    card_counts[desc_to_card[m.trainIdx]] += 1

            best_i = int(card_counts.argmax())
            best_n = int(card_counts[best_i])
            second_n = int(np.partition(card_counts, -2)[-2]) if len(gallery) > 1 else 0

            if best_n < MIN_MATCHES or (best_n - second_n) < MARGIN:
                discarded += 1
                continue

            # Full homography on best candidate
            best = gallery[best_i]
            scene_corners, inliers = homography_corners(
                frame_kps, best["kps"], frame_descs, best["descs"], frame_h, frame_w, homography_flann
            )
            if scene_corners is None:
                discarded += 1
                continue

            # Reject degenerate quads (non-convex or near-edge-on)
            if not is_valid_quad(scene_corners):
                discarded += 1
                continue

            # Dewarp and compute pHash distance (stored for reference, not used to filter)
            dewarped = dewarp(frame, scene_corners)
            phash_dist: int | None = None
            if _PHASH_AVAILABLE and best["ref_phash"] is not None:
                match_phash = compute_phash(dewarped)
                phash_dist = int(match_phash - best["ref_phash"])

            # Save aligned crop
            frame_pos = frame_path.stem.split("_")[-1]
            aligned_path = aligned_dir / f"{best['card_id']}_{frame_pos}.jpg"
            cv2.imwrite(str(aligned_path), dewarped, [cv2.IMWRITE_JPEG_QUALITY, 92])

            # Normalised corners, top-left first
            corners = [(float(scene_corners[i,0,0]) / frame_w, float(scene_corners[i,0,1]) / frame_h)
                       for i in range(4)]
            idx = min(range(4), key=lambda i: corners[i][0] + corners[i][1])
            corners = corners[idx:] + corners[:idx]
            xs, ys = [c[0] for c in corners], [c[1] for c in corners]
            area_pct = (max(xs) - min(xs)) * (max(ys) - min(ys))

            illust_id = (card_id_to_illust or {}).get(best["card_id"])
            con.execute(
                """INSERT OR IGNORE INTO frames
                   (video_id, frame_path, aligned_path, card_id, illustration_id, set_code,
                    num_matches, corner0_x, corner0_y, corner1_x, corner1_y,
                    corner2_x, corner2_y, corner3_x, corner3_y, matching_area_pct,
                    phash_dist)
                   VALUES (?,?,?,?,?,?, ?,?,?,?,?, ?,?,?,?,?, ?)""",
                (video_id, rel_path, str(aligned_path.relative_to(data_dir)),
                 best["card_id"], illust_id, best["set_code"], inliers,
                 corners[0][0], corners[0][1], corners[1][0], corners[1][1],
                 corners[2][0], corners[2][1], corners[3][0], corners[3][1], area_pct,
                 phash_dist),
            )
            con.commit()
            matched += 1

    return matched, discarded, skipped


def process_video(
    video: dict,
    data_dir: Path,
    con,
    rebuild: bool = False,
    use_gallery_cache: bool = True,
    fast_data_dir: Path | None = None,
) -> tuple[int, int]:
    slug = video["slug"]
    video_id = video["video_id"]
    set_codes = [s for s in re.split(r"[\s,]+", video["set_codes"].strip().lower()) if s]
    lang = (video.get("lang") or "en").strip().lower() or "en"
    if not set_codes:
        print(f"  ERROR: no set_codes for {slug}")
        return 0, 0

    frames_dir = data_dir / "datasets" / "packopening" / "frames" / slug
    aligned_dir = data_dir / "datasets" / "packopening" / "aligned" / slug
    cache_root = data_dir / "datasets" / "packopening" / "sift_cache"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir} — re-run 03_extract_frames.py")

    # Detect stale extractions: if the second frame has a tiny PTS (sequential integer
    # from the old fps= filter), the timestamps will all map to frame 0.
    if len(frame_paths) >= 2:
        second_pts = int(frame_paths[1].stem.rsplit("_", 1)[-1])
        if 0 < second_pts < 100:
            print(f"  WARNING: frame filenames look like old sequential integers "
                  f"(second frame PTS={second_pts}). Timestamps will be wrong — "
                  f"re-run 03_extract_frames.py --rebuild first.", file=sys.stderr)

    already_done: set[str] = set()
    if not rebuild:
        rows = con.execute("SELECT frame_path FROM frames WHERE video_id=?", (video_id,)).fetchall()
        already_done = {r["frame_path"] for r in rows}
    else:
        con.execute("DELETE FROM frames WHERE video_id=?", (video_id,))
        con.execute("UPDATE videos SET densified=0 WHERE video_id=?", (video_id,))
        con.commit()

    # Gallery cache goes on fast storage (SSD) if available, otherwise alongside sift_cache
    if use_gallery_cache:
        _cache_base = fast_data_dir if fast_data_dir is not None else data_dir
        gallery_cache_dir = _cache_base / "datasets" / "packopening" / "sift_gallery_cache"
    else:
        gallery_cache_dir = None
    lang_label = f" lang={lang}" if lang != "en" else ""
    print(f"  Loading SIFT gallery for {set_codes}{lang_label}...")
    gallery, all_descs, desc_to_card = load_gallery(
        cache_root, set_codes, lang=lang, gallery_cache_dir=gallery_cache_dir
    )
    if not gallery:
        print("  ERROR: empty gallery")
        return 0, 0
    if len(all_descs) == 0:
        print("  ERROR: gallery has no descriptors")
        return 0, 0

    print("  Building global FLANN index...", end=" ", flush=True)
    global_flann = build_global_flann(all_descs)
    print("done")

    # Build card_id → illustration_id lookup from catalog
    card_id_to_illust: dict[str, str] = {}
    _catalog_base = fast_data_dir if fast_data_dir is not None else data_dir
    _catalog_db = _catalog_base / "catalog" / "scryfall" / "cards.db"
    if _catalog_db.exists():
        _ccon = sqlite3.connect(str(_catalog_db))
        for _row in _ccon.execute("SELECT id, illustration_id FROM cards"):
            if _row[1]:
                card_id_to_illust[_row[0].lower()] = _row[1].lower()
        _ccon.close()
        print(f"  illustration_id lookup: {len(card_id_to_illust):,} cards")
    else:
        print(f"  WARNING: catalog DB not found at {_catalog_db} — illustration_id will be NULL",
              file=sys.stderr)

    homography_flann = build_flann()
    sift = cv2.SIFT_create()

    matched, discarded, skipped = _match_batch(
        frame_paths, gallery, global_flann, desc_to_card,
        homography_flann, sift, already_done, aligned_dir,
        video_id, data_dir, con,
        card_id_to_illust=card_id_to_illust,
    )
    print(f"  matched={matched}  discarded={discarded}  skipped={skipped}  total={len(frame_paths)}")
    return matched, discarded


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="SIFT-match frames to reference cards")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--slug")
    group.add_argument("--video-id", help="YouTube video ID")
    group.add_argument("--all", action="store_true", help="Process all 'frames_extracted' videos")
    p.add_argument("--channel",
                   help="Only process videos from this channel (use with --all), e.g. '@MTGUnpacked'")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--no-gallery-cache", action="store_true",
                   help="Rebuild gallery from individual .npz files even if a cache exists")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    p.add_argument("--fast-data-dir", type=Path, default=cfg.fast_data_dir,
                   help="Fast-storage root for gallery cache (default: cfg.fast_data_dir)")
    args = p.parse_args()

    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"
    con = open_db(db_path)

    if args.all:
        videos = None   # sentinel: use claim_next_video loop below
    elif args.slug:
        v = con.execute("SELECT * FROM videos WHERE slug=?", (args.slug,)).fetchone()
        if not v:
            print(f"ERROR: slug '{args.slug}' not in DB.", file=sys.stderr)
            sys.exit(1)
        videos = [v]
    else:
        v = con.execute("SELECT * FROM videos WHERE video_id=?", (args.video_id,)).fetchone()
        if not v:
            print(f"ERROR: video_id '{args.video_id}' not in DB.", file=sys.stderr)
            sys.exit(1)
        videos = [v]

    current_video_id: list[str | None] = [None]  # mutable cell so inner fn can write it

    def _process_one(video) -> int:
        """Process a single already-claimed video. Returns matched frame count."""
        current_video_id[0] = video["video_id"]
        print(f"\n[{video['video_id']}] {video['title'][:70]}")
        try:
            m, _ = process_video(dict(video), args.data_dir, con,
                                 rebuild=args.rebuild,
                                 use_gallery_cache=not args.no_gallery_cache,
                                 fast_data_dir=args.fast_data_dir)
            set_video_status(con, video["video_id"], "done")
            con.execute("UPDATE videos SET processed_date=date('now') WHERE video_id=?",
                        (video["video_id"],))
            con.commit()
            return m
        except RuntimeError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            print(f"  status reverted to 'frames_extracted' — fix the issue and re-run", file=sys.stderr)
            set_video_status(con, video["video_id"], "frames_extracted")
            return 0
        except Exception:
            import traceback; traceback.print_exc()
            set_video_status(con, video["video_id"], "error")
            return 0
        finally:
            current_video_id[0] = None

    total_matched = 0
    try:
        if videos is None:
            # --all: claim one video at a time so multiple workers can run safely
            channel = args.channel or None
            if channel:
                print(f"Filtering to channel: {channel}")
            while True:
                video = claim_next_video(con, "frames_extracted", "processing", channel=channel)
                if video is None:
                    print("No more videos to process.")
                    break
                total_matched += _process_one(video)
        else:
            for video in videos:
                set_video_status(con, video["video_id"], "processing")
                total_matched += _process_one(video)
    except KeyboardInterrupt:
        vid_id = current_video_id[0]
        if vid_id:
            print(f"\n[interrupted] resetting {vid_id} → frames_extracted", file=sys.stderr)
            set_video_status(con, vid_id, "frames_extracted")
        else:
            print("\n[interrupted]", file=sys.stderr)
        sys.exit(0)

    print(f"\nTotal matched frames: {total_matched}")
    if total_matched:
        print("Next: python 02_data_sets/packopening/code/pipeline/05_lookup_db_manifest.py")


if __name__ == "__main__":
    main()
