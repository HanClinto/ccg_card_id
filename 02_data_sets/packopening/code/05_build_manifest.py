#!/usr/bin/env python3
"""Export manifest.csv and corners.csv from the packopening SQLite database.

manifest.csv  — ManifestRow-compatible training manifest (aligned/ images)
corners.csv   — Corner coordinates for all matched frames (frame/ images)

Both files are written to:
    datasets/packopening/manifest.csv
    datasets/packopening/corners.csv

Usage (run from project root):
    python 02_data_sets/packopening/code/05_build_manifest.py

    # Only include frames from specific sets
    python 02_data_sets/packopening/code/05_build_manifest.py --set-codes lea,leg

    # Include only frames matched against specific videos
    python 02_data_sets/packopening/code/05_build_manifest.py --video-id dQw4w9WgXcQ
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from ccg_card_id.config import cfg
from db import open_db

# ManifestRow column order (must match data.py ManifestRow)
MANIFEST_FIELDNAMES = [
    "image_path", "card_id", "card_name", "set_code",
    "split", "illustration_id", "oracle_id", "lang", "source",
]

CORNERS_FIELDNAMES = [
    "img_path", "card_id",
    "corner0_x", "corner0_y",
    "corner1_x", "corner1_y",
    "corner2_x", "corner2_y",
    "corner3_x", "corner3_y",
    "num_good_matches", "matching_area_pct",
]


def load_card_metadata(data_dir: Path) -> dict[str, dict]:
    """Build card_id → {card_name, oracle_id, illustration_id} from default_cards.json."""
    path = data_dir / "default_cards.json"
    if not path.exists():
        print(f"WARNING: default_cards.json not found — card_name/oracle_id will be empty")
        return {}
    cards = json.loads(path.read_text(encoding="utf-8"))
    return {
        c["id"]: {
            "card_name": c.get("name", ""),
            "oracle_id": c.get("oracle_id", ""),
            "illustration_id": c.get("illustration_id", ""),
        }
        for c in cards
        if "id" in c
    }


def assign_split(card_id: str) -> str:
    """Deterministic train/val/test split by card_id hash (80/10/10)."""
    h = hash(card_id) % 100
    if h < 80:
        return "train"
    elif h < 90:
        return "val"
    else:
        return "test"


def main() -> None:
    p = argparse.ArgumentParser(description="Build training manifests from packopening SQLite DB")
    p.add_argument("--set-codes", help="Comma-separated set codes to include (default: all)")
    p.add_argument("--video-id", help="Limit to one video")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"
    out_dir = args.data_dir / "datasets" / "packopening"
    manifest_path = out_dir / "manifest.csv"
    corners_path = out_dir / "corners.csv"

    con = open_db(db_path)

    # Build query
    where_clauses = []
    params = []
    if args.set_codes:
        codes = [s.strip().lower() for s in args.set_codes.split(",") if s.strip()]
        placeholders = ",".join(["?"] * len(codes))
        where_clauses.append(f"f.set_code IN ({placeholders})")
        params.extend(codes)
    if args.video_id:
        where_clauses.append("f.video_id = ?")
        params.append(args.video_id)
    # Only include frames with valid aligned_path
    where_clauses.append("f.aligned_path IS NOT NULL AND f.aligned_path != ''")

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    query = f"""
        SELECT f.*, v.set_codes as video_set_codes
        FROM frames f
        JOIN videos v ON f.video_id = v.video_id
        {where}
        ORDER BY f.video_id, f.frame_path
    """
    rows = con.execute(query, params).fetchall()
    print(f"Frames to export: {len(rows)}")

    if not rows:
        print("Nothing to export.")
        return

    print("Loading Scryfall card metadata...")
    card_meta = load_card_metadata(args.data_dir)

    manifest_rows = []
    corners_rows = []
    missing_meta = set()

    for row in rows:
        card_id = row["card_id"]
        meta = card_meta.get(card_id, {})
        if not meta:
            missing_meta.add(card_id)

        # manifest.csv — use aligned_path (dewarped card crop)
        manifest_rows.append({
            "image_path": row["aligned_path"],
            "card_id": card_id,
            "card_name": meta.get("card_name", ""),
            "set_code": row["set_code"] or "",
            "split": assign_split(card_id),
            "illustration_id": meta.get("illustration_id", row["illustration_id"] or ""),
            "oracle_id": meta.get("oracle_id", ""),
            "lang": "en",
            "source": "packopening",
        })

        # corners.csv — use original frame_path (undewarped, for corner detection training)
        if all(row[f"corner{i}_x"] is not None for i in range(4)):
            corners_rows.append({
                "img_path": row["frame_path"],
                "card_id": card_id,
                "corner0_x": row["corner0_x"], "corner0_y": row["corner0_y"],
                "corner1_x": row["corner1_x"], "corner1_y": row["corner1_y"],
                "corner2_x": row["corner2_x"], "corner2_y": row["corner2_y"],
                "corner3_x": row["corner3_x"], "corner3_y": row["corner3_y"],
                "num_good_matches": row["num_matches"],
                "matching_area_pct": row["matching_area_pct"],
            })

    # Write manifest.csv
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"manifest.csv: {len(manifest_rows)} rows → {manifest_path}")

    # Write corners.csv
    with corners_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CORNERS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(corners_rows)
    print(f"corners.csv:  {len(corners_rows)} rows → {corners_path}")

    if missing_meta:
        print(f"WARNING: {len(missing_meta)} card_ids not found in default_cards.json (card_name/oracle_id empty)")

    # Summary
    unique_cards = len({r["card_id"] for r in manifest_rows})
    unique_sets = len({r["set_code"] for r in manifest_rows})
    splits = {}
    for r in manifest_rows:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    print(f"\nSummary: {len(manifest_rows)} aligned frames, {unique_cards} unique cards, {unique_sets} sets")
    print(f"Splits: train={splits.get('train',0)} val={splits.get('val',0)} test={splits.get('test',0)}")


if __name__ == "__main__":
    main()
