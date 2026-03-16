#!/usr/bin/env python3
"""Export manifest.csv and corners.csv from the packopening SQLite database.

manifest.csv — ManifestRow-compatible (aligned/ dewarped card crops)
corners.csv  — Corner coordinates on original frames (for dewarping model training)

Written to: datasets/packopening/manifest.csv and corners.csv

Usage (run from project root):
    python 02_data_sets/packopening/code/pipeline/05_lookup_db_manifest.py
    python 02_data_sets/packopening/code/pipeline/05_lookup_db_manifest.py --set-codes lea,leg
    python 02_data_sets/packopening/code/pipeline/05_lookup_db_manifest.py --video-id dQw4w9WgXcQ
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_DIR))

from ccg_card_id.config import cfg
from ccg_card_id.catalog import catalog
from db import open_db

MANIFEST_FIELDNAMES = [
    "image_path", "card_id", "card_name", "set_code",
    "split", "illustration_id", "oracle_id", "lang", "source",
]
CORNERS_FIELDNAMES = [
    "img_path", "card_id",
    "corner0_x", "corner0_y", "corner1_x", "corner1_y",
    "corner2_x", "corner2_y", "corner3_x", "corner3_y",
    "num_good_matches", "matching_area_pct",
]


def load_card_metadata(card_ids: list[str]) -> dict[str, dict]:
    meta = catalog.cards_by_ids(card_ids)
    return {
        cid: {"card_name": c.get("name", ""), "oracle_id": c.get("oracle_id", ""),
              "illustration_id": c.get("illustration_id", "")}
        for cid, c in meta.items()
    }


def assign_split(card_id: str) -> str:
    h = hash(card_id) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def main() -> None:
    p = argparse.ArgumentParser(description="Build manifests from packopening SQLite DB")
    p.add_argument("--set-codes", help="Comma-separated set codes to include (default: all)")
    p.add_argument("--video-id", help="Limit to one video")
    p.add_argument("--data-dir", type=Path, default=cfg.data_dir)
    args = p.parse_args()

    db_path = args.data_dir / "datasets" / "packopening" / "packopening.db"
    out_dir = args.data_dir / "datasets" / "packopening"
    con = open_db(db_path)

    where_clauses = ["f.aligned_path IS NOT NULL AND f.aligned_path != ''"]
    params = []
    if args.set_codes:
        codes = [s for s in re.split(r"[\s,]+", args.set_codes.strip().lower()) if s]
        where_clauses.append(f"f.set_code IN ({','.join(['?']*len(codes))})")
        params.extend(codes)
    if args.video_id:
        where_clauses.append("f.video_id = ?")
        params.append(args.video_id)

    rows = con.execute(
        f"SELECT f.* FROM frames f JOIN videos v ON f.video_id=v.video_id "
        f"WHERE {' AND '.join(where_clauses)} ORDER BY f.video_id, f.frame_path",
        params,
    ).fetchall()

    print(f"Frames to export: {len(rows)}")
    if not rows:
        print("Nothing to export.")
        return

    print("Loading card metadata...")
    unique_card_ids = list({row["card_id"] for row in rows})
    meta = load_card_metadata(unique_card_ids)

    manifest_rows, corners_rows = [], []
    for row in rows:
        card_id = row["card_id"]
        m = meta.get(card_id, {})
        manifest_rows.append({
            "image_path": row["aligned_path"],
            "card_id": card_id,
            "card_name": m.get("card_name", ""),
            "set_code": row["set_code"] or "",
            "split": assign_split(card_id),
            "illustration_id": m.get("illustration_id", row["illustration_id"] or ""),
            "oracle_id": m.get("oracle_id", ""),
            "lang": "en",
            "source": "packopening",
        })
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

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(manifest_rows)

    corners_path = out_dir / "corners.csv"
    with corners_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CORNERS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(corners_rows)

    splits = {}
    for r in manifest_rows:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    unique_cards = len({r["card_id"] for r in manifest_rows})
    unique_sets = len({r["set_code"] for r in manifest_rows})

    print(f"manifest.csv: {len(manifest_rows)} rows → {manifest_path}")
    print(f"corners.csv:  {len(corners_rows)} rows → {corners_path}")
    print(f"Summary: {unique_cards} unique cards across {unique_sets} sets")
    print(f"Splits:  train={splits.get('train',0)}  val={splits.get('val',0)}  test={splits.get('test',0)}")


if __name__ == "__main__":
    main()
