#!/usr/bin/env python3
"""Generate top20_illustration_coverage.csv — frame coverage grid for the
top-20 most-reprinted MTG illustrations in the packopening dataset.

Rows: top-20 illustrations by number of distinct English printings.
Columns: every set in which any of the top-20 cards appears.
  - N/A  : card not printed in that set
  - 0    : printed but no frames captured yet
  - N>0  : N matched frames from that printing

Summary columns (before the set columns):
  card              : card name
  printings         : total distinct English printings of this illustration
  total_frames      : total matched frames across all sets
  sets_with_frames  : number of sets with at least one frame (coverage numerator)
  sets_printed      : number of sets this card was actually printed in (coverage denominator)

Usage (run from project root):
    python 06_eval/generate_top20_coverage.py
    python 06_eval/generate_top20_coverage.py --output path/to/output.csv
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ccg_card_id.config import cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Generate top-20 illustration coverage CSV")
    p.add_argument("--output", type=Path,
                   default=Path("06_eval/top20_illustration_coverage.csv"))
    p.add_argument("--top-n", type=int, default=20,
                   help="How many top illustrations to include (default: 20)")
    args = p.parse_args()

    catalog_db = cfg.card_db_path
    packopening_db = cfg.data_dir / "datasets" / "packopening" / "packopening.db"

    # --- Set name lookup from catalog DB ---
    from ccg_card_id.catalog import catalog
    set_name_map: dict[str, str] = catalog.set_names()

    ccon = sqlite3.connect(str(catalog_db))
    pcon = sqlite3.connect(str(packopening_db))

    # --- Top-N most-reprinted illustrations ---
    top_n = ccon.execute("""
        SELECT illustration_id, COUNT(DISTINCT id) AS n_printings, MIN(name) AS name
        FROM cards WHERE lang='en'
          AND illustration_id IS NOT NULL AND illustration_id != ''
          AND layout NOT IN ('art_series', 'token', 'emblem')
        GROUP BY illustration_id
        ORDER BY n_printings DESC
        LIMIT ?
    """, (args.top_n,)).fetchall()

    # --- Per-illustration: set -> [card_ids] ---
    illust_sets: dict[str, set[str]] = {}
    illust_set_cards: dict[str, dict[str, list[str]]] = {}
    for iid, _, _ in top_n:
        rows = ccon.execute(
            "SELECT id, set_code FROM cards WHERE lang='en' AND illustration_id=?", (iid,)
        ).fetchall()
        by_set: dict[str, list[str]] = defaultdict(list)
        for cid, sc in rows:
            by_set[sc].append(cid)
        illust_sets[iid] = set(by_set.keys())
        illust_set_cards[iid] = dict(by_set)

    all_sets = sorted({sc for sc_set in illust_sets.values() for sc in sc_set})

    # --- Frame counts per card_id ---
    all_card_ids = [cid for iid, _, _ in top_n
                    for cids in illust_set_cards[iid].values()
                    for cid in cids]
    ph = ",".join("?" * len(all_card_ids))
    frame_rows = pcon.execute(
        f"SELECT card_id, COUNT(*) FROM frames WHERE card_id IN ({ph}) GROUP BY card_id",
        all_card_ids
    ).fetchall()
    card_frame_count = {r[0]: r[1] for r in frame_rows}

    ccon.close()
    pcon.close()

    # --- Build output rows ---
    output_rows = []
    for iid, n_printings, name in top_n:
        row: dict = {"card": name, "printings": n_printings}
        total = sets_with_frames = sets_printed = 0
        for sc in all_sets:
            if sc not in illust_sets[iid]:
                row[sc] = "N/A"
            else:
                count = sum(card_frame_count.get(cid, 0)
                            for cid in illust_set_cards[iid][sc])
                row[sc] = count
                total += count
                sets_printed += 1
                if count > 0:
                    sets_with_frames += 1
        row["total_frames"] = total
        row["sets_with_frames"] = sets_with_frames
        row["sets_printed"] = sets_printed
        output_rows.append(row)

    # --- Write CSV ---
    set_col_headers = {sc: f"{set_name_map.get(sc, sc)} [{sc}]" for sc in all_sets}
    summary_fields = ["card", "printings", "total_frames", "sets_with_frames", "sets_printed"]
    fieldnames = summary_fields + all_sets

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        pretty_header = {field: field for field in summary_fields}
        pretty_header.update(set_col_headers)
        writer.writerow(pretty_header)
        writer.writerows(output_rows)

    print(f"Written: {args.output}")
    print(f"  {len(output_rows)} cards × {len(all_sets)} sets")


if __name__ == "__main__":
    main()
