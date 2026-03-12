#!/usr/bin/env python3
"""Build plst_by_host_set.json — mapping host set code → plst/spg card IDs.

Scrapes the three MTG wiki pages that document which cards appear in The List
for each Set/Play Booster release, then resolves card names to Scryfall UUIDs
(all plst and spg printings with that name — option A: let SIFT sort it out).

Output: 02_data_sets/packopening/plst_by_host_set.json

Usage (run from project root):
    python 02_data_sets/packopening/code/build_plst_by_host_set.py
    python 02_data_sets/packopening/code/build_plst_by_host_set.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import urllib.parse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from ccg_card_id.catalog import catalog

# ---------------------------------------------------------------------------
# Wiki API fetch
# ---------------------------------------------------------------------------

_API = "https://mtg.fandom.com/api.php"


def fetch_wikitext(page_title: str) -> str:
    params = urllib.parse.urlencode({
        "action": "parse",
        "page": page_title,
        "format": "json",
        "prop": "wikitext",
    })
    url = f"{_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "ccg_card_id/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read().decode())
    wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
    if not wikitext:
        raise RuntimeError(f"Empty wikitext for page '{page_title}'")
    return wikitext


# ---------------------------------------------------------------------------
# Parser — tick-matrix format (ZNR-SNC, CLB-LCI)
# ---------------------------------------------------------------------------

def _parse_tick_matrix(wikitext: str) -> dict[str, list[str]]:
    """Parse a tick-matrix page into {host_set_code: [card_name, ...]}."""
    lines = wikitext.splitlines()

    # Header columns: lines matching !XXX<br>... (the set columns, not Set code/Version/Card)
    host_sets: list[str] = []
    for line in lines:
        m = re.match(r"^!([A-Z0-9]+)<br>", line)
        if m:
            host_sets.append(m.group(1).lower())

    result: dict[str, list[str]] = {sc: [] for sc in host_sets}

    # Split rows by |- separator
    rows: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if re.match(r"^\|-", line):
            if current:
                rows.append(current)
            current = []
        else:
            current.append(line)
    if current:
        rows.append(current)

    for row in rows:
        # Extract card name from {{card|NAME|...|PLIST}} or {{card|NAME||PLIST}}
        card_name: str | None = None
        for line in row:
            m = re.search(r"\{\{card\|([^|{}\n]+?)(?:\|[^}]*)?\}\}", line)
            if m:
                card_name = m.group(1).strip()
                break
        if not card_name:
            continue

        # Walk lines tracking host-set column index.
        # Fixed columns: set_code | set_name | card  (3 total, then host sets follow)
        # Rules:
        #   - Line starts with | (not |-): new cell.  Split content by || for multi-cell lines.
        #     Each segment = one column.  Skip until past the card cell.
        #   - Line contains {{cell|tick}} but does NOT start with |: standalone tick column.
        host_col = -1  # becomes 0 on first host-set increment
        past_card = False

        for line in row:
            if re.match(r"^\|-", line):
                continue

            if line.startswith("|"):
                content = line[1:]  # strip leading |

                if not past_card:
                    # Still in fixed columns
                    if re.search(r"\{\{card\|", content):
                        past_card = True
                    # (no host_col change yet for fixed columns)
                else:
                    # In host-set columns; || means 2 cells on one line
                    segments = content.split("||")
                    for seg in segments:
                        host_col += 1
                        if 0 <= host_col < len(host_sets):
                            if "{{cell|tick}}" in seg:
                                result[host_sets[host_col]].append(card_name)

            elif "{{cell|tick}}" in line and past_card:
                # Standalone tick line (no leading |) = one host-set column
                host_col += 1
                if 0 <= host_col < len(host_sets):
                    result[host_sets[host_col]].append(card_name)

    return result


# ---------------------------------------------------------------------------
# Parser — row-per-entry format (MKM-Present)
# ---------------------------------------------------------------------------

def _parse_row_per_entry(wikitext: str) -> dict[str, list[str]]:
    """Parse MKM-Present style: each row is (host_set, original_set, ..., card, ...).

    Handles both 5-column (MKM/OTJ section) and 4-column (MH3+ section) tables.
    In both cases column 0 is the host set code and one column contains the card template.
    """
    result: dict[str, list[str]] = defaultdict(list)

    lines = wikitext.splitlines()
    rows: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if re.match(r"^\|-", line):
            if current:
                rows.append(current)
            current = []
        else:
            current.append(line)
    if current:
        rows.append(current)

    for row in rows:
        # Extract cells (lines starting with |, not |-)
        cells = []
        for line in row:
            if line.startswith("|") and not re.match(r"^\|-", line):
                cells.append(line[1:].strip())

        if len(cells) < 2:
            continue

        # Host set = first cell
        host_set = cells[0].strip().lower()
        # Skip header-like values
        if not re.match(r"^[a-z0-9]+$", host_set):
            continue

        # Find card name in any cell
        card_name: str | None = None
        for cell in cells:
            m = re.search(r"\{\{card\|([^|{}\n]+?)(?:\|[^}]*)?\}\}", cell)
            if m:
                card_name = m.group(1).strip()
                break
        if not card_name:
            continue

        result[host_set].append(card_name)

    return dict(result)


# ---------------------------------------------------------------------------
# Merge results from all three pages
# ---------------------------------------------------------------------------

def scrape_all() -> dict[str, list[str]]:
    """Return {host_set_code (lower): [card_name, ...]} merged across all three pages."""
    pages = [
        ("The List/ZNR-SNC",    _parse_tick_matrix),
        ("The List/CLB-LCI",    _parse_tick_matrix),
        ("The List/MKM-Present", _parse_row_per_entry),
    ]

    merged: dict[str, list[str]] = defaultdict(list)

    for page_title, parser in pages:
        print(f"  Fetching '{page_title}'...", end=" ", flush=True)
        wikitext = fetch_wikitext(page_title)
        parsed = parser(wikitext)
        total_cards = sum(len(v) for v in parsed.values())
        print(f"{len(parsed)} host sets, {total_cards} card-slot entries")
        for host_set, names in parsed.items():
            merged[host_set].extend(names)

    # Deduplicate within each host set (same card name appears multiple times = same slot repeated)
    return {sc: sorted(set(names)) for sc, names in sorted(merged.items())}


# ---------------------------------------------------------------------------
# Resolve card names → Scryfall UUIDs (plst + spg)
# ---------------------------------------------------------------------------

def resolve_to_card_ids(name_by_host_set: dict[str, list[str]]) -> dict[str, list[str]]:
    """Replace card names with plst/spg Scryfall UUIDs (all printings with that name)."""
    print("  Loading plst + spg cards from catalog...", end=" ", flush=True)
    plst_cards = catalog.cards_for_sets(["plst", "spg"])
    print(f"{len(plst_cards)} cards")

    # Build name (lower) → [card_id]
    name_to_ids: dict[str, list[str]] = defaultdict(list)
    for c in plst_cards:
        name = (c["name"] if isinstance(c, dict) else c.name).strip()
        cid  = (c["id"]   if isinstance(c, dict) else c.id).strip().lower()
        name_to_ids[name.lower()].append(cid)

    result: dict[str, list[str]] = {}
    unresolved: set[str] = set()

    for host_set, names in name_by_host_set.items():
        ids: list[str] = []
        for name in names:
            matches = name_to_ids.get(name.lower(), [])
            if matches:
                ids.extend(matches)
            else:
                unresolved.add(name)
        result[host_set] = sorted(set(ids))

    if unresolved:
        print(f"  WARNING: {len(unresolved)} card name(s) not found in plst/spg catalog:")
        for name in sorted(unresolved)[:20]:
            print(f"    - {name}")
        if len(unresolved) > 20:
            print(f"    ... and {len(unresolved) - 20} more")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Build plst_by_host_set.json from MTG wiki List pages"
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print results without writing file")
    p.add_argument("--output", type=Path,
                   default=Path("02_data_sets/packopening/plst_by_host_set.json"),
                   help="Output path (default: 02_data_sets/packopening/plst_by_host_set.json)")
    args = p.parse_args()

    print("Scraping wiki pages...")
    name_by_host_set = scrape_all()
    print(f"\n  Total host sets with List entries: {len(name_by_host_set)}")

    print("\nResolving card names to plst/spg UUIDs...")
    id_by_host_set = resolve_to_card_ids(name_by_host_set)

    print("\nSummary:")
    for sc, ids in sorted(id_by_host_set.items()):
        print(f"  {sc.upper():<6}  {len(ids):>4} card IDs")

    total_unique_ids = len({cid for ids in id_by_host_set.values() for cid in ids})
    print(f"\n  {total_unique_ids} unique plst/spg card IDs across all host sets")

    if args.dry_run:
        print("\nDry run — not writing file.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(id_by_host_set, f, indent=2, sort_keys=True)
    print(f"\nWritten → {args.output}")


if __name__ == "__main__":
    main()
