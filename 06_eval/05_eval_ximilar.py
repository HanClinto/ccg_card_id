#!/usr/bin/env python3
"""Evaluate Ximilar TCG card identification API on solring and daniel query datasets.

Metrics (matching our ArcFace eval):
  artwork  — Ximilar returns the correct card name (e.g. "Sol Ring")
  edition  — Ximilar returns the correct set_code + collector_number → resolves to correct card_id

All API responses are cached as JSON under:
  {data_dir}/results/ximilar_cache/{dataset}/{image_stem}.json

Usage (from project root):
  python 06_eval/05_eval_ximilar.py
  python 06_eval/05_eval_ximilar.py --skip-daniel      # solring only
  python 06_eval/05_eval_ximilar.py --dry-run          # print plan, no API calls
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ccg_card_id.config import cfg

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)

XIMILAR_ENDPOINT = "https://api.ximilar.com/collectibles/v2/tcg_id"
BATCH_SIZE = 10   # Ximilar supports multiple records per request
RETRY_DELAY = 5   # seconds between retries on 429/5xx


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------

def open_catalog() -> sqlite3.Connection:
    db_path = Path("/Users/hanclaw/claw/fast_data/ccg_card_id/catalog/scryfall/cards.db")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def build_card_lookup(con: sqlite3.Connection) -> tuple[dict, dict]:
    """Return (id_to_name, (set_code, collector_number) -> card_id) mappings."""
    id_to_name: dict[str, str] = {}
    setnum_to_id: dict[tuple[str, str], str] = {}
    for row in con.execute("SELECT id, name, set_code, collector_number FROM cards WHERE lang='en'"):
        id_to_name[row["id"]] = row["name"]
        setnum_to_id[(row["set_code"].lower(), str(row["collector_number"]).lower())] = row["id"]
    return id_to_name, setnum_to_id


# ---------------------------------------------------------------------------
# Query loading
# ---------------------------------------------------------------------------

def load_solring_queries(data_dir: Path) -> list[tuple[Path, str]]:
    """Return [(image_path, card_id), ...] for all solring aligned frames."""
    aligned_dir = data_dir / "datasets" / "solring" / "04_data" / "aligned"
    rows = []
    for p in sorted(aligned_dir.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        m = UUID_RE.search(p.name)
        if m:
            rows.append((p, m.group(0).lower()))
    return rows


def load_daniel_queries(data_dir: Path) -> list[tuple[Path, str]]:
    """Return [(image_path, card_id), ...] from daniel_scans query manifest."""
    manifest = data_dir / "datasets" / "daniel_scans" / "query_manifest.csv"
    rows = []
    with open(manifest) as f:
        for r in csv.DictReader(f):
            p = Path(r["image_path"])
            if not p.is_absolute():
                p = data_dir / p
            rows.append((p, r["card_id"].lower()))
    return rows


# ---------------------------------------------------------------------------
# Ximilar API
# ---------------------------------------------------------------------------

def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def call_ximilar_batch(
    paths: list[Path],
    api_key: str,
    retries: int = 3,
) -> list[dict]:
    """POST a batch of images to Ximilar; return list of record dicts."""
    records = [{"_base64": _encode_image(p)} for p in paths]
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json",
    }
    payload = {"records": records}

    for attempt in range(retries):
        try:
            resp = requests.post(XIMILAR_ENDPOINT, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"  HTTP {resp.status_code}, retrying in {wait}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json().get("records", [])
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  Request error: {e}, retrying…")
            time.sleep(RETRY_DELAY)
    return []


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def cache_path(cache_dir: Path, image_path: Path) -> Path:
    return cache_dir / (image_path.stem + ".json")


def load_cached(cache_dir: Path, image_path: Path) -> dict | None:
    p = cache_path(cache_dir, image_path)
    if p.exists():
        return json.loads(p.read_text())
    return None


def save_cache(cache_dir: Path, image_path: Path, record: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path(cache_dir, image_path).write_text(json.dumps(record, indent=2))


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def extract_best_match(record: dict) -> dict | None:
    """Pull the best_match dict out of a Ximilar response record."""
    objects = record.get("_objects", [])
    if not objects:
        return None
    ident = objects[0].get("_identification", {})
    return ident.get("best_match") or None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    queries: list[tuple[Path, str]],
    cache_dir: Path,
    api_key: str,
    id_to_name: dict[str, str],
    setnum_to_id: dict[tuple[str, str], str],
    dry_run: bool = False,
    dataset_name: str = "",
) -> dict:
    """Query API (with caching) and compute artwork + edition top-1 accuracy."""
    n_total = len(queries)
    n_cached = sum(1 for p, _ in queries if load_cached(cache_dir, p) is not None)
    n_to_fetch = n_total - n_cached
    print(f"\n{dataset_name}: {n_total} queries ({n_cached} cached, {n_to_fetch} to fetch)")

    if dry_run:
        print("  [dry-run] skipping API calls")
        return {}

    # Fetch uncached in batches
    uncached = [(p, cid) for p, cid in queries if load_cached(cache_dir, p) is None]
    for i in range(0, len(uncached), BATCH_SIZE):
        batch = uncached[i : i + BATCH_SIZE]
        paths = [p for p, _ in batch]
        print(f"  fetching batch {i//BATCH_SIZE + 1}/{(len(uncached)+BATCH_SIZE-1)//BATCH_SIZE} "
              f"({len(paths)} images)…", end=" ", flush=True)
        records = call_ximilar_batch(paths, api_key)
        for (p, _), rec in zip(batch, records):
            save_cache(cache_dir, p, rec)
        print("ok")
        time.sleep(0.2)  # gentle rate limiting

    # Score
    artwork_correct = 0
    edition_correct = 0
    results = []

    for img_path, true_card_id in queries:
        record = load_cached(cache_dir, img_path)
        best = extract_best_match(record) if record else None

        true_name = id_to_name.get(true_card_id, "")
        pred_name = best.get("name", "").strip() if best else ""
        pred_set = (best.get("set_code") or "").strip().lower() if best else ""
        pred_num = str(best.get("card_number") or "").strip().lower() if best else ""

        # Artwork: correct card name (case-insensitive)
        art_hit = pred_name.lower() == true_name.lower() if pred_name and true_name else False

        # Edition: resolve (set_code, collector_number) → card_id and compare
        pred_card_id = setnum_to_id.get((pred_set, pred_num)) if pred_set and pred_num else None
        ed_hit = pred_card_id == true_card_id if pred_card_id else False

        artwork_correct += int(art_hit)
        edition_correct += int(ed_hit)
        results.append({
            "image": img_path.name,
            "true_card_id": true_card_id,
            "true_name": true_name,
            "pred_name": pred_name,
            "pred_set": pred_set,
            "pred_num": pred_num,
            "pred_card_id": pred_card_id or "",
            "artwork_hit": art_hit,
            "edition_hit": ed_hit,
        })

    art_acc = artwork_correct / n_total
    ed_acc = edition_correct / n_total
    print(f"  artwork top-1:  {artwork_correct}/{n_total} = {art_acc:.1%}")
    print(f"  edition top-1:  {edition_correct}/{n_total} = {ed_acc:.1%}")

    return {
        "dataset": dataset_name,
        "n_total": n_total,
        "artwork_correct": artwork_correct,
        "edition_correct": edition_correct,
        "artwork_accuracy": art_acc,
        "edition_accuracy": ed_acc,
        "rows": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate Ximilar TCG API on query datasets")
    p.add_argument("--skip-solring", action="store_true")
    p.add_argument("--skip-daniel", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Plan only, no API calls")
    p.add_argument("--output", type=Path,
                   default=Path("06_eval/ximilar_eval_results.csv"))
    args = p.parse_args()

    # Load API key from .env
    api_key = None
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("XIMILAR_API_KEY="):
                api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not api_key:
        sys.exit("XIMILAR_API_KEY not found in .env")

    data_dir = cfg.data_dir
    cache_root = data_dir / "results" / "ximilar_cache"

    print("Loading catalog…")
    con = open_catalog()
    id_to_name, setnum_to_id = build_card_lookup(con)
    print(f"  {len(id_to_name):,} cards, {len(setnum_to_id):,} set+number entries")

    all_results = []

    if not args.skip_solring:
        queries = load_solring_queries(data_dir)
        r = evaluate_dataset(
            queries,
            cache_dir=cache_root / "solring",
            api_key=api_key,
            id_to_name=id_to_name,
            setnum_to_id=setnum_to_id,
            dry_run=args.dry_run,
            dataset_name="solring",
        )
        if r:
            all_results.append(r)

    if not args.skip_daniel:
        queries = load_daniel_queries(data_dir)
        r = evaluate_dataset(
            queries,
            cache_dir=cache_root / "daniel",
            api_key=api_key,
            id_to_name=id_to_name,
            setnum_to_id=setnum_to_id,
            dry_run=args.dry_run,
            dataset_name="daniel",
        )
        if r:
            all_results.append(r)

    if args.dry_run or not all_results:
        return

    # Write per-query CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "dataset", "image", "true_card_id", "true_name",
            "pred_name", "pred_set", "pred_num", "pred_card_id",
            "artwork_hit", "edition_hit",
        ])
        writer.writeheader()
        for res in all_results:
            for row in res["rows"]:
                writer.writerow({"dataset": res["dataset"], **row})

    print(f"\nPer-query results → {args.output}")

    # Summary
    print("\n=== Summary ===")
    print(f"{'Dataset':<12} {'Artwork':>10} {'Edition':>10}")
    for res in all_results:
        print(f"{res['dataset']:<12} {res['artwork_accuracy']:>10.1%} {res['edition_accuracy']:>10.1%}")


if __name__ == "__main__":
    main()
