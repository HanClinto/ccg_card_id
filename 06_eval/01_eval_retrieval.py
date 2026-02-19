"""
Real-world retrieval evaluation using the Sol Ring dataset.

Tests how well each hash method+size can identify a card from a real-world
camera photo, searching against the full Scryfall reference database.

Ground truth: the Scryfall card ID is embedded in each test image filename,
e.g. 0afa0e33-..._solring_khc_20221219_153056.mp4-0000.jpg

Usage:
    cd 06_eval
    python 01_eval_retrieval.py

    # Run specific methods/sizes only:
    python 01_eval_retrieval.py --methods phash --sizes 64 128

    # Point at a different dataset:
    python 01_eval_retrieval.py --dataset ~/claw/data/ccg_card_id/datasets/solring
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccg_card_id.config import cfg
from ccg_card_id.search.brute_force import CardSearchDB
from reporting import make_run_dir, update_central_result_csvs, write_algorithm_markdown, write_failures_jsonl, write_overview_markdown, write_summary_csv, write_summary_json

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATASET = cfg.data_dir / "datasets" / "solring"
DEFAULT_METHODS = ["phash", "whash_db4"]
DEFAULT_SIZES = [32, 64, 128, 256]
DEFAULT_OUTPUT_ROOT = cfg.data_dir / "results" / "eval"
DEFAULT_WORST_N = 20
DEFAULT_TOP_K = [1, 3, 10]


def hash_bytes_per_card(hash_size: int) -> int:
    # hash_size x hash_size bits, packed to bytes
    return (hash_size * hash_size) // 8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def extract_card_id(filename: str) -> str | None:
    """Extract the Scryfall UUID from a filename like '0afa0e33-..._solring_khc_....jpg'."""
    m = UUID_RE.search(filename)
    return m.group(0) if m else None


def hash_image(img_path: Path, method: str, hash_size: int) -> imagehash.ImageHash | None:
    """Hash a single image. Returns None on error."""
    try:
        img = Image.open(img_path).convert("RGB")
        if method == "phash":
            return imagehash.phash(img, hash_size=hash_size)
        elif method == "whash_db4":
            return imagehash.whash(img, hash_size=hash_size, mode="db4")
        elif method == "dhash":
            return imagehash.dhash(img, hash_size=hash_size)
        elif method == "ahash":
            return imagehash.average_hash(img, hash_size=hash_size)
        else:
            raise ValueError(f"Unknown hash method: {method!r}")
    except Exception as e:
        print(f"  Warning: could not hash {img_path.name}: {e}")
        return None


def _cache_path(output_root: Path, dataset_dir: Path, variant: str) -> Path:
    ds = dataset_dir.name or "dataset"
    return output_root / "cache" / "hash_retrieval" / ds / f"{variant}.jsonl"


def _load_cache(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            key = row.get("image_key")
            if isinstance(key, str):
                rows[key] = row
    return rows


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(
    dataset_dir: Path,
    methods: list[str],
    sizes: list[int],
    top_k: list[int],
    output_root: Path,
    rebuild_cache: bool,
) -> tuple[dict, list[dict]]:
    """
    Run retrieval eval for each (method, size) combination.

    Returns
    -------
    (results, failures)
      results: {(method, size): {correct,total,accuracy}}
      failures: per-image miss rows (for failures.jsonl/reporting)
    """
    test_dir = dataset_dir / "04_data" / "aligned"
    if not test_dir.exists():
        sys.exit(f"Test image directory not found: {test_dir}")

    test_images = sorted(p for p in test_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not test_images:
        sys.exit(f"No images found in {test_dir}")

    print(f"Found {len(test_images)} test images across "
          f"{len({extract_card_id(p.name) for p in test_images} - {None})} unique cards.\n")

    results = {}
    failures: list[dict] = []

    for method in methods:
        for size in sizes:
            variant = f"{method}_{size}"
            vectors_path = cfg.vectors_file(method, size)
            if not vectors_path.exists():
                print(f"[{method}@{size}] Skipping — vector file not found: {vectors_path}")
                continue

            print(f"[{method}@{size}] Loading reference database ({vectors_path.stat().st_size // 1_000_000}MB)...")
            t0 = time.time()
            db = CardSearchDB.from_phash_json(vectors_path)
            print(f"[{method}@{size}] Loaded {len(db):,} cards in {time.time()-t0:.1f}s")

            max_k = max(top_k)
            correct_by_k = {k: 0 for k in top_k}
            total = 0
            local_failures = []

            cache_path = _cache_path(output_root, dataset_dir, variant)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_rows = {} if rebuild_cache else _load_cache(cache_path)
            cache_mode = "w" if rebuild_cache else "a"
            cache_hits = 0
            cache_writes = 0

            with cache_path.open(cache_mode, encoding="utf-8") as cache_out:
                for img_path in tqdm(test_images, desc=f"{method}@{size}", unit="img"):
                    true_id = extract_card_id(img_path.name)
                    if true_id is None:
                        continue

                    image_key = str(img_path.relative_to(dataset_dir))
                    row = cache_rows.get(image_key)

                    if row is None:
                        h = hash_image(img_path, method, size)
                        if h is None:
                            continue

                        t_eval0 = time.perf_counter()
                        query_vec = h.hash.flatten().astype(np.int8)
                        top_results = db.search(query_vec, k=max_k)
                        eval_ms = (time.perf_counter() - t_eval0) * 1000.0

                        top_ids = [r.card_id for r in top_results]
                        top_scores = [float(r.score) for r in top_results]
                        true_rank = next((i for i, cid in enumerate(top_ids, start=1) if cid == true_id), None)

                        row = {
                            "image_key": image_key,
                            "algorithm_variant": variant,
                            "benchmark": "hash_retrieval",
                            "dataset": str(dataset_dir),
                            "evaluated_at": datetime.now(timezone.utc).isoformat(),
                            "eval_ms": round(eval_ms, 3),
                            "true_id": true_id,
                            "top_ids": top_ids,
                            "top_scores": top_scores,
                            "correct_top1": true_id in top_ids[:1],
                            "correct_top3": true_id in top_ids[:3],
                            "correct_top10": true_id in top_ids[:10],
                            "true_rank": true_rank,
                        }
                        cache_out.write(json.dumps(row) + "\n")
                        cache_out.flush()
                        cache_writes += 1
                    else:
                        cache_hits += 1

                    top_ids = row.get("top_ids", [])
                    total += 1
                    for k in top_k:
                        if true_id in top_ids[:k]:
                            correct_by_k[k] += 1

                    if true_id not in top_ids[:1]:
                        top_scores = row.get("top_scores", [])
                        row_fail = {
                            "algorithm_variant": variant,
                            "topk": 1,
                            "image_file": img_path.name,
                            "image_path": image_key,
                            "true_id": true_id,
                            "predicted_id": top_ids[0] if top_ids else "?",
                            "score": float(top_scores[0]) if top_scores else -1.0,
                            "score_type": "hamming_distance",
                            "true_rank": row.get("true_rank"),
                        }
                        local_failures.append(row_fail)
                        failures.append(row_fail)

            print(f"  Cache: hits={cache_hits}, new={cache_writes}, file={cache_path}")

            for k in top_k:
                correct = correct_by_k[k]
                accuracy = correct / total if total > 0 else 0.0
                results[(method, size, k)] = {"correct": correct, "total": total, "accuracy": accuracy}
                print(f"[{method}@{size}] top-{k} accuracy: {correct}/{total} = {accuracy*100:.1f}%")

            if local_failures:
                local_failures.sort(key=lambda r: r["score"], reverse=True)
                print(f"  First few top-1 failures:")
                for row in local_failures[:3]:
                    print(f"    {row['image_file']}")
                    print(f"      true: {row['true_id']}")
                    print(f"      pred: {row['predicted_id']}  (hamming dist: {row['score']:.0f})")
            print()

    return results, failures


def print_summary(results: dict, top_k: list[int]) -> None:
    if not results:
        print("No results.")
        return

    print("=" * 72)
    if len(top_k) == 1:
        print(f"{'Method':<14} {'Size':>6}  {'Correct':>8}  {'Total':>8}  {'Accuracy':>9}")
        print("-" * 72)
        for (method, size, _), r in sorted(results.items()):
            print(f"{method:<14} {size:>6}  {r['correct']:>8}  {r['total']:>8}  {r['accuracy']*100:>8.1f}%")
    else:
        ks_str = "  ".join(f"top-{k}" for k in sorted(top_k))
        print(f"{'Method':<14} {'Size':>6}  {ks_str}")
        print("-" * 72)
        combos = sorted({(m, s) for (m, s, _) in results})
        for method, size in combos:
            accs = "  ".join(
                f"{results[(method, size, k)]['accuracy']*100:>6.1f}%"
                if (method, size, k) in results else "     — "
                for k in sorted(top_k)
            )
            print(f"{method:<14} {size:>6}  {accs}")
    print("=" * 72)


def persist_results(
    dataset: Path,
    methods: list[str],
    sizes: list[int],
    top_k: list[int],
    results: dict,
    failures: list[dict],
    output_root: Path,
    run_id: str | None,
    worst_n: int,
) -> Path:
    out_dir = make_run_dir(output_root, run_id)

    summary_rows = [
        {
            "algorithm_variant": f"{method}_{size}",
            "topk": k,
            "correct": r["correct"],
            "total": r["total"],
            "accuracy": r["accuracy"],
            "bytes_per_card": hash_bytes_per_card(size),
        }
        for (method, size, k), r in sorted(results.items(), key=lambda x: (x[0][0], x[0][1], x[0][2]))
    ]

    payload = {
        "meta": {
            "script": "06_eval/01_eval_retrieval.py",
            "dataset": str(dataset),
            "methods": methods,
            "sizes": sizes,
            "top_k": top_k,
        },
        "summary": summary_rows,
    }

    write_summary_csv(out_dir / "summary.csv", summary_rows)
    write_summary_json(out_dir / "summary.json", payload)
    write_failures_jsonl(out_dir / "failures.jsonl", failures)
    write_overview_markdown(out_dir / "overview.md", summary_rows)

    update_central_result_csvs(
        output_root=output_root,
        summary_rows=summary_rows,
        benchmark="hash_retrieval",
        dataset=str(dataset),
        run_id=out_dir.name,
    )

    by_variant: dict[str, list[dict]] = {}
    for row in failures:
        by_variant.setdefault(row["algorithm_variant"], []).append(row)
    for variant, rows in by_variant.items():
        rows.sort(key=lambda r: r["score"], reverse=True)

    by_variant_summary: dict[str, list[dict]] = {}
    for row in summary_rows:
        by_variant_summary.setdefault(row["algorithm_variant"], []).append(row)

    for variant, rows in sorted(by_variant_summary.items()):
        write_algorithm_markdown(
            out_dir / f"{variant}.md",
            algorithm_variant=variant,
            summary_rows=rows,
            failures=by_variant.get(variant, []),
            top_n=worst_n,
        )

    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CCG card retrieval evaluation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help=f"Path to dataset directory (default: {DEFAULT_DATASET})")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS,
                        choices=["phash", "whash_db4", "dhash", "ahash"],
                        help="Hash methods to evaluate")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES,
                        help="Hash grid sizes to evaluate")
    parser.add_argument("--top-k", nargs="+", type=int, default=DEFAULT_TOP_K,
                        help="Recall@k values to report (e.g. 1 3 10)")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help=f"Root output directory for result runs (default: {DEFAULT_OUTPUT_ROOT})")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Optional run id directory name under output-root (default: timestamp)")
    parser.add_argument("--worst-n", type=int, default=DEFAULT_WORST_N,
                        help="Number of worst-case failures to include in markdown reports")
    parser.add_argument("--no-write-results", action="store_true",
                        help="Do not write JSON/CSV/Markdown artifacts")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="Ignore cached per-image eval JSONL and recompute")
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    print(f"Methods: {args.methods}")
    print(f"Sizes:   {args.sizes}")
    print(f"Top-k:   {args.top_k}\n")

    results, failures = run_eval(
        args.dataset,
        args.methods,
        args.sizes,
        args.top_k,
        args.output_root,
        args.rebuild_cache,
    )
    print_summary(results, args.top_k)

    if not args.no_write_results:
        out_dir = persist_results(
            dataset=args.dataset,
            methods=args.methods,
            sizes=args.sizes,
            top_k=args.top_k,
            results=results,
            failures=failures,
            output_root=args.output_root,
            run_id=args.run_id,
            worst_n=args.worst_n,
        )
        print(f"\nSaved run artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
