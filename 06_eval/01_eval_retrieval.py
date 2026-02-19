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

import argparse
import re
import sys
import time
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from tqdm import tqdm

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccg_card_id.config import cfg
from ccg_card_id.search.brute_force import CardSearchDB

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATASET = Path.home() / "claw" / "data" / "ccg_card_id" / "datasets" / "solring"
DEFAULT_METHODS = ["phash", "whash_db4"]
DEFAULT_SIZES = [32, 64, 128, 256]

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


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(dataset_dir: Path, methods: list[str], sizes: list[int]) -> dict:
    """
    Run retrieval eval for each (method, size) combination.

    Returns a dict: {(method, size): {"correct": int, "total": int, "accuracy": float}}
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

    for method in methods:
        for size in sizes:
            vectors_path = cfg.vectors_file(method, size)
            if not vectors_path.exists():
                print(f"[{method}@{size}] Skipping — vector file not found: {vectors_path}")
                continue

            print(f"[{method}@{size}] Loading reference database ({vectors_path.stat().st_size // 1_000_000}MB)...")
            t0 = time.time()
            db = CardSearchDB.from_phash_json(vectors_path)
            print(f"[{method}@{size}] Loaded {len(db):,} cards in {time.time()-t0:.1f}s")

            correct = 0
            total = 0
            failures = []

            for img_path in tqdm(test_images, desc=f"{method}@{size}", unit="img"):
                true_id = extract_card_id(img_path.name)
                if true_id is None:
                    continue

                h = hash_image(img_path, method, size)
                if h is None:
                    continue

                query_vec = h.hash.flatten().astype(np.int8)
                results_top1 = db.search(query_vec, k=1)

                total += 1
                if results_top1 and results_top1[0].card_id == true_id:
                    correct += 1
                else:
                    best = results_top1[0] if results_top1 else None
                    failures.append((img_path.name, true_id, best.card_id if best else "?", best.score if best else -1))

            accuracy = correct / total if total > 0 else 0.0
            results[(method, size)] = {"correct": correct, "total": total, "accuracy": accuracy}

            print(f"[{method}@{size}] Accuracy: {correct}/{total} = {accuracy*100:.1f}%")
            if failures:
                print(f"  First few failures:")
                for fname, true_id, pred_id, score in failures[:3]:
                    print(f"    {fname}")
                    print(f"      true: {true_id}")
                    print(f"      pred: {pred_id}  (hamming dist: {score:.0f})")
            print()

    return results


def print_summary(results: dict) -> None:
    if not results:
        print("No results.")
        return

    print("=" * 60)
    print(f"{'Method':<14} {'Size':>6}  {'Correct':>8}  {'Total':>8}  {'Accuracy':>9}")
    print("-" * 60)
    for (method, size), r in sorted(results.items()):
        print(f"{method:<14} {size:>6}  {r['correct']:>8}  {r['total']:>8}  {r['accuracy']*100:>8.1f}%")
    print("=" * 60)


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
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    print(f"Methods: {args.methods}")
    print(f"Sizes:   {args.sizes}\n")

    results = run_eval(args.dataset, args.methods, args.sizes)
    print_summary(results)


if __name__ == "__main__":
    main()
