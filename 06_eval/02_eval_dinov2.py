#!/usr/bin/env python3
"""
DINOv2 zero-shot retrieval evaluation.

Mirrors 01_eval_retrieval.py (phash/whash) but uses DINOv2 cosine-similarity
search instead of Hamming distance on perceptual hashes. Results are directly
comparable: same test dataset, same ground-truth extraction, same metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Must be set before torch is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

tqdm.monitor_interval = 0   # no background monitor thread

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccg_card_id.config import cfg, DINOV2_MODELS
from ccg_card_id.search.brute_force import CardSearchDB
from reporting import make_run_dir, update_central_result_csvs, write_algorithm_markdown, write_failures_jsonl, write_overview_markdown, write_summary_csv, write_summary_json

# DINOv2 hub model names
_HUB_NAMES: dict[str, str] = {
    "small": "dinov2_vits14",
    "base":  "dinov2_vitb14",
    "large": "dinov2_vitl14",
    "giant": "dinov2_vitg14",
}

# Standard ImageNet preprocessing expected by DINOv2
_TRANSFORM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_DATASET = cfg.data_dir / "datasets" / "solring"
DEFAULT_MODELS = ["small"]
DEFAULT_BATCH_SIZE = 32
DEFAULT_TOP_K = [1, 3, 10]
DEFAULT_OUTPUT_ROOT = cfg.data_dir / "results" / "eval"
DEFAULT_WORST_N = 20

UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_card_id(filename: str) -> str | None:
    m = UUID_RE.search(filename)
    return m.group(0) if m else None


def _cache_path(output_root: Path, dataset_dir: Path, variant: str) -> Path:
    ds = dataset_dir.name or "dataset"
    return output_root / "cache" / "dinov2_retrieval" / ds / f"dinov2_{variant}.jsonl"


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


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(variant: str, device: torch.device) -> torch.nn.Module:
    hub_name = _HUB_NAMES.get(variant, variant)
    print(f"  Loading {hub_name} via torch.hub on {device}...")
    model = torch.hub.load("facebookresearch/dinov2", hub_name, verbose=False)
    model = model.to(device).eval()
    return model


def embed_images(
    img_paths: list[Path],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[list[Path], np.ndarray]:
    valid_paths: list[Path] = []
    all_embs: list[np.ndarray] = []

    for i in tqdm(range(0, len(img_paths), batch_size), desc="  Embedding test images", unit="batch"):
        batch_paths = img_paths[i : i + batch_size]
        batch_imgs: list[Image.Image] = []
        batch_valid: list[Path] = []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                batch_imgs.append(img)
                batch_valid.append(p)
            except Exception as e:
                print(f"\n  Warning: could not load {p.name}: {e}")

        if batch_imgs:
            tensors = torch.stack([_TRANSFORM(img) for img in batch_imgs]).to(device)
            with torch.inference_mode():
                embs = model(tensors).cpu().float().numpy()
            all_embs.append(embs)
            valid_paths.extend(batch_valid)

    if not all_embs:
        return [], np.empty((0, 0), dtype=np.float32)

    return valid_paths, np.vstack(all_embs)


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(
    dataset_dir: Path,
    variants: list[str],
    batch_size: int,
    top_k: list[int],
    output_root: Path,
    rebuild_cache: bool,
) -> tuple[dict, list[dict]]:
    test_dir = dataset_dir / "04_data" / "aligned"
    if not test_dir.exists():
        sys.exit(f"Test image directory not found: {test_dir}")

    test_images = sorted(
        p for p in test_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not test_images:
        sys.exit(f"No images found in {test_dir}")

    unique_cards = {extract_card_id(p.name) for p in test_images} - {None}
    print(
        f"Found {len(test_images)} test images across "
        f"{len(unique_cards)} unique cards.\n"
    )

    results: dict = {}
    failures: list[dict] = []
    device = get_device()

    for variant in variants:
        npz_path = cfg.dinov2_vectors_file(variant)

        print(f"[dinov2-{variant}]")

        if not npz_path.exists():
            print(f"  Skipping — vector file not found: {npz_path}")
            print(
                "  Build it first:\n"
                f"    python 04_vectorize/dinov2/01_build_vectors.py --model {variant}\n"
            )
            continue

        size_mb = npz_path.stat().st_size // 1_000_000
        print(f"  Loading reference DB ({size_mb} MB)...")
        t0 = time.time()
        db = CardSearchDB.from_dinov2_npz(npz_path)
        print(
            f"  Loaded {len(db):,} cards, dim={db.vectors.shape[1]} "
            f"in {time.time() - t0:.1f}s"
        )

        model = load_model(variant, device)
        valid_paths, query_matrix = embed_images(
            test_images, model, device, batch_size
        )

        del model
        if device.type != "cpu":
            torch.mps.empty_cache() if device.type == "mps" else torch.cuda.empty_cache()

        if len(valid_paths) == 0:
            print("  No embeddings computed — skipping.\n")
            continue

        true_ids = [extract_card_id(p.name) for p in valid_paths]

        valid_mask = [tid is not None for tid in true_ids]
        query_matrix = query_matrix[valid_mask]
        true_ids = [tid for tid in true_ids if tid is not None]
        valid_paths = [p for p, ok in zip(valid_paths, valid_mask) if ok]

        total = len(true_ids)
        if total == 0:
            print("  No valid ground-truth IDs found — skipping.\n")
            continue

        max_k = max(top_k)
        print(f"  Searching {total} queries against {len(db):,} cards (top-{max_k})...")
        t0 = time.time()
        batch_results = db.search_batch(query_matrix, k=max_k)
        print(f"  Search done in {time.time() - t0:.2f}s")

        for k in top_k:
            correct = 0
            local_failures = []

            for img_path, true_id, top_results in zip(valid_paths, true_ids, batch_results):
                top_k_ids = [r.card_id for r in top_results[:k]]
                if true_id in top_k_ids:
                    correct += 1
                elif k == 1:
                    best = top_results[0] if top_results else None
                    true_rank = None
                    for idx, cand in enumerate(top_results, start=1):
                        if cand.card_id == true_id:
                            true_rank = idx
                            break
                    row = {
                        "algorithm_variant": f"dinov2_{variant}",
                        "topk": 1,
                        "image_file": img_path.name,
                        "image_path": str(img_path.relative_to(dataset_dir)),
                        "true_id": true_id,
                        "predicted_id": best.card_id if best else "?",
                        "score": float(best.score) if best else -1.0,
                        "score_type": "cosine_similarity",
                        "true_rank": true_rank,
                    }
                    local_failures.append(row)
                    failures.append(row)

            accuracy = correct / total if total > 0 else 0.0
            results[(variant, k)] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            }
            print(f"  top-{k} accuracy: {correct}/{total} = {accuracy * 100:.1f}%")

            if k == 1 and local_failures:
                local_failures.sort(key=lambda r: r["score"], reverse=True)
                print("  First few top-1 failures:")
                for row in local_failures[:3]:
                    print(f"    {row['image_file']}")
                    print(f"      true: {row['true_id']}")
                    print(f"      pred: {row['predicted_id']}  (cosine sim: {row['score']:.4f})")

        print()

    return results, failures


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: dict, top_k: list[int]) -> None:
    if not results:
        print("No results to display.")
        return

    print("=" * 64)
    if len(top_k) == 1:
        print(f"{'Model':<20} {'Correct':>8}  {'Total':>8}  {'Accuracy':>9}")
        print("-" * 64)
        for (variant, k), r in sorted(results.items()):
            name = f"dinov2-{variant}"
            print(f"{name:<20} {r['correct']:>8}  {r['total']:>8}  {r['accuracy'] * 100:>8.1f}%")
    else:
        ks_str = "  ".join(f"top-{k}" for k in sorted(top_k))
        print(f"{'Model':<20}  {ks_str}")
        print("-" * 64)
        variants = sorted({v for (v, _) in results})
        for variant in variants:
            name = f"dinov2-{variant}"
            accs = "  ".join(
                f"{results[(variant, k)]['accuracy'] * 100:>6.1f}%"
                if (variant, k) in results else "     — "
                for k in sorted(top_k)
            )
            print(f"{name:<20}  {accs}")
    print("=" * 64)


def persist_results(
    dataset: Path,
    models: list[str],
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
            "algorithm_variant": f"dinov2_{variant}",
            "topk": k,
            "correct": r["correct"],
            "total": r["total"],
            "accuracy": r["accuracy"],
        }
        for (variant, k), r in sorted(results.items(), key=lambda x: (x[0][0], x[0][1]))
    ]

    payload = {
        "meta": {
            "script": "06_eval/02_eval_dinov2.py",
            "dataset": str(dataset),
            "models": models,
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
        benchmark="dinov2_retrieval",
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DINOv2 zero-shot card retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to dataset directory (must contain 04_data/aligned/).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=list(DINOV2_MODELS.keys()),
        help="DINOv2 variant(s) to evaluate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Images per embedding batch.",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=DEFAULT_TOP_K,
        help="Recall@k values to report (e.g. 1 5 10).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output directory for result runs.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run id directory name under output-root (default: timestamp).",
    )
    parser.add_argument(
        "--worst-n",
        type=int,
        default=DEFAULT_WORST_N,
        help="Number of worst-case failures to include in markdown reports.",
    )
    parser.add_argument(
        "--no-write-results",
        action="store_true",
        help="Do not write JSON/CSV/Markdown artifacts.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore cached per-image eval JSONL and recompute.",
    )
    args = parser.parse_args()

    print(f"Dataset: {args.dataset}")
    print(f"Models:  {args.models}")
    print(f"Batch:   {args.batch_size}")
    print(f"Top-k:   {args.top_k}")
    print()

    results, failures = run_eval(
        args.dataset,
        args.models,
        args.batch_size,
        args.top_k,
        args.output_root,
        args.rebuild_cache,
    )
    print_summary(results, args.top_k)

    if not args.no_write_results:
        out_dir = persist_results(
            dataset=args.dataset,
            models=args.models,
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
