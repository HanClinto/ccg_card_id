#!/usr/bin/env python3
"""Evaluate MobileViT-XXS retrieval accuracy across multiple query datasets.

Reports two accuracy metrics for each model variant × query dataset:

  artwork  — query matches any gallery card sharing the same illustration_id
             (tests artwork identification regardless of printing)

  edition  — query must match the exact same card (Scryfall card_id)
             (tests printing-level identification)

Built-in query dataset:
  solring  — homography-aligned Sol Ring video frames (--dataset)

Extra query datasets (pass any number of --query-manifest name=path/to/manifest.csv):
  daniel           datasets/daniel_scans/query_manifest.csv
  clint_backgrounds datasets/clint_cards_with_backgrounds/query_manifest.csv
  munchie          datasets/munchie/manifest.csv

Build query manifests first:
  python 02_data_sets/daniel_scans/code/01_build_query_manifest.py
  python 02_data_sets/clint_cards_with_backgrounds/code/01_build_query_manifest.py
  python 02_data_sets/munchie/code/03_build_manifest.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "04_build" / "mobilevit_xxs"))
from ccg_card_id.config import cfg
from reporting import (
    make_run_dir,
    update_central_result_csvs,
    write_algorithm_markdown,
    write_failures_jsonl,
    write_overview_markdown,
    write_summary_csv,
    write_summary_json,
)
from retrieval import (  # type: ignore
    BackboneFeatureModel,
    evaluate_retrieval,
    load_finetuned_model,
    load_manifest_gallery,
    load_query_manifest,
    load_solring_queries,
)

DEFAULT_SOLRING_DATASET = cfg.data_dir / "datasets" / "solring"
DEFAULT_OUTPUT_ROOT = cfg.data_dir / "results" / "eval"
DEFAULT_MANIFEST = cfg.data_dir / "mobilevit_xxs" / "manifest.csv"
DEFAULT_MOBILEVIT_RESULTS = cfg.data_dir / "results" / "mobilevit_xxs"


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _discover_latest_checkpoint(results_root: Path) -> Path | None:
    cands = sorted(
        results_root.glob("mobilevit_xxs_*/last.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def _discover_checkpoints_every_n_epochs(results_root: Path, every: int = 5) -> list[Path]:
    checkpoints: list[tuple[int, Path]] = []
    for run_dir in sorted(results_root.glob("mobilevit_xxs_*")):
        if not run_dir.is_dir():
            continue
        for ckpt in sorted(run_dir.glob("epoch_*.pt")):
            try:
                epoch = int(ckpt.stem.split("_")[-1])
            except ValueError:
                continue
            if every > 0 and (epoch % every != 0):
                continue
            checkpoints.append((epoch, ckpt))
    checkpoints.sort(key=lambda x: (x[0], str(x[1])))
    return [p for _, p in checkpoints]


def _build_card_to_illustration(manifest_csv: Path) -> dict[str, str]:
    """Map card_id (lower) -> illustration_id (lower) from manifest."""
    mapping: dict[str, str] = {}
    with manifest_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = str(row.get("card_id", "")).lower()
            illus = str(row.get("illustration_id", "")).lower()
            if cid and illus:
                mapping[cid] = illus
    return mapping


def _parse_query_manifest_args(specs: list[str]) -> list[tuple[str, Path]]:
    """Parse 'name=path/to/manifest.csv' args into (name, path) pairs."""
    result: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--query-manifest must be 'name=path', got: {spec!r}")
        name, path_str = spec.split("=", 1)
        result.append((name.strip(), Path(path_str.strip())))
    return result


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate MobileViT-XXS retrieval across multiple query datasets"
    )
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                   help="Gallery manifest CSV (Scryfall reference images)")
    p.add_argument("--dataset", type=Path, default=DEFAULT_SOLRING_DATASET,
                   help="Sol Ring dataset dir (built-in query source)")
    p.add_argument("--skip-solring", action="store_true",
                   help="Skip the built-in Sol Ring query set")
    p.add_argument(
        "--query-manifest", action="append", default=[], metavar="NAME=PATH",
        help="Extra query manifest: 'name=path/to/manifest.csv'. Repeatable. "
             "Manifest must have columns: image_path, card_id, illustration_id. "
             "Example: --query-manifest daniel=/data/datasets/daniel_scans/query_manifest.csv",
    )
    p.add_argument("--checkpoint", type=Path, action="append", default=[])
    p.add_argument("--results-root", type=Path, default=DEFAULT_MOBILEVIT_RESULTS)
    p.add_argument("--checkpoint-every", type=int, default=5)
    p.add_argument("--skip-base", action="store_true")
    p.add_argument("--skip-finetuned", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--worst-n", type=int, default=20)
    p.add_argument("--gallery-cache-root", type=Path, default=None)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--fp16", dest="fp16", action="store_true")
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.set_defaults(fp16=True)
    p.add_argument("--no-write-results", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found: {args.manifest}\n"
            "Build it with: python 04_build/mobilevit_xxs/01_build_manifest.py"
        )

    # Resolve checkpoints
    if args.skip_finetuned:
        args.checkpoint = []
    elif not args.checkpoint:
        auto_ckpts = _discover_checkpoints_every_n_epochs(
            args.results_root, every=args.checkpoint_every
        )
        if auto_ckpts:
            args.checkpoint = auto_ckpts
            print(f"Auto-discovered {len(auto_ckpts)} checkpoint(s) under {args.results_root}")
            for ckpt in auto_ckpts:
                print(f"  - {ckpt}")
        else:
            latest = _discover_latest_checkpoint(args.results_root)
            if latest is not None:
                args.checkpoint = [latest]
                print(f"No epoch checkpoints found; using latest: {latest}")
            else:
                print(f"No checkpoints found under {args.results_root}")

    # Load gallery
    gallery_paths, gallery_card_ids, gallery_illus_ids = load_manifest_gallery(args.manifest)
    card_to_illus = _build_card_to_illustration(args.manifest)

    # Collect query sources: list of (name, paths, card_ids, illus_ids, cache_dir)
    QuerySource = tuple  # (name, paths, card_ids, illus_ids, cache_dir)
    query_sources: list[QuerySource] = []

    if not args.skip_solring and args.dataset.exists():
        q_paths, q_card_ids = load_solring_queries(args.dataset)
        q_illus_ids = [card_to_illus.get(cid, "") for cid in q_card_ids]
        n_resolved = sum(1 for x in q_illus_ids if x)
        print(f"solring: {len(q_paths)} queries, {n_resolved}/{len(q_paths)} illustration_ids resolved")
        cache_dir = args.dataset / "cache" / "mobilevit_xxs" / f"img{args.image_size}"
        query_sources.append(("solring", q_paths, q_card_ids, q_illus_ids, cache_dir))

    for name, manifest_path in _parse_query_manifest_args(args.query_manifest):
        if not manifest_path.exists():
            print(f"WARNING: query manifest not found, skipping {name}: {manifest_path}")
            continue
        q_paths, q_card_ids, q_illus_ids = load_query_manifest(manifest_path)
        # Fill in any missing illustration_ids from gallery manifest lookup
        q_illus_ids = [
            illus if illus else card_to_illus.get(cid, "")
            for illus, cid in zip(q_illus_ids, q_card_ids)
        ]
        n_resolved = sum(1 for x in q_illus_ids if x)
        print(f"{name}: {len(q_paths)} queries, {n_resolved}/{len(q_paths)} illustration_ids resolved")
        cache_dir = manifest_path.parent / "cache" / "mobilevit_xxs" / f"img{args.image_size}"
        query_sources.append((name, q_paths, q_card_ids, q_illus_ids, cache_dir))

    if not query_sources:
        print("No query sources available. Exiting.")
        return

    device = pick_device(force_cpu=args.cpu)
    gallery_cache_root = args.gallery_cache_root or (
        cfg.data_dir / "vectors" / "mobilevit_xxs" / f"img{args.image_size}"
        / f"gallery_manifest_{args.manifest.stem}"
    )
    gallery_cache_root.mkdir(parents=True, exist_ok=True)

    print(f"Manifest: {args.manifest}")
    print(f"Gallery:  {len(gallery_paths)} images")
    print(f"Device:   {device}")
    print(f"Query sources: {[s[0] for s in query_sources]}")

    all_summary_rows: list[dict] = []
    all_failures: list[dict] = []

    def _run_model(model: torch.nn.Module, variant: str, emb_dim: int) -> None:
        for (qs_name, q_paths, q_card_ids, q_illus_ids, query_cache_dir) in query_sources:
            query_cache_dir.mkdir(parents=True, exist_ok=True)
            criteria = {
                "artwork": (gallery_illus_ids, q_illus_ids),
                "edition": (gallery_card_ids, q_card_ids),
            }
            results = evaluate_retrieval(
                model=model,
                gallery_paths=gallery_paths,
                query_paths=q_paths,
                criteria=criteria,
                device=device,
                batch_size=args.batch_size,
                image_size=args.image_size,
                label=f"{variant}_{qs_name}",
                gallery_cache_root=gallery_cache_root,
                query_cache_root=query_cache_dir,
                rebuild_cache=args.rebuild_cache,
                use_fp16=args.fp16,
            )
            for criterion_name, (metrics, failures) in results.items():
                print(
                    f"[{variant}][{qs_name}][{criterion_name}] "
                    f"top1={metrics['top1']:.4f}  top3={metrics['top3']:.4f}  top10={metrics['top10']:.4f}"
                )
                for k in (1, 3, 10):
                    all_summary_rows.append({
                        "algorithm_variant": variant,
                        "query_dataset": qs_name,
                        "criterion": criterion_name,
                        "topk": k,
                        "correct": int(round(metrics[f"top{k}"] * metrics["n_queries"])),
                        "total": metrics["n_queries"],
                        "accuracy": metrics[f"top{k}"],
                        "bytes_per_card": emb_dim * 4,
                    })
                for f in failures:
                    path = Path(str(f.get("image_path", "")))
                    # Try to make path relative to query source dir for readability
                    try:
                        rel = str(path.relative_to(query_cache_dir.parent.parent.parent))
                    except ValueError:
                        rel = str(f.get("image_path", ""))
                    all_failures.append({
                        "algorithm_variant": variant,
                        "query_dataset": qs_name,
                        "criterion": criterion_name,
                        "topk": 1,
                        **f,
                        "image_path": rel,
                    })

    if not args.skip_base:
        print("Running base MobileViT-XXS...")
        base = BackboneFeatureModel("mobilevit_xxs").to(device).eval()
        _run_model(base, "mobilevit_xxs_base_320d", emb_dim=320)

    for ckpt in args.checkpoint:
        print(f"Running fine-tuned checkpoint: {ckpt}")
        model, ckpt_meta = load_finetuned_model(ckpt, device)
        emb_dim = int(ckpt_meta.get("args", {}).get("embedding_dim", 128))
        epoch = int(ckpt_meta.get("epoch", 0))
        label_field = str(ckpt_meta.get(
            "label_field", ckpt_meta.get("args", {}).get("label_field", "card_id")
        ))
        variant = f"mobilevit_xxs_ft_{label_field}_e{epoch}_{emb_dim}d"
        _run_model(model, variant, emb_dim=emb_dim)

    if args.no_write_results:
        return

    out_dir = make_run_dir(args.output_root, args.run_id)
    payload = {
        "meta": {
            "script": "06_eval/04_eval_mobilevit_xxs.py",
            "gallery_manifest": str(args.manifest),
            "query_sources": [s[0] for s in query_sources],
            "checkpoints": [str(c) for c in args.checkpoint],
            "skip_base": args.skip_base,
        },
        "summary": all_summary_rows,
    }

    write_summary_csv(out_dir / "summary.csv", all_summary_rows)
    write_summary_json(out_dir / "summary.json", payload)
    write_failures_jsonl(out_dir / "failures.jsonl", all_failures)
    write_overview_markdown(out_dir / "overview.md", all_summary_rows)

    # Update central CSVs per query dataset
    for qs_name in {r["query_dataset"] for r in all_summary_rows}:
        qs_rows = [r for r in all_summary_rows if r["query_dataset"] == qs_name]
        update_central_result_csvs(
            output_root=args.output_root,
            summary_rows=qs_rows,
            benchmark="mobilevit_retrieval",
            dataset=qs_name,
            run_id=out_dir.name,
        )

    # Per-variant per-dataset markdown files
    groups: dict[str, list[dict]] = {}
    failure_groups: dict[str, list[dict]] = {}
    for row in all_summary_rows:
        key = f"{row['algorithm_variant']}_{row['query_dataset']}_{row['criterion']}"
        groups.setdefault(key, []).append(row)
    for row in all_failures:
        key = f"{row['algorithm_variant']}_{row['query_dataset']}_{row['criterion']}"
        failure_groups.setdefault(key, []).append(row)

    for key, rows in sorted(groups.items()):
        write_algorithm_markdown(
            out_dir / f"{key}.md",
            algorithm_variant=key,
            summary_rows=rows,
            failures=failure_groups.get(key, []),
            top_n=args.worst_n,
        )

    print(f"Saved run artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
