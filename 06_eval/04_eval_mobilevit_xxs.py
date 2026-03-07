#!/usr/bin/env python3
"""Evaluate MobileViT-XXS retrieval accuracy on the Sol Ring query set.

Reports two accuracy metrics for each model variant:

  artwork  — query matches any gallery card sharing the same illustration_id
             (tests whether the model can identify the artwork regardless of printing)

  edition  — query must match the exact same card (card_id / Scryfall UUID)
             (tests whether the model can distinguish specific printings)
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
    load_solring_queries,
)

DEFAULT_DATASET = cfg.data_dir / "datasets" / "solring"
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


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate MobileViT-XXS retrieval (artwork + edition accuracy)"
    )
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
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
    p.add_argument("--query-cache-root", type=Path, default=None)
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
            "Build it with: python 04_build/mobilevit_xxs/01_build_manifest.py  "
            "(for scryfall-only manifest.csv)"
        )

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

    # Load gallery — card_ids and illustration_ids from manifest
    gallery_paths, gallery_card_ids, gallery_illus_ids = load_manifest_gallery(args.manifest)

    # Load sol_ring queries (card_ids from filenames)
    query_paths, query_card_ids = load_solring_queries(args.dataset)

    # Map query card_ids → illustration_ids via manifest lookup
    card_to_illus = _build_card_to_illustration(args.manifest)
    query_illus_ids = [card_to_illus.get(cid, "") for cid in query_card_ids]

    n_artwork_lookups = sum(1 for x in query_illus_ids if x)
    print(f"Query illustration_id lookup: {n_artwork_lookups}/{len(query_card_ids)} resolved")
    if n_artwork_lookups < len(query_card_ids):
        print("  WARNING: some query card_ids not found in manifest — artwork accuracy may be understated")

    device = pick_device(force_cpu=args.cpu)
    print(f"Manifest: {args.manifest}")
    print(f"Dataset:  {args.dataset}")
    print(f"Gallery:  {len(gallery_paths)} images")
    print(f"Queries:  {len(query_paths)} images")
    print(f"Device:   {device}")

    gallery_cache_root = args.gallery_cache_root or (
        cfg.data_dir / "vectors" / "mobilevit_xxs" / f"img{args.image_size}"
        / f"gallery_manifest_{args.manifest.stem}"
    )
    query_cache_root = args.query_cache_root or (
        args.dataset / "cache" / "mobilevit_xxs" / f"img{args.image_size}"
    )
    gallery_cache_root.mkdir(parents=True, exist_ok=True)
    query_cache_root.mkdir(parents=True, exist_ok=True)

    # Evaluation criteria: embed once, score both
    criteria = {
        "artwork": (gallery_illus_ids, query_illus_ids),
        "edition": (gallery_card_ids, query_card_ids),
    }

    summary_rows: list[dict] = []
    failures_all: list[dict] = []

    def _run_eval(model: torch.nn.Module, variant: str, emb_dim: int) -> None:
        results = evaluate_retrieval(
            model=model,
            gallery_paths=gallery_paths,
            query_paths=query_paths,
            criteria=criteria,
            device=device,
            batch_size=args.batch_size,
            image_size=args.image_size,
            label=variant,
            gallery_cache_root=gallery_cache_root,
            query_cache_root=query_cache_root,
            rebuild_cache=args.rebuild_cache,
            use_fp16=args.fp16,
        )
        for criterion_name, (metrics, failures) in results.items():
            print(
                f"[{variant}][{criterion_name}] "
                f"top1={metrics['top1']:.4f}  top3={metrics['top3']:.4f}  top10={metrics['top10']:.4f}"
            )
            for k in (1, 3, 10):
                summary_rows.append({
                    "algorithm_variant": variant,
                    "criterion": criterion_name,
                    "topk": k,
                    "correct": int(round(metrics[f"top{k}"] * metrics["n_queries"])),
                    "total": metrics["n_queries"],
                    "accuracy": metrics[f"top{k}"],
                    "bytes_per_card": emb_dim * 4,
                })
            for f in failures:
                path = Path(str(f.get("image_path", "")))
                rel = (
                    str(path.relative_to(args.dataset))
                    if path.is_absolute() and str(path).startswith(str(args.dataset))
                    else str(f.get("image_path", ""))
                )
                failures_all.append({
                    "algorithm_variant": variant,
                    "criterion": criterion_name,
                    "topk": 1,
                    **f,
                    "image_path": rel,
                })

    if not args.skip_base:
        print("Running base MobileViT-XXS...")
        base = BackboneFeatureModel("mobilevit_xxs").to(device).eval()
        _run_eval(base, "mobilevit_xxs_base_320d", emb_dim=320)

    for ckpt in args.checkpoint:
        print(f"Running fine-tuned checkpoint: {ckpt}")
        model, ckpt_meta = load_finetuned_model(ckpt, device)
        emb_dim = int(ckpt_meta.get("args", {}).get("embedding_dim", 128))
        epoch = int(ckpt_meta.get("epoch", 0))
        label_field = str(ckpt_meta.get("label_field", ckpt_meta.get("args", {}).get("label_field", "card_id")))
        variant = f"mobilevit_xxs_ft_{label_field}_e{epoch}_{emb_dim}d"
        _run_eval(model, variant, emb_dim=emb_dim)

    if args.no_write_results:
        return

    out_dir = make_run_dir(args.output_root, args.run_id)
    payload = {
        "meta": {
            "script": "06_eval/04_eval_mobilevit_xxs.py",
            "dataset": str(args.dataset),
            "manifest": str(args.manifest),
            "checkpoints": [str(c) for c in args.checkpoint],
            "skip_base": args.skip_base,
        },
        "summary": summary_rows,
    }

    write_summary_csv(out_dir / "summary.csv", summary_rows)
    write_summary_json(out_dir / "summary.json", payload)
    write_failures_jsonl(out_dir / "failures.jsonl", failures_all)
    write_overview_markdown(out_dir / "overview.md", summary_rows)

    update_central_result_csvs(
        output_root=args.output_root,
        summary_rows=summary_rows,
        benchmark="mobilevit_retrieval",
        dataset=str(args.dataset),
        run_id=out_dir.name,
    )

    by_variant_summary: dict[str, list[dict]] = {}
    by_variant_failures: dict[str, list[dict]] = {}
    for row in summary_rows:
        key = f"{row['algorithm_variant']}_{row['criterion']}"
        by_variant_summary.setdefault(key, []).append(row)
    for row in failures_all:
        key = f"{row['algorithm_variant']}_{row['criterion']}"
        by_variant_failures.setdefault(key, []).append(row)

    for key, rows in sorted(by_variant_summary.items()):
        write_algorithm_markdown(
            out_dir / f"{key}.md",
            algorithm_variant=key,
            summary_rows=rows,
            failures=by_variant_failures.get(key, []),
            top_n=args.worst_n,
        )

    print(f"Saved run artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
