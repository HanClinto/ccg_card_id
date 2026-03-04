#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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


def _manifest_key(path: Path) -> str:
    st = path.stat()
    raw = f"{path}:{st.st_mtime_ns}:{st.st_size}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _discover_latest_checkpoint(results_root: Path) -> Path | None:
    cands = sorted(results_root.glob("mobilevit_xxs_arcface_*/last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate MobileViT-XXS retrieval (base + fine-tuned)")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--checkpoint", type=Path, action="append", default=[])
    p.add_argument("--results-root", type=Path, default=DEFAULT_MOBILEVIT_RESULTS, help="Root where MobileViT training runs are stored (used for auto-discovery)")
    p.add_argument("--skip-base", action="store_true")
    p.add_argument("--skip-finetuned", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--worst-n", type=int, default=20)
    p.add_argument("--cache-root", type=Path, default=None, help="Optional embedding cache root (default: <output-root>/cache/mobilevit_retrieval/<manifest_key>)")
    p.add_argument("--rebuild-cache", action="store_true", help="Ignore cached embeddings and recompute")
    p.add_argument("--fp16", dest="fp16", action="store_true", help="Use float16 for similarity search on MPS/CUDA (faster, lower memory)")
    p.add_argument("--no-fp16", dest="fp16", action="store_false", help="Disable float16 similarity search")
    p.set_defaults(fp16=True)
    p.add_argument("--no-write-results", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}. Build it with 04_build/mobilevit_xxs/01_build_manifest.py")

    if args.skip_finetuned:
        args.checkpoint = []
    elif not args.checkpoint:
        auto_ckpt = _discover_latest_checkpoint(args.results_root)
        if auto_ckpt is not None:
            args.checkpoint = [auto_ckpt]
            print(f"Auto-discovered checkpoint: {auto_ckpt}")
        else:
            print(f"No checkpoint found under {args.results_root}; running base-only (or pass --checkpoint).")

    gallery_paths, gallery_ids = load_manifest_gallery(args.manifest)
    query_paths, query_ids = load_solring_queries(args.dataset)
    device = pick_device(force_cpu=args.cpu)

    print(f"Manifest: {args.manifest}")
    print(f"Dataset:  {args.dataset}")
    print(f"Gallery: {len(gallery_paths)} images")
    print(f"Queries: {len(query_paths)} images")
    print(f"Device:  {device}")
    print(f"FP16 search: {args.fp16 and device.type in {'mps', 'cuda'}}")

    manifest_key = _manifest_key(args.manifest)
    cache_root = args.cache_root or (args.output_root / "cache" / "mobilevit_retrieval" / manifest_key)
    cache_root.mkdir(parents=True, exist_ok=True)
    print(f"Embedding cache: {cache_root}")

    summary_rows: list[dict] = []
    failures_all: list[dict] = []

    if not args.skip_base:
        print("Running base MobileViT-XXS retrieval...")
        base = BackboneFeatureModel("mobilevit_xxs").to(device).eval()
        metrics, failures = evaluate_retrieval(
            model=base,
            gallery_paths=gallery_paths,
            gallery_ids=gallery_ids,
            query_paths=query_paths,
            query_ids=query_ids,
            device=device,
            batch_size=args.batch_size,
            image_size=args.image_size,
            label="base",
            cache_root=cache_root,
            rebuild_cache=args.rebuild_cache,
            use_fp16=args.fp16,
        )
        variant = "mobilevit_xxs_base_320d"
        print(f"[{variant}] top1={metrics['top1']:.4f} top3={metrics['top3']:.4f} top10={metrics['top10']:.4f}")
        for k in (1, 3, 10):
            summary_rows.append(
                {
                    "algorithm_variant": variant,
                    "topk": k,
                    "correct": int(round(metrics[f"top{k}"] * metrics["n_queries"])),
                    "total": metrics["n_queries"],
                    "accuracy": metrics[f"top{k}"],
                    "bytes_per_card": 320 * 4,
                }
            )
        for f in failures:
            p = Path(str(f.get("image_path", "")))
            rel = str(p.relative_to(args.dataset)) if p.is_absolute() and str(p).startswith(str(args.dataset)) else str(f.get("image_path", ""))
            failures_all.append({"algorithm_variant": variant, "topk": 1, **f, "image_path": rel})

    for ckpt in args.checkpoint:
        print(f"Running fine-tuned retrieval for checkpoint: {ckpt}")
        model, ckpt_meta = load_finetuned_model(ckpt, device)
        emb_dim = int(ckpt_meta.get("args", {}).get("embedding_dim", 128))
        epoch = int(ckpt_meta.get("epoch", 0))
        variant = f"mobilevit_xxs_ft_e{epoch}_{emb_dim}d"

        metrics, failures = evaluate_retrieval(
            model=model,
            gallery_paths=gallery_paths,
            gallery_ids=gallery_ids,
            query_paths=query_paths,
            query_ids=query_ids,
            device=device,
            batch_size=args.batch_size,
            image_size=args.image_size,
            label=variant,
            cache_root=cache_root,
            rebuild_cache=args.rebuild_cache,
            use_fp16=args.fp16,
        )
        print(f"[{variant}] top1={metrics['top1']:.4f} top3={metrics['top3']:.4f} top10={metrics['top10']:.4f}")

        for k in (1, 3, 10):
            summary_rows.append(
                {
                    "algorithm_variant": variant,
                    "topk": k,
                    "correct": int(round(metrics[f"top{k}"] * metrics["n_queries"])),
                    "total": metrics["n_queries"],
                    "accuracy": metrics[f"top{k}"],
                    "bytes_per_card": emb_dim * 4,
                }
            )
        for f in failures:
            p = Path(str(f.get("image_path", "")))
            rel = str(p.relative_to(args.dataset)) if p.is_absolute() and str(p).startswith(str(args.dataset)) else str(f.get("image_path", ""))
            failures_all.append({"algorithm_variant": variant, "topk": 1, **f, "image_path": rel})

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
        by_variant_summary.setdefault(row["algorithm_variant"], []).append(row)
    for row in failures_all:
        by_variant_failures.setdefault(row["algorithm_variant"], []).append(row)

    for variant, rows in sorted(by_variant_summary.items()):
        write_algorithm_markdown(
            out_dir / f"{variant}.md",
            algorithm_variant=variant,
            summary_rows=rows,
            failures=by_variant_failures.get(variant, []),
            top_n=args.worst_n,
        )

    print(f"Saved run artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
