#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate MobileViT-XXS retrieval (base + fine-tuned)")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--checkpoint", type=Path, action="append", default=[])
    p.add_argument("--skip-base", action="store_true")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--worst-n", type=int, default=20)
    p.add_argument("--no-write-results", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    gallery_paths, gallery_ids = load_manifest_gallery(args.manifest)
    query_paths, query_ids = load_solring_queries(args.dataset)
    device = pick_device(force_cpu=args.cpu)

    print(f"Gallery: {len(gallery_paths)} images")
    print(f"Queries: {len(query_paths)} images")
    print(f"Device:  {device}")

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
