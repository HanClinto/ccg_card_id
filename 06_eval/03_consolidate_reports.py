#!/usr/bin/env python3
"""Consolidate eval run artifacts into a shareable Markdown/PDF report.

Inputs:
  - per-run folders under CCG_DATA_DIR/results/eval/<run_id>
  - summary.csv + failures.jsonl in each run folder
  - optional latest_results.csv/history_results.csv in eval root

Outputs:
  - markdown report
  - optional PDF (if pandoc is available)
"""

from __future__ import annotations

import argparse
import csv
import sys
import json
import shutil
import subprocess
from collections import defaultdict
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccg_card_id.config import cfg
from ccg_card_id.catalog import catalog


DEFAULT_RESULTS_ROOT = cfg.data_dir / "results" / "eval"
DEFAULT_REPORTS_ROOT = cfg.data_dir / "results" / "reports"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def _img_md(path: Path, reports_root: Path, assets_dir: Path) -> str:
    """Copy image into report-local assets dir and return relative markdown ref."""
    assets_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    target = assets_dir / f"{digest}{path.suffix.lower()}"
    if not target.exists():
        shutil.copy2(path, target)
    rel = target.relative_to(reports_root)
    return f'![]({rel.as_posix()})'


def _front_image_for_id(card_id: str) -> Path | None:
    m = UUID_RE.search(card_id or "")
    if not m:
        return None
    cid = m.group(0).lower()
    p = cfg.scryfall_images_dir / "front" / cid[0] / cid[1] / f"{cid}.png"
    return p if p.exists() else None


def _load_card_index() -> dict[str, dict[str, str]]:
    try:
        set_names = catalog.set_names()
        return {
            c["id"].lower(): {
                "name": c["name"],
                "set": c["set_code"].upper(),
                "set_name": set_names.get(c["set_code"], ""),
            }
            for c in catalog.all_cards()
        }
    except Exception:
        return {}


def _parse_input_label(image_key: str, card_index: dict[str, dict[str, str]]) -> str:
    base = Path(image_key).name
    parts = base.split("_")
    if len(parts) >= 4:
        sample = "_".join(parts[3:])
        sample = re.sub(r"\.[^.]+$", "", sample)
        return sample
    return re.sub(r"\.[^.]+$", "", base)


def _label_for_card_id(card_id: str, card_index: dict[str, dict[str, str]]) -> str:
    cid = (card_id or "").lower()
    meta = card_index.get(cid, {})
    name = meta.get("name") or "Unknown"
    set_code = meta.get("set") or "?"
    set_name = meta.get("set_name") or "Unknown set"
    return f"{name} [{set_code}] ({set_name})"


def _load_variant_cache_rows(results_root: Path, variant: str) -> list[dict]:
    rows: list[dict] = []

    # Hash per-image cache rows
    for jf in (results_root / "cache" / "hash_retrieval").rglob("*.jsonl"):
        recs = _read_jsonl(jf)
        for r in recs:
            if r.get("algorithm_variant") == variant:
                rows.append(r)

    # DINO per-image cache rows (if present)
    for jf in (results_root / "cache" / "dinov2_retrieval").rglob("*.jsonl"):
        recs = _read_jsonl(jf)
        for r in recs:
            if r.get("algorithm_variant") == variant:
                rows.append(r)

    # Fallback: derive from run failures.jsonl (top-1 misses)
    if not rows:
        for run in _collect_runs(results_root):
            for r in _read_jsonl(run / "failures.jsonl"):
                if r.get("algorithm_variant") != variant:
                    continue
                pred = r.get("predicted_id")
                rows.append({
                    "algorithm_variant": variant,
                    "dataset": str(cfg.data_dir / "datasets" / "solring"),
                    "image_key": r.get("image_path", ""),
                    "true_id": r.get("true_id"),
                    "top_ids": [pred] if pred else [],
                    "top_scores": [r.get("score")] if r.get("score") is not None else [],
                    "true_rank": r.get("true_rank"),
                })
    return rows


def _pick_failure_examples(cache_rows: list[dict]) -> dict[str, dict | None]:
    buckets = {"top1": None, "top3": None, "top10": None}

    def to_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in {"1", "true", "yes"}
        return None

    def flags(r: dict):
        c1 = to_bool(r.get("correct_top1"))
        c3 = to_bool(r.get("correct_top3"))
        c10 = to_bool(r.get("correct_top10"))
        tr = r.get("true_rank")
        tr_i = int(tr) if isinstance(tr, int) or (isinstance(tr, str) and tr.isdigit()) else None
        if c1 is None and tr_i is not None:
            c1 = tr_i <= 1
        if c3 is None and tr_i is not None:
            c3 = tr_i <= 3
        if c10 is None and tr_i is not None:
            c10 = tr_i <= 10
        return c1, c3, c10

    for r in cache_rows:
        c1, c3, c10 = flags(r)
        if buckets["top1"] is None and c1 is False and c3 is True:
            buckets["top1"] = r
        if buckets["top3"] is None and c3 is False and c10 is True:
            buckets["top3"] = r
        if buckets["top10"] is None and c10 is False:
            buckets["top10"] = r

    for r in cache_rows:
        c1, c3, c10 = flags(r)
        if buckets["top1"] is None and c1 is False:
            buckets["top1"] = r
        if buckets["top3"] is None and c3 is False:
            buckets["top3"] = r
        if buckets["top10"] is None and (c10 is False or c10 is None):
            buckets["top10"] = r

    return buckets


def _collect_runs(results_root: Path) -> list[Path]:
    runs: list[Path] = []
    if not results_root.exists():
        return runs
    for p in results_root.rglob("summary.csv"):
        run_dir = p.parent
        if "cache" in run_dir.parts:
            continue
        runs.append(run_dir)
    runs = sorted(set(runs))
    return runs


def _bytes_per_card_from_variant(variant: str) -> int | None:
    # Hash variants: method_<size>
    try:
        size = int(variant.split("_")[-1])
        return (size * size) // 8
    except Exception:
        pass

    # DINOv2 variants (float32 vectors)
    dinov2_dims = {
        "dinov2_small": 384,
        "dinov2_base": 768,
        "dinov2_large": 1024,
        "dinov2_giant": 1536,
    }
    if variant in dinov2_dims:
        return dinov2_dims[variant] * 4

    return None


def _metrics_from_cache(results_root: Path) -> list[dict]:
    cache_dir = results_root / "cache" / "hash_retrieval"
    rows: list[dict] = []
    if not cache_dir.exists():
        return rows
    for jf in cache_dir.rglob("*.jsonl"):
        recs = _read_jsonl(jf)
        if not recs:
            continue
        variant = recs[0].get("algorithm_variant", jf.stem)
        total = len(recs)
        for k in (1, 3, 10):
            correct = 0
            for r in recs:
                top_ids = r.get("top_ids", [])
                true_id = r.get("true_id")
                if true_id in top_ids[:k]:
                    correct += 1
            rows.append({
                "run_id": "cache_snapshot",
                "algorithm_variant": variant,
                "topk": str(k),
                "correct": str(correct),
                "total": str(total),
                "accuracy": str(correct / total if total else 0.0),
                "bytes_per_card": str(_bytes_per_card_from_variant(variant) or ""),
            })
    return rows


def _latest_metrics(results_root: Path, runs: list[Path]) -> list[dict]:
    latest_csv = results_root / "latest_results.csv"
    rows = _read_csv(latest_csv)
    if rows:
        return rows

    # fallback: latest row per (algorithm_variant, topk, dataset) from run summaries
    agg: dict[tuple[str, str, str], dict] = {}
    for run in runs:
        summary = _read_csv(run / "summary.csv")
        for r in summary:
            key = (r.get("algorithm_variant", ""), r.get("topk", ""), "")
            row = {
                "run_id": run.name,
                "algorithm_variant": r.get("algorithm_variant", ""),
                "topk": r.get("topk", ""),
                "correct": r.get("correct", ""),
                "total": r.get("total", ""),
                "accuracy": r.get("accuracy", ""),
                "bytes_per_card": r.get("bytes_per_card", ""),
            }
            agg[key] = row
    if agg:
        return list(agg.values())

    # final fallback: infer current snapshot directly from cache jsonl files
    return _metrics_from_cache(results_root)


def _find_failures_for_variant(runs: list[Path], variant: str, n: int) -> list[dict]:
    # prefer most recent runs first
    out: list[dict] = []
    for run in reversed(runs):
        for row in _read_jsonl(run / "failures.jsonl"):
            if row.get("algorithm_variant") == variant:
                out.append(row)
        if len(out) >= n:
            break
    return out[:n]


def build_report(results_root: Path, reports_root: Path, worst_n: int = 8) -> tuple[Path, Path | None]:
    reports_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    md_path = reports_root / f"eval_report_{ts}.md"
    pdf_path = reports_root / f"eval_report_{ts}.pdf"
    assets_dir = reports_root / f"{md_path.stem}_assets"

    runs = _collect_runs(results_root)
    latest = _latest_metrics(results_root, runs)
    card_index = _load_card_index()

    lines: list[str] = []
    lines.append(f"# CCG Card ID Evaluation Report ({ts})")
    lines.append("")
    lines.append(f"- Results root: `{results_root}`")
    lines.append(f"- Run folders discovered: **{len(runs)}**")
    lines.append("")

    lines.append("## Latest comparison table")
    lines.append("")
    lines.append("| Algorithm | Bytes/Card | Top-1 | Top-1 % | Top-3 | Top-3 % | Top-10 | Top-10 % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    def _acc_pct(a: str) -> str:
        try:
            return f"{float(a)*100:.2f}%"
        except Exception:
            return "-"

    latest_by_variant: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    bytes_by_variant: dict[str, str] = {}
    for r in latest:
        v = r.get("algorithm_variant", "")
        try:
            k = int(float(r.get("topk", 0) or 0))
        except Exception:
            continue
        latest_by_variant[v][k] = r
        if v not in bytes_by_variant or not bytes_by_variant[v] or bytes_by_variant[v] == "":
            b = r.get("bytes_per_card", "")
            if b in (None, ""):
                b = _bytes_per_card_from_variant(v)
            bytes_by_variant[v] = str(b if b is not None else "-")

    def _count(v: str, k: int) -> str:
        row = latest_by_variant.get(v, {}).get(k)
        return str(row.get("correct", "-")) if row else "-"

    def _pct(v: str, k: int) -> str:
        row = latest_by_variant.get(v, {}).get(k)
        return _acc_pct(str(row.get("accuracy", ""))) if row else "-"

    for v in sorted(latest_by_variant.keys()):
        lines.append(
            f"| {v} | {bytes_by_variant.get(v, '-')} | {_count(v,1)} | {_pct(v,1)} | {_count(v,3)} | {_pct(v,3)} | {_count(v,10)} | {_pct(v,10)} |"
        )

    lines.append("")
    lines.append("\\newpage")
    lines.append("")
    lines.append("## Algorithm breakdowns + representative failures")
    lines.append("")

    variants = sorted({r.get("algorithm_variant", "") for r in latest if r.get("algorithm_variant")})
    by_variant = defaultdict(list)
    for r in latest:
        by_variant[r.get("algorithm_variant", "")].append(r)

    for i, v in enumerate(variants):
        if i > 0:
            lines.append("\\newpage")
            lines.append("")
        lines.append(f"### {v}")
        lines.append("")

        cache_rows = _load_variant_cache_rows(results_root, v)
        picks = _pick_failure_examples(cache_rows)

        def _row_image_key(row: dict | None) -> str:
            if not row:
                return ""
            return str(row.get("image_key") or row.get("image_path") or "")

        def _input_img(row: dict | None) -> str:
            if not row:
                return "-"
            ds = Path(row.get("dataset", cfg.data_dir / "datasets" / "solring"))
            image_key = _row_image_key(row)
            p = ds / image_key
            return _img_md(p, reports_root, assets_dir) if p.exists() else image_key

        def _retrieved_img(row: dict | None) -> str:
            if not row:
                return "-"
            top_ids = row.get("top_ids", [])
            pred = top_ids[0] if top_ids else ""
            p = _front_image_for_id(pred)
            return _img_md(p, reports_root, assets_dir) if p else (pred or "-")

        def _expected_img(row: dict | None) -> str:
            if not row:
                return "-"
            true_id = str(row.get("true_id", ""))
            p = _front_image_for_id(true_id)
            return _img_md(p, reports_root, assets_dir) if p else (true_id or "-")

        def _input_meta(row: dict | None) -> str:
            if not row:
                return "-"
            image_key = _row_image_key(row)
            return _parse_input_label(image_key, card_index)

        def _retrieved_meta(row: dict | None) -> str:
            if not row:
                return "-"
            top_ids = row.get("top_ids", [])
            pred = top_ids[0] if top_ids else ""
            return _label_for_card_id(pred, card_index) if pred else "-"

        def _expected_meta(row: dict | None) -> str:
            if not row:
                return "-"
            true_id = str(row.get("true_id", ""))
            return _label_for_card_id(true_id, card_index) if true_id else "-"

        def _retrieved_score(row: dict | None) -> str:
            if not row:
                return "-"
            top_scores = row.get("top_scores", [])
            if top_scores:
                return str(top_scores[0])
            return "-"

        def _expected_score(row: dict | None) -> str:
            if not row:
                return "-"
            top_scores = row.get("top_scores", [])
            true_rank = row.get("true_rank")
            exp_score = None
            if isinstance(true_rank, int) and 1 <= true_rank <= len(top_scores):
                exp_score = top_scores[true_rank - 1]
            if exp_score is None and true_rank is None:
                return "-"
            if exp_score is None:
                return f"(rank={true_rank})"
            if true_rank is None:
                return f"{exp_score}"
            return f"{exp_score} (rank={true_rank})"

        rows_by_k = {int(float(r.get("topk", 0) or 0)): r for r in by_variant.get(v, []) if str(r.get("topk", "")).strip()}

        def _acc_value(k: int) -> str:
            r = rows_by_k.get(k)
            if not r:
                return "-"
            correct = r.get("correct", "-")
            total = r.get("total", "-")
            pct = _acc_pct(str(r.get("accuracy", "")))
            return f"{correct} / {total} ({pct})"

        lines.append("")
        lines.append("| **Top-k Acc** | **Example Failure** | **Retrieved** | **Expected** |")
        lines.append("|---|---|---|---|")
        order = [("top1", 1), ("top3", 3), ("top10", 10)]
        for key, k in order:
            row_pick = picks.get(key)
            lines.append(f"| Top-{k}: {_acc_value(k)} | {_input_meta(row_pick)} | {_retrieved_meta(row_pick)} | {_expected_meta(row_pick)} |")
            lines.append(f"|  | {_input_img(row_pick)} | {_retrieved_img(row_pick)} | {_expected_img(row_pick)} |")
            lines.append(f"|  |  | score={_retrieved_score(row_pick)} | score={_expected_score(row_pick)} |")

        fails = _find_failures_for_variant(runs, v, worst_n)
        lines.append("")
        lines.append(f"Top {len(fails)} sampled failures:")
        lines.append("")
        lines.append("| image_path (relative) | true_id | predicted_id | score | true_rank |")
        lines.append("|---|---|---|---:|---:|")
        if fails:
            for f in fails:
                lines.append(
                    f"| {f.get('image_path','')} | {f.get('true_id','')} | {f.get('predicted_id','')} | {f.get('score','')} | {f.get('true_rank','-')} |"
                )
        else:
            lines.append("| (none yet) | - | - | - | - |")
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    built_pdf: Path | None = None
    if shutil.which("pandoc"):
        try:
            cmd = ["pandoc", str(md_path), "-o", str(pdf_path), "--resource-path", str(reports_root)]
            if shutil.which("tectonic"):
                cmd.extend(["--pdf-engine", "tectonic"])
            # Tight margins + slightly smaller base font to reduce table overlap.
            cmd.extend(["-V", "geometry:margin=0.15in", "-V", "fontsize=9pt"])
            subprocess.run(cmd, check=True)
            built_pdf = pdf_path
        except Exception:
            built_pdf = None

    return md_path, built_pdf


def main() -> None:
    p = argparse.ArgumentParser(description="Consolidate eval artifacts into report markdown/pdf")
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    p.add_argument("--worst-n", type=int, default=8)
    args = p.parse_args()

    md, pdf = build_report(args.results_root, args.reports_root, args.worst_n)
    print(f"Markdown report: {md}")
    if pdf:
        print(f"PDF report:      {pdf}")
    else:
        print("PDF report:      skipped (pandoc not available or conversion failed)")


if __name__ == "__main__":
    main()
