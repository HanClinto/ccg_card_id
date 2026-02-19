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
import re
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ccg_card_id.config import cfg


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


def _img_md(path: Path, reports_root: Path, width_in: float = 1.4) -> str:
    rel = path
    try:
        rel = path.relative_to(reports_root)
    except Exception:
        try:
            rel = Path("..") / path.relative_to(reports_root.parent)
        except Exception:
            rel = path
    # Pandoc-friendly markdown image syntax renders in both Markdown preview and PDF.
    return f'![]({rel.as_posix()}){{width={width_in}in}}'


def _front_image_for_id(card_id: str) -> Path | None:
    m = UUID_RE.search(card_id or "")
    if not m:
        return None
    cid = m.group(0).lower()
    p = cfg.data_dir / "images" / "png" / "front" / cid[0] / cid[1] / f"{cid}.png"
    return p if p.exists() else None


def _load_card_index() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    src = cfg.scryfall_default_cards
    if not src.exists():
        return out
    try:
        cards = json.loads(src.read_text(encoding="utf-8"))
    except Exception:
        return out
    for c in cards:
        cid = str(c.get("id", "")).lower()
        if not cid:
            continue
        out[cid] = {
            "name": str(c.get("name", "")),
            "set": str(c.get("set", "")).upper(),
            "set_name": str(c.get("set_name", "")),
        }
    return out


def _parse_input_label(image_key: str, card_index: dict[str, dict[str, str]]) -> str:
    base = Path(image_key).name
    m = UUID_RE.search(base)
    cid = m.group(0).lower() if m else ""
    meta = card_index.get(cid, {})
    name = meta.get("name") or (base.split("_")[1] if "_" in base else "Card")
    parts = base.split("_")
    set_code = (parts[2].upper() if len(parts) >= 3 else (meta.get("set") or "?"))
    sample = ""
    if len(parts) >= 4:
        sample = "_".join(parts[3:])
        sample = re.sub(r"\.[^.]+$", "", sample)
    if sample:
        return f"{name} [{set_code}] ({sample})"
    return f"{name} [{set_code}]"


def _label_for_card_id(card_id: str, card_index: dict[str, dict[str, str]]) -> str:
    cid = (card_id or "").lower()
    meta = card_index.get(cid, {})
    name = meta.get("name") or "Unknown"
    set_code = meta.get("set") or "?"
    set_name = meta.get("set_name") or "Unknown set"
    return f"{name} [{set_code}] ({set_name})"


def _load_variant_cache_rows(results_root: Path, variant: str) -> list[dict]:
    rows: list[dict] = []
    for jf in (results_root / "cache" / "hash_retrieval").rglob("*.jsonl"):
        recs = _read_jsonl(jf)
        for r in recs:
            if r.get("algorithm_variant") == variant:
                rows.append(r)
    return rows


def _pick_failure_examples(cache_rows: list[dict]) -> dict[str, dict | None]:
    # Prefer unique rows across categories when possible
    buckets = {"top1": None, "top3": None, "top5": None}

    def rank(r: dict) -> int:
        tr = r.get("true_rank")
        return int(tr) if isinstance(tr, int) or (isinstance(tr, str) and tr.isdigit()) else 9999

    sorted_rows = sorted(cache_rows, key=rank)

    for r in sorted_rows:
        tr = rank(r)
        if buckets["top1"] is None and 2 <= tr <= 3:
            buckets["top1"] = r
        if buckets["top3"] is None and 4 <= tr <= 5:
            buckets["top3"] = r
        if buckets["top5"] is None and 6 <= tr <= 10:
            buckets["top5"] = r

    # relaxed fallback
    for r in sorted_rows:
        tr = rank(r)
        if buckets["top1"] is None and tr > 1:
            buckets["top1"] = r
        if buckets["top3"] is None and tr > 3:
            buckets["top3"] = r
        if buckets["top5"] is None and tr > 5:
            buckets["top5"] = r

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
    try:
        size = int(variant.split("_")[-1])
        return (size * size) // 8
    except Exception:
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
    lines.append("| algorithm_variant | top-k | correct | total | accuracy | bytes/card |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    def _acc_pct(a: str) -> str:
        try:
            return f"{float(a)*100:.2f}%"
        except Exception:
            return "-"

    for r in sorted(latest, key=lambda x: (x.get("algorithm_variant", ""), int(float(x.get("topk", 0) or 0)))):
        lines.append(
            f"| {r.get('algorithm_variant','')} | {r.get('topk','')} | {r.get('correct','')} | {r.get('total','')} | {_acc_pct(r.get('accuracy',''))} | {r.get('bytes_per_card','-')} |"
        )

    lines.append("")
    lines.append("## Algorithm breakdowns + representative failures")
    lines.append("")

    variants = sorted({r.get("algorithm_variant", "") for r in latest if r.get("algorithm_variant")})
    by_variant = defaultdict(list)
    for r in latest:
        by_variant[r.get("algorithm_variant", "")].append(r)

    for v in variants:
        lines.append(f"### {v}")
        lines.append("")
        lines.append("Accuracy snapshot:")
        lines.append("")
        lines.append("| top-k | correct | total | accuracy | bytes/card |")
        lines.append("|---:|---:|---:|---:|---:|")
        for r in sorted(by_variant[v], key=lambda x: int(float(x.get("topk", 0) or 0))):
            lines.append(f"| {r.get('topk','')} | {r.get('correct','')} | {r.get('total','')} | {_acc_pct(r.get('accuracy',''))} | {r.get('bytes_per_card','-')} |")

        cache_rows = _load_variant_cache_rows(results_root, v)
        picks = _pick_failure_examples(cache_rows)

        def _input_img(row: dict | None) -> str:
            if not row:
                return "-"
            ds = Path(row.get("dataset", cfg.data_dir / "datasets" / "solring"))
            image_key = str(row.get("image_key", ""))
            p = ds / image_key
            caption = f"Input: {_parse_input_label(image_key, card_index)}"
            img = _img_md(p, reports_root) if p.exists() else image_key
            return f"{img}<br/><sub>{caption}</sub>"

        def _retrieved_img(row: dict | None) -> str:
            if not row:
                return "-"
            top_ids = row.get("top_ids", [])
            pred = top_ids[0] if top_ids else ""
            p = _front_image_for_id(pred)
            caption = f"Retrieved: {_label_for_card_id(pred, card_index)}" if pred else "Retrieved: -"
            img = _img_md(p, reports_root) if p else (pred or "-")
            return f"{img}<br/><sub>{caption}</sub>"

        def _expected_img(row: dict | None) -> str:
            if not row:
                return "-"
            true_id = str(row.get("true_id", ""))
            p = _front_image_for_id(true_id)
            caption = f"Expected: {_label_for_card_id(true_id, card_index)}" if true_id else "Expected: -"
            img = _img_md(p, reports_root) if p else (true_id or "-")
            return f"{img}<br/><sub>{caption}</sub>"

        lines.append("")
        lines.append("Failure gallery (3×3)")
        lines.append("")
        lines.append("|  | top-1 miss | top-3 miss | top-5 miss |")
        lines.append("|---|---|---|---|")
        lines.append(f"| Input | {_input_img(picks['top1'])} | {_input_img(picks['top3'])} | {_input_img(picks['top5'])} |")
        lines.append(f"| Retrieved | {_retrieved_img(picks['top1'])} | {_retrieved_img(picks['top3'])} | {_retrieved_img(picks['top5'])} |")
        lines.append(f"| Expected | {_expected_img(picks['top1'])} | {_expected_img(picks['top3'])} | {_expected_img(picks['top5'])} |")

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

    lines.append("## Recommendations")
    lines.append("")
    lines.append("- Use top-1 for strict identification confidence; top-3/top-10 for operator-assist workflows.")
    lines.append("- Keep brute-force as ground-truth benchmark path for deterministic comparisons.")
    lines.append("- Continue collecting failure examples for lighting/angle classes to guide pre-processing improvements.")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    built_pdf: Path | None = None
    if shutil.which("pandoc"):
        try:
            cmd = ["pandoc", str(md_path), "-o", str(pdf_path)]
            if shutil.which("tectonic"):
                cmd.extend(["--pdf-engine", "tectonic"])
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
