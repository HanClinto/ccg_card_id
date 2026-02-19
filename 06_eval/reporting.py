"""Utilities for writing evaluation artifacts and Markdown reports."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def timestamp_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def timestamp_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_dir(output_root: Path, run_id: str | None = None) -> Path:
    run = run_id or timestamp_run_id()
    out_dir = output_root / run
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["algorithm_variant", "topk", "correct", "total", "accuracy"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def write_failures_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default) + "\n")


def write_algorithm_markdown(
    path: Path,
    algorithm_variant: str,
    summary_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    top_n: int,
) -> None:
    lines: list[str] = []
    lines.append(f"# {algorithm_variant} evaluation")
    lines.append("")
    lines.append("## Accuracy")
    lines.append("")
    lines.append("| top-k | correct | total | accuracy |")
    lines.append("|---:|---:|---:|---:|")
    for r in sorted(summary_rows, key=lambda x: x["topk"]):
        lines.append(
            f"| {r['topk']} | {r['correct']} | {r['total']} | {r['accuracy'] * 100:.2f}% |"
        )

    lines.append("")
    lines.append(f"## Worst-case failures (top {top_n})")
    lines.append("")
    lines.append("| image_path | true_id | predicted_id | score | true_rank |")
    lines.append("|---|---|---|---:|---:|")

    if failures:
        for row in failures[:top_n]:
            score = row.get("score")
            score_str = f"{score:.6f}" if isinstance(score, (float, int)) else str(score)
            true_rank = row.get("true_rank")
            true_rank_str = str(true_rank) if true_rank is not None else "-"
            lines.append(
                f"| {row.get('image_path','')} | {row.get('true_id','')} | {row.get('predicted_id','')} | {score_str} | {true_rank_str} |"
            )
    else:
        lines.append("| (none) | - | - | - | - |")

    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_overview_markdown(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("# Evaluation overview")
    lines.append("")
    lines.append("| algorithm_variant | top-k | correct | total | accuracy |")
    lines.append("|---|---:|---:|---:|---:|")

    for row in sorted(summary_rows, key=lambda r: (r["algorithm_variant"], r["topk"])):
        lines.append(
            f"| {row['algorithm_variant']} | {row['topk']} | {row['correct']} | {row['total']} | {row['accuracy'] * 100:.2f}% |"
        )

    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_central_result_csvs(
    output_root: Path,
    summary_rows: list[dict[str, Any]],
    *,
    benchmark: str,
    dataset: str,
    run_id: str,
    run_at: str | None = None,
) -> tuple[Path, Path]:
    """Append run records to history CSV and refresh latest-results CSV.

    history: append-only across runs
    latest:  one row per (benchmark, algorithm_variant, topk, dataset) with newest run
    """
    output_root.mkdir(parents=True, exist_ok=True)
    run_at = run_at or timestamp_iso_utc()

    history_path = output_root / "history_results.csv"
    latest_path = output_root / "latest_results.csv"

    fields = [
        "run_at",
        "run_id",
        "benchmark",
        "dataset",
        "algorithm_variant",
        "topk",
        "correct",
        "total",
        "accuracy",
    ]

    history_rows: list[dict[str, Any]] = []
    if history_path.exists():
        with history_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            history_rows = list(reader)

    new_rows: list[dict[str, Any]] = []
    for r in summary_rows:
        new_rows.append({
            "run_at": run_at,
            "run_id": run_id,
            "benchmark": benchmark,
            "dataset": dataset,
            "algorithm_variant": r.get("algorithm_variant"),
            "topk": r.get("topk"),
            "correct": r.get("correct"),
            "total": r.get("total"),
            "accuracy": r.get("accuracy"),
        })

    history_rows.extend(new_rows)

    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history_rows:
            writer.writerow({k: row.get(k) for k in fields})

    latest_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in history_rows:
        key = (
            str(row.get("benchmark", "")),
            str(row.get("algorithm_variant", "")),
            str(row.get("topk", "")),
            str(row.get("dataset", "")),
        )
        cur = latest_by_key.get(key)
        if cur is None or str(row.get("run_at", "")) >= str(cur.get("run_at", "")):
            latest_by_key[key] = row

    latest_rows = sorted(
        latest_by_key.values(),
        key=lambda r: (str(r.get("benchmark", "")), str(r.get("algorithm_variant", "")), str(r.get("topk", ""))),
    )
    with latest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in latest_rows:
            writer.writerow({k: row.get(k) for k in fields})

    return history_path, latest_path
