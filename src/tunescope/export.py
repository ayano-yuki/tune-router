from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tunescope.artifacts import ensure_dir, read_json


EXPORT_COLUMNS = [
    "experiment_id",
    "source_id",
    "method",
    "sample_count",
    "train_status",
    "eval_status",
    "train_loss",
    "train_runtime",
    "jglue_accuracy",
    "jglue_macro_f1",
    "jglue_exact_match",
    "jglue_qa_f1",
    "jglue_sts_pearson",
    "xlsum_rouge_l",
    "elyza_judge_score",
    "tokens_per_second",
    "mean_output_tokens",
    "artifact_size_mb",
]


def _rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in report.get("experiments", []):
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        row = {
            "experiment_id": item.get("experiment_id"),
            "source_id": item.get("source_id"),
            "method": item.get("method"),
            "sample_count": item.get("sample_count"),
        }
        for column in EXPORT_COLUMNS:
            if column not in row:
                row[column] = metrics.get(column)
        rows.append(row)
    return rows


def export_metrics(root: Path, report_json: str, output: str, fmt: str) -> Path:
    source = Path(report_json)
    if not source.is_absolute():
        source = root / source
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = root / output_path
    ensure_dir(output_path.parent)
    rows = _rows(read_json(source))

    if fmt == "csv":
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    elif fmt == "jsonl":
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
    else:
        raise ValueError(f"Unsupported export format: {fmt}")
    return output_path

