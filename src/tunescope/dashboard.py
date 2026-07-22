from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from tunescope.artifacts import ensure_dir, read_json


def _format(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def generate_dashboard(root: Path, report_json: str, output: str) -> Path:
    source = Path(report_json)
    if not source.is_absolute():
        source = root / source
    report = read_json(source)
    experiments = report.get("experiments", [])
    if not isinstance(experiments, list):
        experiments = []

    rows = []
    for item in experiments:
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        rows.append(
            [
                item.get("experiment_id"),
                item.get("source_id"),
                item.get("method"),
                item.get("sample_count"),
                metrics.get("jglue_accuracy"),
                metrics.get("xlsum_rouge_l"),
                metrics.get("elyza_judge_score"),
                metrics.get("tokens_per_second"),
                metrics.get("artifact_size_mb"),
            ]
        )

    headers = ["ID", "Source", "Method", "Count", "JGLUE Acc", "XL-Sum R-L", "ELYZA", "Tok/s", "Artifact MB"]
    table_rows = "\n".join(
        "<tr>" + "".join(f"<td>{html.escape(_format(cell))}</td>" for cell in row) + "</tr>" for row in rows
    )
    header_row = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    payload = html.escape(source.read_text(encoding="utf-8"))
    document = f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TuneScope Dashboard</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; color: #172026; background: #f7f8fa; }}
    h1 {{ margin: 0 0 16px; font-size: 28px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid #d9dee7; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #eef2f6; }}
    pre {{ white-space: pre-wrap; background: #101820; color: #f7f8fa; padding: 16px; overflow: auto; }}
  </style>
</head>
<body>
  <h1>TuneScope Dashboard</h1>
  <table>
    <thead><tr>{header_row}</tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
  <h2>Raw Report JSON</h2>
  <pre>{payload}</pre>
</body>
</html>
"""
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = root / output_path
    ensure_dir(output_path.parent)
    output_path.write_text(document, encoding="utf-8", newline="\n")
    return output_path

