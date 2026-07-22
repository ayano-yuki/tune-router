from __future__ import annotations

from pathlib import Path
from typing import Any

from tunescope.artifacts import default_output_dir, ensure_dir, load_metrics, load_report_matrix, write_json
from tunescope.config import ConfigError, load_all
from tunescope.registry import artifact_by_experiment


PRIMARY_METRICS = [
    ("jglue_accuracy", "jglue", "accuracy"),
    ("jglue_macro_f1", "jglue", "macro_f1"),
    ("jglue_exact_match", "jglue", "exact_match"),
    ("jglue_qa_f1", "jglue", "qa_f1"),
    ("jglue_sts_pearson", "jglue", "sts_pearson"),
    ("xlsum_rouge_l", "xlsum_ja", "rougeL"),
    ("elyza_judge_score", "elyza_tasks_100", "judge_score"),
]


def _numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _format(value: Any) -> str:
    numeric = _numeric(value)
    if numeric is None:
        return "-" if value is None else str(value)
    return f"{numeric:.4g}"


def _task_value(metrics: dict[str, Any], task_id: str, key: str) -> Any:
    tasks = metrics.get("tasks")
    if not isinstance(tasks, dict):
        return None
    task_metrics = tasks.get(task_id)
    if not isinstance(task_metrics, dict):
        return None
    return task_metrics.get(key)


def _flatten_metrics(train_metrics: dict[str, Any], eval_metrics: dict[str, Any], artifact: dict[str, Any] | None) -> dict[str, Any]:
    flattened: dict[str, Any] = {
        "train_status": train_metrics.get("status"),
        "train_loss": train_metrics.get("train_loss"),
        "train_runtime": train_metrics.get("train_runtime"),
        "eval_status": eval_metrics.get("status"),
        "mean_output_tokens": eval_metrics.get("mean_output_tokens"),
        "tokens_per_second": eval_metrics.get("tokens_per_second"),
        "refusal_rate": eval_metrics.get("refusal_rate"),
        "json_valid_rate": eval_metrics.get("json_valid_rate"),
    }
    for name, task_id, key in PRIMARY_METRICS:
        flattened[name] = _task_value(eval_metrics, task_id, key)
    if artifact:
        flattened["artifact_kind"] = artifact.get("kind")
        flattened["artifact_exists"] = artifact.get("exists")
        flattened["artifact_size_mb"] = _numeric(artifact.get("size_bytes")) / (1024**2) if artifact.get("size_bytes") else 0
        flattened["artifact_path"] = artifact.get("path")
    return flattened


def _resolve_result_dir(root: Path, results_dir: str | None, source_id: str, source_experiment: dict[str, Any]) -> Path:
    if results_dir:
        result_root = Path(results_dir)
        if not result_root.is_absolute():
            result_root = root / result_root
        return result_root / source_id
    return default_output_dir(root, source_experiment)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _delta(value: Any, baseline: Any) -> float | None:
    numeric = _numeric(value)
    base = _numeric(baseline)
    if numeric is None or base is None:
        return None
    return numeric - base


def _efficiency(delta: float | None, runtime: Any) -> float | None:
    numeric_runtime = _numeric(runtime)
    if delta is None or numeric_runtime is None or numeric_runtime <= 0:
        return None
    return delta / (numeric_runtime / 3600)


def generate_report(
    root: Path,
    output: str,
    matrix_path: str | None = None,
    experiment_ids: list[str] | None = None,
    results_dir: str | None = None,
) -> Path:
    configs = load_all(root)
    matrix = load_report_matrix(root, matrix_path)
    ids = experiment_ids or list(matrix.get("experiments", []))
    if not ids:
        raise ConfigError("No experiments selected for report.")

    artifacts = artifact_by_experiment(root)
    entries: list[dict[str, Any]] = []
    for experiment_id in ids:
        if experiment_id not in configs["experiments"]:
            raise ConfigError(f"Unknown experiment {experiment_id!r}.")
        experiment = configs["experiments"][experiment_id]
        source_id = str(experiment.get("reuses_result_from") or experiment_id)
        source_experiment = configs["experiments"].get(source_id, experiment)
        result_dir = _resolve_result_dir(root, results_dir, source_id, source_experiment)
        train_metrics = load_metrics(result_dir / "train_metrics.json")
        eval_metrics = load_metrics(result_dir / "eval_metrics.json")
        artifact = artifacts.get(source_id)
        flattened = _flatten_metrics(train_metrics, eval_metrics, artifact)
        entries.append(
            {
                "experiment_id": experiment_id,
                "source_id": source_id,
                "method": experiment.get("method"),
                "dataset": experiment.get("dataset") or "none",
                "sample_count": experiment.get("sample_count"),
                "result_dir": str(result_dir),
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
                "artifact": artifact,
                "metrics": flattened,
            }
        )

    baseline = next((entry for entry in entries if entry["experiment_id"] == "B0"), None)
    if baseline is None and entries:
        baseline = entries[0]
    baseline_metrics = baseline["metrics"] if baseline else {}

    summary_rows = []
    delta_rows = []
    cost_rows = []
    for entry in entries:
        metrics = entry["metrics"]
        summary_rows.append(
            [
                str(entry["experiment_id"]),
                str(entry["source_id"]),
                str(entry["method"]),
                str(entry["sample_count"]),
                _format(metrics.get("train_status")),
                _format(metrics.get("eval_status")),
                _format(metrics.get("jglue_accuracy")),
                _format(metrics.get("jglue_macro_f1")),
                _format(metrics.get("jglue_exact_match")),
                _format(metrics.get("jglue_qa_f1")),
                _format(metrics.get("jglue_sts_pearson")),
                _format(metrics.get("xlsum_rouge_l")),
                _format(metrics.get("elyza_judge_score")),
                _format(metrics.get("tokens_per_second")),
            ]
        )

        acc_delta = _delta(metrics.get("jglue_accuracy"), baseline_metrics.get("jglue_accuracy"))
        rouge_delta = _delta(metrics.get("xlsum_rouge_l"), baseline_metrics.get("xlsum_rouge_l"))
        judge_delta = _delta(metrics.get("elyza_judge_score"), baseline_metrics.get("elyza_judge_score"))
        delta_rows.append(
            [
                str(entry["experiment_id"]),
                str(entry["source_id"]),
                _format(acc_delta),
                _format(rouge_delta),
                _format(judge_delta),
                _format(_delta(metrics.get("refusal_rate"), baseline_metrics.get("refusal_rate"))),
                _format(_delta(metrics.get("tokens_per_second"), baseline_metrics.get("tokens_per_second"))),
            ]
        )

        cost_rows.append(
            [
                str(entry["experiment_id"]),
                str(entry["source_id"]),
                _format(metrics.get("train_runtime")),
                _format(metrics.get("artifact_size_mb")),
                _format(metrics.get("mean_output_tokens")),
                _format(_efficiency(acc_delta, metrics.get("train_runtime"))),
                _format(_efficiency(rouge_delta, metrics.get("train_runtime"))),
                _format(_efficiency(judge_delta, metrics.get("train_runtime"))),
                str(metrics.get("artifact_kind") or "-"),
            ]
        )

    output_path = root / output
    ensure_dir(output_path.parent)
    content = "\n".join(
        [
            "# TuneScope Experiment Report",
            "",
            f"Baseline: `{baseline['experiment_id'] if baseline else '-'}`",
            "",
            "## Summary Metrics",
            "",
            _markdown_table(
                [
                    "ID",
                    "Source",
                    "Method",
                    "Count",
                    "Train",
                    "Eval",
                    "JGLUE Acc",
                    "JGLUE F1",
                    "JGLUE EM",
                    "JGLUE QA F1",
                    "JGLUE STS r",
                    "XL-Sum R-L",
                    "ELYZA Judge",
                    "Tok/s",
                ],
                summary_rows,
            ),
            "",
            "## Baseline Deltas",
            "",
            _markdown_table(
                ["ID", "Source", "JGLUE Acc Δ", "XL-Sum R-L Δ", "ELYZA Judge Δ", "Refusal Δ", "Tok/s Δ"],
                delta_rows,
            ),
            "",
            "## Cost And Artifacts",
            "",
            _markdown_table(
                [
                    "ID",
                    "Source",
                    "Train sec",
                    "Artifact MB",
                    "Mean Tokens",
                    "Acc Δ/hour",
                    "R-L Δ/hour",
                    "Judge Δ/hour",
                    "Artifact",
                ],
                cost_rows,
            ),
            "",
            "Generated from `experiments/results` and `experiments/manifests/artifacts.yaml`.",
            "",
        ]
    )
    output_path.write_text(content, encoding="utf-8", newline="\n")
    write_json(
        output_path.with_suffix(".json"),
        {
            "baseline": baseline["experiment_id"] if baseline else None,
            "experiments": entries,
        },
    )
    return output_path

