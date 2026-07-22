import json
from pathlib import Path

from tunescope.artifacts import normalize_cli_path
from tunescope.dashboard import generate_dashboard
from tunescope.export import export_metrics
from tunescope.judge import judge_elyza


ROOT = Path(__file__).resolve().parents[1]


def test_normalize_cli_path_accepts_windows_separators_on_posix() -> None:
    assert normalize_cli_path(r"experiments\results\Q1", platform_name="posix") == "experiments/results/Q1"
    assert normalize_cli_path(r"experiments\results\Q1", platform_name="nt") == r"experiments\results\Q1"


def write_report_json(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "baseline": "B0",
                "experiments": [
                    {
                        "experiment_id": "B0",
                        "source_id": "B0",
                        "method": "base",
                        "sample_count": 0,
                        "metrics": {"jglue_accuracy": 0.2, "tokens_per_second": 10.0},
                    },
                    {
                        "experiment_id": "Q2",
                        "source_id": "Q2",
                        "method": "qlora_sft",
                        "sample_count": 500,
                        "metrics": {"jglue_accuracy": 0.3, "tokens_per_second": 8.0},
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_dashboard_and_export(tmp_path) -> None:
    report = tmp_path / "report.json"
    write_report_json(report)

    dashboard = generate_dashboard(ROOT, str(report), str(tmp_path / "dashboard.html"))
    csv_path = export_metrics(ROOT, str(report), str(tmp_path / "metrics.csv"), "csv")
    jsonl_path = export_metrics(ROOT, str(report), str(tmp_path / "metrics.jsonl"), "jsonl")

    assert "TuneScope Dashboard" in dashboard.read_text(encoding="utf-8")
    assert "experiment_id" in csv_path.read_text(encoding="utf-8")
    assert len(jsonl_path.read_text(encoding="utf-8").splitlines()) == 2


def test_judge_elyza_heuristic(tmp_path) -> None:
    output_dir = tmp_path / "Q2"
    predictions = output_dir / "predictions"
    predictions.mkdir(parents=True)
    (predictions / "elyza_tasks_100.jsonl").write_text(
        json.dumps(
            {
                "id": "1",
                "prompt": "説明してください",
                "response": "これは十分な長さの日本語回答です。具体的に説明し、自然な文で回答しています。",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    scores = judge_elyza(ROOT, "Q2", str(output_dir), None, "heuristic", None)

    assert scores.exists()
    aggregate = json.loads((output_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    assert aggregate["elyza_judge_score"] >= 1.0
