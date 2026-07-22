import json
from pathlib import Path

from tunescope.evaluation import apply_elyza_judge_scores


ROOT = Path(__file__).resolve().parents[1]


def test_apply_elyza_judge_scores_updates_metrics(tmp_path) -> None:
    output_dir = tmp_path / "Q2"
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True)
    predictions = [
        {"id": "1", "prompt": "a", "response": "A"},
        {"id": "2", "prompt": "b", "response": "B"},
    ]
    with (predictions_dir / "elyza_tasks_100.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    scores_path = tmp_path / "scores.csv"
    scores_path.write_text("id,score,comment\n1,4,good\n2,2,short\n", encoding="utf-8")

    apply_elyza_judge_scores(ROOT, "Q2", str(scores_path), str(output_dir))

    metrics = json.loads((output_dir / "metrics" / "elyza_tasks_100.json").read_text(encoding="utf-8"))
    aggregate = json.loads((output_dir / "eval_metrics.json").read_text(encoding="utf-8"))
    assert metrics["judge_score"] == 3.0
    assert aggregate["elyza_judge_score"] == 3.0

