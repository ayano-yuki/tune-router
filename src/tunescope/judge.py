from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from tunescope.artifacts import ensure_dir, read_jsonl, resolve_output_dir
from tunescope.config import ConfigError, get_experiment
from tunescope.evaluation import apply_elyza_judge_scores


def _heuristic_score(row: dict[str, Any]) -> dict[str, Any]:
    response = str(row.get("response") or "").strip()
    score = 1.0
    comments = []
    if len(response) >= 40:
        score += 1.0
    if len(response) >= 120:
        score += 1.0
    if "できません" not in response and "申し訳" not in response:
        score += 1.0
    if any(marker in response for marker in ["。", "、", "\n"]):
        score += 1.0
    score = max(1.0, min(5.0, score))
    comments.append("heuristic length/refusal/style score")
    return {"id": str(row.get("id", "")), "score": score, "comment": "; ".join(comments)}


def _parse_judge_stdout(stdout: str, row: dict[str, Any]) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise ConfigError("Judge command returned empty output.")
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            score = value.get("score") or value.get("judge_score")
            comment = value.get("comment") or value.get("reason") or ""
            return {"id": str(row.get("id", "")), "score": float(score), "comment": str(comment)}
    except json.JSONDecodeError:
        pass
    return {"id": str(row.get("id", "")), "score": float(text.split()[0]), "comment": text}


def _external_score(command: str, row: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        payload_path = Path(tmp) / "prediction.json"
        payload_path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        rendered = command.format(
            input=str(payload_path),
            prompt=str(row.get("prompt", "")),
            response=str(row.get("response", "")),
            id=str(row.get("id", "")),
        )
        args = shlex.split(rendered, posix=False)
        result = subprocess.run(args, check=False, capture_output=True, text=True, encoding="utf-8")
        if result.returncode != 0:
            raise ConfigError(f"Judge command failed with {result.returncode}: {result.stderr.strip()}")
        return _parse_judge_stdout(result.stdout, row)


def judge_elyza(
    root: Path,
    experiment_id: str,
    output_dir_arg: str | None,
    scores_output: str | None,
    provider: str,
    judge_command: str | None,
) -> Path:
    experiment = get_experiment(experiment_id, root)
    output_dir = ensure_dir(resolve_output_dir(root, experiment, output_dir_arg))
    predictions_path = output_dir / "predictions" / "elyza_tasks_100.jsonl"
    if not predictions_path.exists():
        raise ConfigError(f"ELYZA predictions not found: {predictions_path}")

    rows = read_jsonl(predictions_path)
    score_rows = []
    for row in rows:
        if provider == "heuristic":
            score_rows.append(_heuristic_score(row))
        elif provider == "command":
            if not judge_command:
                raise ConfigError("--judge-command is required when --provider command.")
            score_rows.append(_external_score(judge_command, row))
        else:
            raise ConfigError(f"Unsupported judge provider: {provider}")

    scores_path = Path(scores_output) if scores_output else output_dir / "metrics" / "elyza_judge_scores.jsonl"
    if not scores_path.is_absolute():
        scores_path = root / scores_path
    ensure_dir(scores_path.parent)
    with scores_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in score_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    apply_elyza_judge_scores(root, experiment_id, str(scores_path), str(output_dir))
    return scores_path

