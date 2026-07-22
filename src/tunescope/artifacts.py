from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tunescope.config import ConfigError, load_all, load_yaml
from tunescope.dataset_setup import artifact_paths


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object.")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConfigError("PyYAML is required. Run: uv sync --group dev") from exc

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ConfigError(f"{path}:{line_number} must contain a JSON object.")
            records.append(value)
    return records


def default_output_dir(root: Path, experiment: dict[str, Any]) -> Path:
    outputs = experiment.get("outputs")
    if isinstance(outputs, dict) and outputs.get("result_dir"):
        return root / str(outputs["result_dir"])
    return root / "experiments" / "results" / str(experiment["id"])


def normalize_cli_path(value: str, platform_name: str | None = None) -> str:
    platform_name = platform_name or os.name
    if platform_name != "nt":
        return value.replace("\\", "/")
    return value


def resolve_output_dir(root: Path, experiment: dict[str, Any], output_dir: str | None) -> Path:
    return Path(normalize_cli_path(output_dir)).resolve() if output_dir else default_output_dir(root, experiment)


def prepared_dataset_path(root: Path, experiment: dict[str, Any]) -> Path | None:
    dataset_id = experiment.get("dataset")
    if dataset_id is None:
        return None
    configs = load_all(root)
    dataset_config = configs["datasets"].get(str(dataset_id))
    if dataset_config is None:
        raise ConfigError(f"Unknown dataset {dataset_id!r}.")
    split = str(dataset_config.get("split", "train"))
    sample_count = experiment.get("sample_count", "all")
    seed = int(experiment.get("seed", 42))
    output_path, _ = artifact_paths(root, str(dataset_id), split, sample_count, seed)
    if not output_path.exists():
        raise ConfigError(
            f"Prepared dataset not found: {output_path}. "
            "Run `uv run tunescope setup-datasets --experiment-id "
            f"{experiment['id']}` first."
        )
    return output_path


def run_metadata(root: Path, experiment: dict[str, Any], command: str, output_dir: Path) -> dict[str, Any]:
    return {
        "command": command,
        "experiment": {
            "id": experiment["id"],
            "name": experiment.get("name"),
            "method": experiment.get("method"),
            "phase": experiment.get("phase"),
            "dataset": experiment.get("dataset"),
            "sample_count": experiment.get("sample_count"),
            "seed": experiment.get("seed"),
            "train_config": experiment.get("train_config"),
            "evaluation_config": experiment.get("evaluation_config"),
            "reuses_result_from": experiment.get("reuses_result_from"),
            "starts_from_experiment": experiment.get("starts_from_experiment"),
        },
        "paths": {
            "root": str(root),
            "output_dir": str(output_dir),
        },
    }


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path)


def load_report_matrix(root: Path, matrix_path: str | None) -> dict[str, Any]:
    if matrix_path:
        return load_yaml(root / matrix_path)
    return load_yaml(root / "experiments" / "manifests" / "initial_matrix.yaml")
