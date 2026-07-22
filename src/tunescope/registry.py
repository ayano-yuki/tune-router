from __future__ import annotations

from pathlib import Path
from typing import Any

from tunescope.artifacts import ensure_dir, load_metrics, write_yaml
from tunescope.config import ConfigError, load_yaml


REGISTRY_PATH = Path("experiments") / "manifests" / "artifacts.yaml"


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _artifact_kind(model_dir: Path, method: str) -> str:
    if method in {"lora_sft", "qlora_sft", "dpo"}:
        return "adapter" if (model_dir / "adapter_config.json").exists() else "adapter_or_model"
    if method == "full_sft":
        return "full_model"
    return "model"


def load_artifact_registry(root: Path) -> dict[str, Any]:
    path = root / REGISTRY_PATH
    if not path.exists():
        return {"id": "artifact_registry", "artifacts": []}
    registry = load_yaml(path)
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, list):
        raise ConfigError(f"{path} artifacts must be a list.")
    return registry


def write_artifact_registry(root: Path, registry: dict[str, Any]) -> Path:
    path = root / REGISTRY_PATH
    ensure_dir(path.parent)
    write_yaml(path, registry)
    return path


def register_artifact(
    root: Path,
    experiment: dict[str, Any],
    output_dir: Path,
    status: str,
) -> dict[str, Any]:
    registry = load_artifact_registry(root)
    artifacts = [item for item in registry.get("artifacts", []) if item.get("experiment_id") != experiment["id"]]
    model_dir = output_dir / "model"
    train_metrics = load_metrics(output_dir / "train_metrics.json")
    artifact = {
        "experiment_id": experiment["id"],
        "name": experiment.get("name"),
        "method": experiment.get("method"),
        "status": status,
        "kind": _artifact_kind(model_dir, str(experiment.get("method"))),
        "path": str(model_dir),
        "exists": model_dir.exists(),
        "size_bytes": _dir_size_bytes(model_dir),
        "base_model": experiment.get("base_model"),
        "base_model_revision": experiment.get("base_model_revision"),
        "dataset": experiment.get("dataset"),
        "sample_count": experiment.get("sample_count"),
        "seed": experiment.get("seed"),
        "reuses_result_from": experiment.get("reuses_result_from"),
        "starts_from_experiment": experiment.get("starts_from_experiment"),
        "train_status": train_metrics.get("status"),
        "train_loss": train_metrics.get("train_loss"),
        "train_runtime": train_metrics.get("train_runtime"),
    }
    artifacts.append(artifact)
    registry["artifacts"] = sorted(artifacts, key=lambda item: str(item.get("experiment_id")))
    write_artifact_registry(root, registry)
    return artifact


def artifact_by_experiment(root: Path) -> dict[str, dict[str, Any]]:
    registry = load_artifact_registry(root)
    return {
        str(item["experiment_id"]): item
        for item in registry.get("artifacts", [])
        if isinstance(item, dict) and item.get("experiment_id")
    }

