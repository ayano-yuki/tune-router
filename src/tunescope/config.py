from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when a TuneScope configuration file is invalid."""


@dataclass(frozen=True)
class ValidationMessage:
    level: str
    message: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ConfigError("PyYAML is required. Install with: pip install -e \".[dev]\"") from exc

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level.")
    return data


def yaml_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob("*.yaml"))


def load_config_dir(root: Path, relative_dir: str) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for path in yaml_files(root / relative_dir):
        data = load_yaml(path)
        config_id = data.get("id")
        if not isinstance(config_id, str) or not config_id:
            raise ConfigError(f"{path} must define a non-empty string id.")
        if config_id in configs:
            raise ConfigError(f"Duplicate config id {config_id!r} in {path}.")
        data["_path"] = str(path.relative_to(root))
        configs[config_id] = data
    return configs


def load_all(root: Path | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    root = root or repo_root()
    return {
        "datasets": load_config_dir(root, "configs/datasets"),
        "train": load_config_dir(root, "configs/train"),
        "evaluation": load_config_dir(root, "configs/evaluation"),
        "experiments": load_config_dir(root, "configs/experiments"),
    }


def load_matrix(root: Path | None = None, matrix_path: str = "experiments/manifests/initial_matrix.yaml") -> dict[str, Any]:
    root = root or repo_root()
    return load_yaml(root / matrix_path)


def validate_workspace(root: Path | None = None) -> list[ValidationMessage]:
    root = root or repo_root()
    messages: list[ValidationMessage] = []

    required_dirs = [
        "configs/datasets",
        "configs/train",
        "configs/evaluation",
        "configs/experiments",
        "datasets/manifests",
        "experiments/manifests",
        "experiments/results",
        "reports",
        "src/tunescope",
    ]
    for relative_dir in required_dirs:
        if not (root / relative_dir).exists():
            messages.append(ValidationMessage("error", f"Missing directory: {relative_dir}"))

    configs = load_all(root)
    matrix = load_matrix(root)
    experiment_ids = set(configs["experiments"])
    dataset_ids = set(configs["datasets"])
    train_ids = set(configs["train"])
    evaluation_ids = set(configs["evaluation"])

    matrix_experiments = matrix.get("experiments", [])
    priority = matrix.get("priority", [])
    if not isinstance(matrix_experiments, list):
        messages.append(ValidationMessage("error", "initial_matrix.experiments must be a list."))
        matrix_experiments = []
    if not isinstance(priority, list):
        messages.append(ValidationMessage("error", "initial_matrix.priority must be a list."))
        priority = []

    for experiment_id in matrix_experiments:
        if experiment_id not in experiment_ids:
            messages.append(ValidationMessage("error", f"Matrix references unknown experiment: {experiment_id}"))

    for experiment_id in priority:
        if experiment_id not in matrix_experiments:
            messages.append(ValidationMessage("error", f"Priority experiment is not in matrix: {experiment_id}"))

    for experiment_id, experiment in configs["experiments"].items():
        for field in ["id", "name", "phase", "method", "base_model", "evaluation_config", "outputs"]:
            if field not in experiment:
                messages.append(ValidationMessage("error", f"{experiment_id} missing required field: {field}"))

        dataset_id = experiment.get("dataset")
        if dataset_id is not None and dataset_id not in dataset_ids:
            messages.append(ValidationMessage("error", f"{experiment_id} references unknown dataset: {dataset_id}"))

        train_id = experiment.get("train_config")
        if train_id is not None and train_id not in train_ids:
            messages.append(ValidationMessage("error", f"{experiment_id} references unknown train config: {train_id}"))

        evaluation_id = experiment.get("evaluation_config")
        if evaluation_id is not None and evaluation_id not in evaluation_ids:
            messages.append(
                ValidationMessage("error", f"{experiment_id} references unknown evaluation config: {evaluation_id}")
            )

        reused = experiment.get("reuses_result_from")
        if reused is not None and reused not in experiment_ids:
            messages.append(ValidationMessage("error", f"{experiment_id} reuses unknown experiment: {reused}"))

    unresolved_revisions = [
        dataset_id
        for dataset_id, dataset in configs["datasets"].items()
        if dataset.get("revision") == "TODO_PIN_HF_COMMIT"
    ]
    if unresolved_revisions:
        joined = ", ".join(sorted(unresolved_revisions))
        messages.append(ValidationMessage("warning", f"Dataset revisions still need pinning: {joined}"))

    unresolved_models = [
        experiment_id
        for experiment_id, experiment in configs["experiments"].items()
        if str(experiment.get("base_model", "")).startswith("TODO_")
    ]
    if unresolved_models:
        joined = ", ".join(sorted(unresolved_models))
        messages.append(ValidationMessage("warning", f"Base model placeholders still need selection: {joined}"))

    if not messages:
        messages.append(ValidationMessage("ok", "Workspace configuration is valid."))
    return messages


def get_experiment(experiment_id: str, root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    experiments = load_all(root)["experiments"]
    try:
        return experiments[experiment_id]
    except KeyError as exc:
        known = ", ".join(sorted(experiments))
        raise ConfigError(f"Unknown experiment {experiment_id!r}. Known experiments: {known}") from exc

