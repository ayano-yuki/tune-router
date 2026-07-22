from __future__ import annotations

from pathlib import Path
from typing import Any

from tunescope.artifacts import normalize_cli_path
from tunescope.config import ConfigError, load_all, load_yaml
from tunescope.dataset_setup import TODO_REVISION


def _dump_yaml(path: Path, data: dict[str, Any]) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConfigError("PyYAML is required. Run: uv sync --group dev") from exc

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def _config_path(root: Path, config: dict[str, Any]) -> Path:
    path = config.get("_path")
    if not isinstance(path, str):
        raise ConfigError(f"Config {config.get('id')!r} does not include _path metadata.")
    return root / path


def _hf_dataset_sha(repo_id: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo_id)
    sha = getattr(info, "sha", None)
    if not sha:
        raise ConfigError(f"Could not resolve dataset sha for {repo_id!r}.")
    return str(sha)


def _hf_model_sha(repo_id: str) -> str:
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id)
    sha = getattr(info, "sha", None)
    if not sha:
        raise ConfigError(f"Could not resolve model sha for {repo_id!r}.")
    return str(sha)


def pin_dataset_revisions(root: Path, dataset_ids: list[str] | None = None, force: bool = False) -> list[dict[str, str]]:
    configs = load_all(root)["datasets"]
    selected = dataset_ids or sorted(configs)
    changes: list[dict[str, str]] = []
    for dataset_id in selected:
        if dataset_id not in configs:
            known = ", ".join(sorted(configs))
            raise ConfigError(f"Unknown dataset {dataset_id!r}. Known datasets: {known}")
        config = configs[dataset_id]
        current = str(config.get("revision", ""))
        if current and current != TODO_REVISION and not force:
            changes.append({"id": dataset_id, "status": "skipped", "revision": current})
            continue

        revision = _hf_dataset_sha(str(config["name"]))
        path = _config_path(root, config)
        data = load_yaml(path)
        data["revision"] = revision
        _dump_yaml(path, data)
        changes.append({"id": dataset_id, "status": "pinned", "revision": revision})
    return changes


def set_base_model(
    root: Path,
    model: str,
    experiment_ids: list[str] | None = None,
    methods: list[str] | None = None,
    validate_model: bool = False,
    pin_revision: bool = False,
) -> list[dict[str, str]]:
    configs = load_all(root)["experiments"]
    selected = experiment_ids or sorted(configs)
    method_filter = set(methods or [])
    model = normalize_cli_path(model)
    model_revision = _hf_model_sha(model) if validate_model or pin_revision else None
    changes: list[dict[str, str]] = []

    for experiment_id in selected:
        if experiment_id not in configs:
            known = ", ".join(sorted(configs))
            raise ConfigError(f"Unknown experiment {experiment_id!r}. Known experiments: {known}")
        config = configs[experiment_id]
        if method_filter and str(config.get("method")) not in method_filter:
            changes.append({"id": experiment_id, "status": "skipped", "base_model": str(config.get("base_model"))})
            continue

        path = _config_path(root, config)
        data = load_yaml(path)
        data["base_model"] = model
        if model_revision:
            data["base_model_revision"] = model_revision
        _dump_yaml(path, data)
        changes.append({"id": experiment_id, "status": "updated", "base_model": model})
    return changes
