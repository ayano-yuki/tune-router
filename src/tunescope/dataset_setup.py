from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tunescope.config import ConfigError, load_all


TODO_REVISION = "TODO_PIN_HF_COMMIT"


@dataclass(frozen=True)
class PreparedDataset:
    dataset_id: str
    output_path: Path
    manifest_path: Path
    record_count: int
    skipped: bool = False
    invalid_record_count: int = 0


def _lookup(record: dict[str, Any], path: str | None) -> Any:
    if not path:
        return None
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _conversation_role(value: str) -> str:
    roles = {
        "human": "user",
        "gpt": "assistant",
    }
    return roles.get(value, value)


def _conversation_to_text(value: Any) -> str:
    if not isinstance(value, list):
        return ""

    turns = []
    for item in value:
        if not isinstance(item, dict):
            return ""
        role = _conversation_role(_as_text(item.get("role") or item.get("from")))
        content = _as_text(item.get("content") or item.get("value"))
        if not content:
            return ""
        turns.append((role, content))

    if len(turns) == 1:
        return turns[0][1]
    return "\n".join(f"{role}: {content}" if role else content for role, content in turns)


def _field_text(record: dict[str, Any], field: str | None) -> str:
    value = _lookup(record, field)
    if isinstance(value, list):
        return _conversation_to_text(value)
    return _as_text(value)


def _first_text(record: dict[str, Any], fields: Iterable[str | None]) -> str:
    for field in fields:
        value = _field_text(record, field)
        if value:
            return value
    return ""


def _messages_from_existing(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None
    messages = []
    for item in value:
        if not isinstance(item, dict):
            return None
        role = _as_text(item.get("role"))
        content = _as_text(item.get("content"))
        if not role or not content:
            return None
        messages.append({"role": role, "content": content})
    if len(messages) < 2:
        return None
    return {"messages": messages}


def normalize_record(record: dict[str, Any], dataset_config: dict[str, Any]) -> dict[str, Any]:
    normalization = dataset_config.get("normalization")
    if not isinstance(normalization, dict):
        return record

    target_format = normalization.get("target_format")
    fields = normalization.get("fields")
    if not isinstance(fields, dict):
        fields = {}

    if target_format == "messages":
        existing = _messages_from_existing(record.get("messages"))
        if existing is not None:
            return existing

        user_fields = [
            fields.get("user"),
            "instruction",
            "input",
            "question",
            "prompt",
            "text",
        ]
        assistant_fields = [
            fields.get("assistant"),
            "output",
            "response",
            "answer",
            "completion",
            "summary",
        ]
        instruction = _first_text(record, (field for field in user_fields if field))
        input_text = _as_text(record.get("input"))
        assistant = _first_text(record, (field for field in assistant_fields if field))

        if input_text and input_text != instruction and "input" not in fields.values():
            user = f"{instruction}\n\n{input_text}" if instruction else input_text
        else:
            user = instruction

        if not user or not assistant:
            raise ConfigError(
                f"Cannot normalize record to messages for dataset {dataset_config.get('id')!r}; "
                "expected user and assistant text fields."
            )

        return {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        }

    if target_format == "preference":
        prompt = _first_text(record, [fields.get("prompt"), "prompt", "instruction", "input"])
        chosen = _first_text(record, [fields.get("chosen"), "chosen", "chosen_response", "accept"])
        rejected = _first_text(record, [fields.get("rejected"), "rejected", "rejected_response", "reject"])
        if not prompt or not chosen or not rejected:
            raise ConfigError(
                f"Cannot normalize record to preference for dataset {dataset_config.get('id')!r}; "
                "expected prompt, chosen, and rejected text fields."
            )
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    return record


def sample_records(records: list[dict[str, Any]], sample_count: int | str, seed: int) -> list[dict[str, Any]]:
    if sample_count == "all":
        return list(records)
    if not isinstance(sample_count, int) or sample_count < 1:
        raise ConfigError("sample_count must be a positive integer or 'all'.")
    if sample_count >= len(records):
        return list(records)
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    selected = sorted(indices[:sample_count])
    return [records[index] for index in selected]


def _json_default(value: Any) -> str:
    return str(value)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default))
            handle.write("\n")
            count += 1
    return count


def _sample_label(sample_count: int | str) -> str:
    return f"n{sample_count}" if isinstance(sample_count, int) else "all"


def artifact_paths(
    root: Path,
    dataset_id: str,
    split: str,
    sample_count: int | str,
    seed: int,
) -> tuple[Path, Path]:
    stem = f"{dataset_id}__{split}__{_sample_label(sample_count)}__seed{seed}"
    output_path = root / "datasets" / "prepared" / dataset_id / f"{stem}.jsonl"
    manifest_path = root / "datasets" / "manifests" / f"{stem}.yaml"
    return output_path, manifest_path


def write_manifest(
    path: Path,
    dataset_config: dict[str, Any],
    output_path: Path,
    record_count: int,
    sample_count: int | str,
    seed: int,
    floating_revision: bool,
    invalid_record_count: int = 0,
) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ConfigError("PyYAML is required. Run: uv sync --group dev") from exc

    normalization = dataset_config.get("normalization", {"target_format": "raw"})
    if isinstance(normalization, dict):
        normalization = dict(normalization)
        if invalid_record_count:
            normalization["skipped_invalid_records"] = invalid_record_count

    manifest = {
        "dataset": {
            "id": dataset_config["id"],
            "name": dataset_config["name"],
            "revision": dataset_config.get("revision"),
            "revision_policy": "floating" if floating_revision else "pinned",
            "split": dataset_config.get("split"),
            "subset": dataset_config.get("subset"),
            "usage": dataset_config.get("usage"),
            "sample_count": sample_count,
            "seed": seed,
        },
        "artifact": {
            "path": str(output_path),
            "format": "jsonl",
            "records": record_count,
        },
        "normalization": normalization,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(manifest, handle, allow_unicode=True, sort_keys=False)


def _load_hf_records(dataset_config: dict[str, Any], allow_floating_revision: bool) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ConfigError("datasets is required. Run: uv sync --group dev") from exc

    revision = dataset_config.get("revision")
    if revision == TODO_REVISION:
        if not allow_floating_revision:
            raise ConfigError(
                f"Dataset {dataset_config['id']!r} has no pinned revision. "
                "Set configs/datasets/*.yaml revision or pass --allow-floating-revision."
            )
        revision = None

    kwargs: dict[str, Any] = {"split": dataset_config.get("split", "train")}
    if revision:
        kwargs["revision"] = revision
    if dataset_config.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True
    subset = dataset_config.get("subset")
    if subset:
        dataset = load_dataset(dataset_config["name"], subset, **kwargs)
    else:
        dataset = load_dataset(dataset_config["name"], **kwargs)
    return [dict(record) for record in dataset]


def prepare_dataset(
    root: Path,
    dataset_id: str,
    sample_count: int | str,
    seed: int,
    allow_floating_revision: bool = False,
    force: bool = False,
) -> PreparedDataset:
    configs = load_all(root)
    datasets = configs["datasets"]
    if dataset_id not in datasets:
        known = ", ".join(sorted(datasets))
        raise ConfigError(f"Unknown dataset {dataset_id!r}. Known datasets: {known}")

    dataset_config = datasets[dataset_id]
    split = str(dataset_config.get("split", "train"))
    output_path, manifest_path = artifact_paths(root, dataset_id, split, sample_count, seed)
    if output_path.exists() and manifest_path.exists() and not force:
        return PreparedDataset(dataset_id, output_path, manifest_path, _count_jsonl(output_path), skipped=True)

    records = _load_hf_records(dataset_config, allow_floating_revision)
    sampled = sample_records(records, sample_count, seed)
    normalization = dataset_config.get("normalization")
    skip_invalid = isinstance(normalization, dict) and bool(normalization.get("skip_invalid"))
    normalized = []
    invalid_record_count = 0
    for record in sampled:
        try:
            normalized.append(normalize_record(record, dataset_config))
        except ConfigError:
            if not skip_invalid:
                raise
            invalid_record_count += 1
    if not normalized:
        raise ConfigError(f"Dataset {dataset_id!r} has no valid records after normalization.")
    record_count = write_jsonl(output_path, normalized)
    write_manifest(
        manifest_path,
        dataset_config,
        output_path,
        record_count,
        sample_count,
        seed,
        floating_revision=dataset_config.get("revision") == TODO_REVISION and allow_floating_revision,
        invalid_record_count=invalid_record_count,
    )
    return PreparedDataset(dataset_id, output_path, manifest_path, record_count, invalid_record_count=invalid_record_count)


def _count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def experiment_dataset_requests(
    root: Path,
    experiment_ids: list[str],
) -> list[tuple[str, int | str, int]]:
    configs = load_all(root)
    requests: list[tuple[str, int | str, int]] = []
    seen: set[tuple[str, int | str, int]] = set()
    for experiment_id in experiment_ids:
        experiment = configs["experiments"].get(experiment_id)
        if experiment is None:
            known = ", ".join(sorted(configs["experiments"]))
            raise ConfigError(f"Unknown experiment {experiment_id!r}. Known experiments: {known}")

        dataset_id = experiment.get("dataset")
        if dataset_id is None:
            continue
        sample_count = experiment.get("sample_count", "all")
        seed = int(experiment.get("seed", 42))
        key = (str(dataset_id), sample_count, seed)
        if key not in seen:
            requests.append(key)
            seen.add(key)
    return requests
