from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from tunescope.artifacts import (
    ensure_dir,
    prepared_dataset_path,
    read_jsonl,
    resolve_output_dir,
    run_metadata,
    write_json,
    write_yaml,
)
from tunescope.checkpoints import latest_checkpoint
from tunescope.config import ConfigError, get_experiment, load_all
from tunescope.dataset_setup import artifact_paths
from tunescope.registry import register_artifact


SFT_METHODS = {"lora_sft", "qlora_sft", "full_sft"}


def _deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _reject_placeholder_model(model: str, dry_run: bool, allow_placeholder_model: bool) -> None:
    if model.startswith("TODO_") and not dry_run and not allow_placeholder_model:
        raise ConfigError(
            f"Model is still a placeholder: {model}. "
            "Pin configs/experiments/*.yaml base_model or pass --allow-placeholder-model."
        )


def _train_config(root: Path, experiment: dict[str, Any]) -> dict[str, Any]:
    train_id = experiment.get("train_config")
    if not train_id:
        raise ConfigError(f"{experiment['id']} does not define train_config.")
    configs = load_all(root)
    try:
        base = configs["train"][str(train_id)]
    except KeyError as exc:
        raise ConfigError(f"Unknown train_config {train_id!r}.") from exc
    return _deep_merge(base, experiment.get("overrides"))


def _precision_flags(config: dict[str, Any]) -> dict[str, bool | None]:
    precision = str(config.get("precision", "")).lower()
    return {
        "bf16": True if precision == "bf16" else None,
        "fp16": precision in {"fp16", "float16"},
    }


def _model_revision(experiment: dict[str, Any], model_name: str) -> str | None:
    if Path(model_name).exists():
        return None
    revision = experiment.get("base_model_revision")
    return str(revision) if revision else None


def _filter_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    allowed = set(inspect.signature(cls).parameters)
    return {key: value for key, value in values.items() if key in allowed and value is not None}


def _lora_config(config: dict[str, Any]) -> Any | None:
    lora = config.get("lora")
    if not isinstance(lora, dict):
        return None
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora.get("r", 8)),
        lora_alpha=int(lora.get("alpha", 16)),
        lora_dropout=float(lora.get("dropout", 0.0)),
        target_modules=lora.get("target_modules"),
        bias="none",
    )


def _quantization_config(config: dict[str, Any]) -> Any | None:
    quantization = config.get("quantization")
    if not isinstance(quantization, dict) or not quantization.get("load_in_4bit"):
        return None
    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ConfigError("QLoRA requires bitsandbytes. Run: uv sync --group dev") from exc

    import torch
    from transformers import BitsAndBytesConfig

    dtype_name = str(quantization.get("bnb_4bit_compute_dtype", "bfloat16"))
    compute_dtype = getattr(torch, dtype_name)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(quantization.get("bnb_4bit_quant_type", "nf4")),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=bool(quantization.get("bnb_4bit_use_double_quant", True)),
    )


def _records_to_dataset(records: list[dict[str, Any]]) -> Any:
    from datasets import Dataset

    return Dataset.from_list(records)


def _format_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        rendered = []
        for message in messages:
            rendered.append(f"{message.get('role', 'user')}: {message.get('content', '')}")
        return "\n".join(rendered)


def _sft_dataset(path: Path, tokenizer: Any) -> Any:
    records = read_jsonl(path)
    converted: list[dict[str, Any]] = []
    for record in records:
        if "text" in record:
            converted.append(record)
        elif isinstance(record.get("messages"), list):
            converted.append({"text": _format_messages(tokenizer, record["messages"])})
        else:
            raise ConfigError(f"{path} contains a record without text or messages.")
    return _records_to_dataset(converted)


def _dpo_dataset(path: Path) -> Any:
    records = read_jsonl(path)
    for record in records:
        missing = {"prompt", "chosen", "rejected"} - set(record)
        if missing:
            raise ConfigError(f"{path} contains a DPO record missing: {', '.join(sorted(missing))}")
    return _records_to_dataset(records)


def _training_plan(
    root: Path,
    experiment: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
    command: str,
    require_dataset: bool = True,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    dataset_path = None
    if experiment.get("dataset") is not None:
        if require_dataset:
            dataset_path = prepared_dataset_path(root, experiment)
        else:
            configs = load_all(root)
            dataset_config = configs["datasets"][str(experiment["dataset"])]
            dataset_path, _ = artifact_paths(
                root,
                str(experiment["dataset"]),
                str(dataset_config.get("split", "train")),
                experiment.get("sample_count", "all"),
                int(experiment.get("seed", 42)),
            )
    return {
        **run_metadata(root, experiment, command, output_dir),
        "train_config": config,
        "dataset_path": str(dataset_path) if dataset_path else None,
        "model_output_dir": str(output_dir / "model"),
        "resume_from_checkpoint": resume_from_checkpoint,
    }


def train_sft(
    root: Path,
    experiment_id: str,
    output_dir_arg: str | None = None,
    dry_run: bool = False,
    max_steps: int | None = None,
    allow_placeholder_model: bool = False,
    resume_from_checkpoint: str | None = None,
    auto_resume: bool = False,
) -> Path:
    experiment = get_experiment(experiment_id, root)
    method = str(experiment.get("method"))
    if method not in SFT_METHODS:
        raise ConfigError(f"{experiment_id} is {method}; expected one of {', '.join(sorted(SFT_METHODS))}.")

    output_dir = ensure_dir(resolve_output_dir(root, experiment, output_dir_arg))
    config = _train_config(root, experiment)
    model_name = str(experiment.get("base_model"))
    _reject_placeholder_model(model_name, dry_run, allow_placeholder_model)
    resolved_resume = _resolve_resume_checkpoint(output_dir, resume_from_checkpoint, auto_resume)
    plan = _training_plan(
        root,
        experiment,
        output_dir,
        config,
        "train-sft",
        require_dataset=not dry_run,
        resume_from_checkpoint=resolved_resume,
    )
    write_yaml(output_dir / "run.yaml", plan)

    if dry_run:
        write_json(output_dir / "train_metrics.json", {"status": "dry_run", "experiment_id": experiment_id})
        register_artifact(root, experiment, output_dir, status="dry_run")
        return output_dir

    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    revision = _model_revision(experiment, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_path = prepared_dataset_path(root, experiment)
    if dataset_path is None:
        raise ConfigError(f"{experiment_id} has no dataset.")
    train_dataset = _sft_dataset(dataset_path, tokenizer)

    args_values = {
        "output_dir": str(output_dir / "model"),
        "per_device_train_batch_size": config.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
        "num_train_epochs": config.get("num_train_epochs"),
        "learning_rate": config.get("learning_rate"),
        "warmup_ratio": config.get("warmup_ratio"),
        "weight_decay": config.get("weight_decay"),
        "lr_scheduler_type": config.get("lr_scheduler_type"),
        "max_length": config.get("max_seq_length"),
        "packing": config.get("packing"),
        "gradient_checkpointing": config.get("gradient_checkpointing"),
        "seed": config.get("seed"),
        "do_train": True,
        "save_strategy": "epoch",
        "report_to": "none",
        "model_init_kwargs": {"revision": revision, "trust_remote_code": True} if revision else {"trust_remote_code": True},
        **_precision_flags(config),
    }
    if max_steps is not None:
        args_values["max_steps"] = max_steps
    training_args = SFTConfig(**_filter_kwargs(SFTConfig, args_values))

    peft_config = None if method == "full_sft" else _lora_config(config)
    trainer = SFTTrainer(
        model=model_name,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        quantization_config=_quantization_config(config),
    )
    result = trainer.train(resume_from_checkpoint=resolved_resume)
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))
    metrics = dict(getattr(result, "metrics", {}) or {})
    metrics["status"] = "completed"
    metrics["experiment_id"] = experiment_id
    write_json(output_dir / "train_metrics.json", metrics)
    register_artifact(root, experiment, output_dir, status="completed")
    return output_dir


def _dpo_model_path(root: Path, experiment: dict[str, Any], output_dir: Path) -> str:
    base_model = str(experiment.get("base_model"))
    if not base_model.startswith("TODO_"):
        return base_model
    starts_from = experiment.get("starts_from_experiment")
    if starts_from:
        source = get_experiment(str(starts_from), root)
        source_dir = default_source_dir(root, source)
        model_dir = source_dir / "model"
        if model_dir.exists():
            return str(model_dir)
    return base_model


def default_source_dir(root: Path, experiment: dict[str, Any]) -> Path:
    from tunescope.artifacts import default_output_dir

    return default_output_dir(root, experiment)


def train_dpo(
    root: Path,
    experiment_id: str,
    output_dir_arg: str | None = None,
    dry_run: bool = False,
    max_steps: int | None = None,
    allow_placeholder_model: bool = False,
    resume_from_checkpoint: str | None = None,
    auto_resume: bool = False,
) -> Path:
    experiment = get_experiment(experiment_id, root)
    if experiment.get("method") != "dpo":
        raise ConfigError(f"{experiment_id} is {experiment.get('method')}; expected dpo.")

    output_dir = ensure_dir(resolve_output_dir(root, experiment, output_dir_arg))
    config = _train_config(root, experiment)
    model_name = _dpo_model_path(root, experiment, output_dir)
    _reject_placeholder_model(model_name, dry_run, allow_placeholder_model)
    resolved_resume = _resolve_resume_checkpoint(output_dir, resume_from_checkpoint, auto_resume)
    plan = _training_plan(
        root,
        experiment,
        output_dir,
        config,
        "train-dpo",
        require_dataset=not dry_run,
        resume_from_checkpoint=resolved_resume,
    )
    plan["resolved_model"] = model_name
    write_yaml(output_dir / "run.yaml", plan)

    if dry_run:
        write_json(output_dir / "train_metrics.json", {"status": "dry_run", "experiment_id": experiment_id})
        register_artifact(root, experiment, output_dir, status="dry_run")
        return output_dir

    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    revision = _model_revision(experiment, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset_path = prepared_dataset_path(root, experiment)
    if dataset_path is None:
        raise ConfigError(f"{experiment_id} has no dataset.")
    train_dataset = _dpo_dataset(dataset_path)

    args_values = {
        "output_dir": str(output_dir / "model"),
        "per_device_train_batch_size": config.get("per_device_train_batch_size"),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps"),
        "num_train_epochs": config.get("num_train_epochs"),
        "learning_rate": config.get("learning_rate"),
        "warmup_ratio": config.get("warmup_ratio"),
        "weight_decay": config.get("weight_decay"),
        "lr_scheduler_type": config.get("lr_scheduler_type"),
        "max_length": config.get("max_seq_length"),
        "beta": config.get("beta"),
        "gradient_checkpointing": config.get("gradient_checkpointing"),
        "seed": config.get("seed"),
        "do_train": True,
        "save_strategy": "epoch",
        "report_to": "none",
        "model_init_kwargs": {"revision": revision, "trust_remote_code": True} if revision else {"trust_remote_code": True},
        **_precision_flags(config),
    }
    if max_steps is not None:
        args_values["max_steps"] = max_steps
    training_args = DPOConfig(**_filter_kwargs(DPOConfig, args_values))

    trainer = DPOTrainer(
        model=model_name,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=_lora_config(config),
    )
    result = trainer.train(resume_from_checkpoint=resolved_resume)
    trainer.save_model(str(output_dir / "model"))
    tokenizer.save_pretrained(str(output_dir / "model"))
    metrics = dict(getattr(result, "metrics", {}) or {})
    metrics["status"] = "completed"
    metrics["experiment_id"] = experiment_id
    write_json(output_dir / "train_metrics.json", metrics)
    register_artifact(root, experiment, output_dir, status="completed")
    return output_dir


def _resolve_resume_checkpoint(
    output_dir: Path,
    resume_from_checkpoint: str | None,
    auto_resume: bool,
) -> str | None:
    if resume_from_checkpoint:
        return resume_from_checkpoint
    if not auto_resume:
        return None
    checkpoint = latest_checkpoint(output_dir)
    return str(checkpoint) if checkpoint else None
