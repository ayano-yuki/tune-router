from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from config import LABELS
from types_local import Metrics
from utils import enable_system_cert_store


def import_inference_deps():
    enable_system_cert_store()
    try:
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise SystemExit(
            "Inference dependencies are missing. Run with uv so pyproject dependencies are installed:\n"
            "  uv run --project .\\poc python .\\poc\\src\\cli.py predict-qwen\n"
            f"Missing import: {exc}"
        ) from exc
    return {
        "torch": torch,
        "PeftModel": PeftModel,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
    }


def import_training_deps():
    deps = import_inference_deps()
    try:
        import numpy as np
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Run with uv so pyproject dependencies are installed:\n"
            "  uv run --project .\\poc python .\\poc\\src\\cli.py train-qwen\n"
            f"Missing import: {exc}"
        ) from exc
    deps.update(
        {
            "np": np,
            "Dataset": Dataset,
            "LoraConfig": LoraConfig,
            "TaskType": TaskType,
            "get_peft_model": get_peft_model,
            "DataCollatorWithPadding": DataCollatorWithPadding,
            "Trainer": Trainer,
            "TrainingArguments": TrainingArguments,
            "set_seed": set_seed,
        }
    )
    return deps


def compute_metrics_builder(np_module):
    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        predictions = np_module.argmax(logits, axis=-1)
        correct = int((predictions == label_ids).sum())
        total = int(len(label_ids))
        f1_values = []
        for label_id in range(len(LABELS)):
            tp = int(((predictions == label_id) & (label_ids == label_id)).sum())
            fp = int(((predictions == label_id) & (label_ids != label_id)).sum())
            fn = int(((predictions != label_id) & (label_ids == label_id)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            f1_values.append(f1)
        return {
            "accuracy": correct / total if total else 0.0,
            "macro_f1": sum(f1_values) / len(f1_values),
        }

    return compute_metrics


def training_args_kwargs(training_args_cls, args: argparse.Namespace) -> dict:
    kwargs = {
        "output_dir": str(args.output),
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "report_to": "none",
        "remove_unused_columns": False,
        "warmup_ratio": 0.1,
        "max_grad_norm": 0.3,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 2,
    }
    signature = inspect.signature(training_args_cls.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    if "fp16" in signature.parameters:
        kwargs["fp16"] = args.fp16
    if "bf16" in signature.parameters:
        kwargs["bf16"] = args.bf16
    return kwargs


def trainer_processing_kwargs(trainer_cls, tokenizer) -> dict:
    signature = inspect.signature(trainer_cls.__init__)
    if "processing_class" in signature.parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in signature.parameters:
        return {"tokenizer": tokenizer}
    return {}


def tokenize_records(records: list[dict], tokenizer, dataset_cls, max_length: int):
    label2id = {label: index for index, label in enumerate(LABELS)}
    dataset = dataset_cls.from_list(
        [{"text": record["text"], "labels": label2id[record["gold_label"]]} for record in records]
    )
    return dataset.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )


def load_qwen_router(base_model: str, adapter: Path | None, deps: dict):
    label2id = {label: index for index, label in enumerate(LABELS)}
    id2label = {index: label for label, index in label2id.items()}
    tokenizer = deps["AutoTokenizer"].from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = deps["AutoModelForSequenceClassification"].from_pretrained(
        base_model,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=deps["torch"].float32,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if adapter:
        adapter_path = Path(adapter)
        config_path = adapter_path / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"adapter_config.json not found in adapter path: {adapter_path}. "
                "Use a directory that contains a full PEFT adapter, such as a checkpoint directory."
            )
        model = deps["PeftModel"].from_pretrained(model, str(adapter_path))
    return tokenizer, model


def evaluate_predictions(records: list[dict], predictions: list[dict]) -> tuple[Metrics, dict]:
    confusion = {actual: {pred: 0 for pred in LABELS} for actual in LABELS}
    correct = 0
    for record, prediction in zip(records, predictions):
        actual = record["gold_label"]
        predicted = prediction["predicted_label"]
        confusion[actual][predicted] += 1
        correct += int(actual == predicted)

    f1_values = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in LABELS if actual != label)
        fn = sum(confusion[label][pred] for pred in LABELS if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)

    total = len(records)
    return Metrics(
        accuracy=correct / total if total else 0.0,
        macro_f1=sum(f1_values) / len(f1_values),
        total=total,
        correct=correct,
    ), confusion
