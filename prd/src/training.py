from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

from learned import PlanClient, parse_plan


DEFAULT_ORCHESTRATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def import_training_dependencies() -> dict[str, Any]:
    try:
        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install the training extra:\n"
            "  uv sync --project .\\prd --extra training --system-certs\n"
            f"Missing import: {exc}"
        ) from exc
    return {
        "torch": torch,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForSeq2Seq": DataCollatorForSeq2Seq,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def train_orchestrator(args: Any) -> None:
    deps = import_training_dependencies()
    deps["set_seed"](args.seed)
    train_records = read_jsonl(Path(args.train))
    dev_records = read_jsonl(Path(args.dev)) if args.dev and Path(args.dev).exists() else []
    if not train_records:
        raise ValueError("training dataset is empty")

    tokenizer = deps["AutoTokenizer"].from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = deps["AutoModelForCausalLM"].from_pretrained(
        args.base_model,
        torch_dtype=_training_dtype(deps["torch"], args),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    lora = deps["LoraConfig"](
        task_type=deps["TaskType"].CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = deps["get_peft_model"](model, lora)
    train_dataset = _tokenized_dataset(train_records, tokenizer, deps["torch"], args.max_length)
    dev_dataset = _tokenized_dataset(dev_records, tokenizer, deps["torch"], args.max_length) if dev_records else None
    training_args = deps["TrainingArguments"](
        **_training_args_kwargs(deps["TrainingArguments"], args, bool(dev_dataset))
    )
    trainer = deps["Trainer"](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=deps["DataCollatorForSeq2Seq"](
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    (Path(args.output) / "orchestrator_config.json").write_text(
        json.dumps(
            {
                "format": "tune-orchestrator-adapter-v1",
                "base_model": args.base_model,
                "training_method": "lora_causal_lm_sft",
                "train_examples": len(train_records),
                "dev_examples": len(dev_records),
                "max_length": args.max_length,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
        newline="\n",
    )


class LocalAdapterPlanClient(PlanClient):
    def __init__(
        self,
        adapter: Path,
        base_model: str | None = None,
        max_new_tokens: int = 1024,
        device: str = "auto",
    ) -> None:
        deps = import_training_dependencies()
        config_path = adapter / "orchestrator_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        self.base_model = base_model or config.get("base_model") or DEFAULT_ORCHESTRATOR_MODEL
        self.tokenizer = deps["AutoTokenizer"].from_pretrained(self.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = deps["AutoModelForCausalLM"].from_pretrained(self.base_model, trust_remote_code=True)
        self.model = deps["PeftModel"].from_pretrained(model, str(adapter))
        self.torch = deps["torch"]
        target_device = ("cuda" if self.torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model.to(target_device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        completion = generated[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()


def evaluate_adapter(
    client: PlanClient,
    records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    valid = 0
    graph_correct = 0
    labels_correct = 0
    predictions = []
    for record in records:
        messages = record["messages"]
        expected = parse_plan(str(messages[-1]["content"]))
        raw_prediction = client.generate(messages[:-1])
        error = None
        predicted: dict[str, Any] | None = None
        try:
            predicted = parse_plan(raw_prediction)
            valid += 1
            graph_correct += int(predicted.get("graph_id") == expected.get("graph_id"))
            labels_correct += int(predicted.get("selected_labels") == expected.get("selected_labels"))
        except (ValueError, json.JSONDecodeError) as exc:
            error = str(exc)
        predictions.append(
            {
                "query_id": record.get("metadata", {}).get("query_id"),
                "expected": expected,
                "predicted": predicted,
                "raw_prediction": raw_prediction,
                "error": error,
            }
        )
    total = len(records)
    metrics = {
        "total": total,
        "valid_plan_rate": valid / total if total else 0.0,
        "graph_accuracy": graph_correct / total if total else 0.0,
        "selected_labels_exact_match": labels_correct / total if total else 0.0,
    }
    return metrics, predictions


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tokenized_dataset(records: list[dict[str, Any]], tokenizer: Any, torch: Any, max_length: int) -> Any:
    items = []
    for record in records:
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 3 or messages[-1].get("role") != "assistant":
            raise ValueError("each SFT record must end with an assistant message")
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        prompt_length = min(len(prompt_ids), len(full_ids))
        labels = [-100] * prompt_length + full_ids[prompt_length:]
        if not any(label != -100 for label in labels):
            raise ValueError("max_length truncates the complete assistant target")
        items.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels})

    class TokenizedDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(items)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return items[index]

    return TokenizedDataset()


def _training_dtype(torch: Any, args: Any) -> Any:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def _training_args_kwargs(training_args_cls: Any, args: Any, has_dev: bool) -> dict[str, Any]:
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
        "warmup_ratio": 0.05,
        "lr_scheduler_type": "cosine",
        "save_total_limit": 2,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    if has_dev:
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = "eval_loss"
        kwargs["greater_is_better"] = False
    signature = inspect.signature(training_args_cls.__init__)
    evaluation_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[evaluation_key] = "epoch" if has_dev else "no"
    return kwargs
