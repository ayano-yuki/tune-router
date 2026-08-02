from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from config import LABELS, OSS_SOURCES, ROUTER_BASE_MODEL
from json_store import read_records, write_dataset_files, write_json
from oss_data import build_oss_dataset
from qwen_router import (
    compute_metrics_builder,
    evaluate_predictions,
    import_training_deps,
    load_qwen_router,
    tokenize_records,
    training_args_kwargs,
    trainer_processing_kwargs,
)
from reporting import report_markdown
from splitting import split_dataset
from synthetic_data import build_synthetic_dataset


FIXED_EPOCHS = 2.0
FIXED_LEARNING_RATE = 2e-5
FIXED_MAX_LENGTH = 256
FIXED_LORA_R = 8
FIXED_LORA_ALPHA = 16
FIXED_FP16 = False
FIXED_BF16 = False


def cmd_prepare_data(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    records = build_oss_dataset(
        args.per_label,
        args.seed,
        streaming=args.streaming,
        cache_dir=args.dataset_cache_dir,
        max_source_scan=args.max_source_scan,
    )
    splits = split_dataset(records)
    validate_split_text_diversity(splits)
    write_dataset_files(out_dir, splits, args.per_label, args.seed, data_origin="oss_huggingface")
    (out_dir / "report.md").write_text(
        report_markdown(splits["train"], splits["dev"], splits["test"]),
        encoding="utf-8",
        newline="\n",
    )
    print_data_outputs(out_dir, args.per_label)
    print("sources:")
    for label, source in OSS_SOURCES.items():
        print(f"  {label}: {source['dataset']} ({source['license']})")


def cmd_prepare_synthetic_data(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    records = build_synthetic_dataset(args.per_label, args.seed)
    splits = split_dataset(records)
    validate_split_text_diversity(splits)
    write_dataset_files(out_dir, splits, args.per_label, args.seed, data_origin="synthetic_poc")
    (out_dir / "report.md").write_text(
        report_markdown(splits["train"], splits["dev"], splits["test"]),
        encoding="utf-8",
        newline="\n",
    )
    print_data_outputs(out_dir, args.per_label)
    print("data origin: synthetic_poc")


def print_data_outputs(out_dir: Path, per_label: int) -> None:
    print(f"dataset: {out_dir / 'dataset.json'}")
    print(f"train: {out_dir / 'train.json'}")
    print(f"dev: {out_dir / 'dev.json'}")
    print(f"test: {out_dir / 'test.json'}")
    print(f"requested per label: about {per_label}")


def validate_split_text_diversity(splits: dict[str, list[dict]]) -> None:
    for split_name, rows in splits.items():
        by_label: dict[str, list[str]] = {label: [] for label in LABELS}
        for row in rows:
            label = row.get("gold_label")
            text = str(row.get("text", "")).strip()
            if label in by_label and text:
                by_label[label].append(text)

        for label, texts in by_label.items():
            if not texts:
                continue
            unique_count = len(Counter(text.lower() for text in texts))
            if len(texts) >= 20 and unique_count <= 1:
                raise RuntimeError(
                    f"Low text diversity detected: split={split_name} label={label} "
                    f"count={len(texts)} unique={unique_count}. "
                    "Likely selecting a constant prompt field from source data."
                )


def cmd_train_qwen(args: argparse.Namespace) -> None:
    # Enforce file-defined hyperparameters regardless of CLI overrides.
    args.epochs = FIXED_EPOCHS
    args.learning_rate = FIXED_LEARNING_RATE
    args.max_length = FIXED_MAX_LENGTH
    args.lora_r = FIXED_LORA_R
    args.lora_alpha = FIXED_LORA_ALPHA
    args.fp16 = FIXED_FP16
    args.bf16 = FIXED_BF16

    deps = import_training_deps()
    deps["set_seed"](args.seed)

    train_records = read_records(Path(args.train))
    dev_records = read_records(Path(args.dev))
    tokenizer, model = load_qwen_router(args.base_model, None, deps)

    lora_config = deps["LoraConfig"](
        task_type=deps["TaskType"].SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
    )
    model = deps["get_peft_model"](model, lora_config)

    train_dataset = tokenize_records(train_records, tokenizer, deps["Dataset"], args.max_length)
    dev_dataset = tokenize_records(dev_records, tokenizer, deps["Dataset"], args.max_length)
    training_args = deps["TrainingArguments"](**training_args_kwargs(deps["TrainingArguments"], args))
    trainer = deps["Trainer"](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=deps["DataCollatorWithPadding"](tokenizer=tokenizer),
        compute_metrics=compute_metrics_builder(deps["np"]),
        **trainer_processing_kwargs(deps["Trainer"], tokenizer),
    )
    trainer.train()

    # Abort if training diverged; do not save a collapsed adapter.
    for entry in trainer.state.log_history:
        eval_loss = entry.get("eval_loss")
        if eval_loss is not None and not math.isfinite(float(eval_loss)):
            raise RuntimeError(
                "Training diverged: eval_loss is non-finite (NaN/Inf). "
                "Retry without mixed precision and with a fresh output directory."
            )

    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    write_json(
        Path(args.output) / "router_config.json",
        {
            "base_model": args.base_model,
            "labels": list(LABELS),
            "target_models": {label: config["target_model"] for label, config in LABELS.items()},
            "fine_tuning": {
                "method": "lora_sequence_classification",
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "max_length": args.max_length,
            },
        },
    )
    print(f"saved Qwen router adapter to {args.output}")


def cmd_evaluate_qwen(args: argparse.Namespace) -> None:
    deps = import_training_deps()
    records = read_records(Path(args.data))
    tokenizer, model = load_qwen_router(args.base_model, Path(args.adapter), deps)
    torch = deps["torch"]
    model.eval()

    predictions = []
    label_names = list(LABELS)
    with torch.no_grad():
        for record in records:
            inputs = tokenizer(
                record["text"],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            logits = model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()
            best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            predicted_label = label_names[best_index]
            predictions.append(
                {
                    "question_id": record["question_id"],
                    "text": record["text"],
                    "gold_label": record["gold_label"],
                    "predicted_label": predicted_label,
                    "target_model": LABELS[predicted_label]["target_model"],
                    "confidence": round(probabilities[best_index], 6),
                    "correct": predicted_label == record["gold_label"],
                    "scores": {label: probabilities[index] for index, label in enumerate(label_names)},
                }
            )

    metrics, confusion = evaluate_predictions(records, predictions)
    write_json(Path(args.predictions), {"records": predictions})
    if args.report:
        train = read_records(Path(args.train)) if args.train else []
        dev = read_records(Path(args.dev)) if args.dev else []
        mistakes = [prediction for prediction in predictions if not prediction["correct"]]
        Path(args.report).write_text(
            report_markdown(train, dev, records, metrics, confusion, mistakes),
            encoding="utf-8",
            newline="\n",
        )
    print(f"accuracy={metrics.accuracy:.3f} macro_f1={metrics.macro_f1:.3f}")


def cmd_predict_qwen(args: argparse.Namespace) -> None:
    deps = import_training_deps()
    tokenizer, model = load_qwen_router(args.base_model, Path(args.adapter), deps)
    torch = deps["torch"]
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=args.max_length)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        logits = model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()
    label_names = list(LABELS)
    best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
    label = label_names[best_index]
    print(
        json.dumps(
            {
                "base_model": args.base_model,
                "adapter": str(args.adapter),
                "predicted_label": label,
                "target_model": LABELS[label]["target_model"],
                "confidence": probabilities[best_index],
                "scores": {name: probabilities[index] for index, name in enumerate(label_names)},
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def cmd_run(args: argparse.Namespace) -> None:
    cmd_prepare_data(args)
    train_args = argparse.Namespace(**vars(args))
    train_args.train = str(Path(args.out) / "train.json")
    train_args.dev = str(Path(args.out) / "dev.json")
    train_args.output = Path(args.out) / "qwen-router-lora"
    cmd_train_qwen(train_args)

    eval_args = argparse.Namespace(**vars(args))
    eval_args.data = str(Path(args.out) / "test.json")
    eval_args.train = str(Path(args.out) / "train.json")
    eval_args.dev = str(Path(args.out) / "dev.json")
    eval_args.adapter = str(train_args.output)
    eval_args.predictions = str(Path(args.out) / "predictions_test.json")
    eval_args.report = str(Path(args.out) / "report.md")
    cmd_evaluate_qwen(eval_args)


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", default=ROUTER_BASE_MODEL)
    parser.add_argument("--epochs", type=float, default=FIXED_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=FIXED_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=FIXED_MAX_LENGTH)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=FIXED_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=FIXED_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")


def add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--per-label", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="poc/artifacts")


def add_oss_data_args(parser: argparse.ArgumentParser) -> None:
    add_data_args(parser)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--max-source-scan", type=int, default=200_000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TuneRouter Qwen2.5-0.5B PoC")
    subparsers = parser.add_subparsers(required=True)

    prepare = subparsers.add_parser("prepare-data", help="build local JSON data from OSS datasets")
    add_oss_data_args(prepare)
    prepare.set_defaults(func=cmd_prepare_data)

    synthetic = subparsers.add_parser("prepare-synthetic-data", help="generate synthetic local JSON data")
    add_data_args(synthetic)
    synthetic.set_defaults(func=cmd_prepare_synthetic_data)

    train = subparsers.add_parser("train-qwen", help="fine-tune Qwen2.5-0.5B router with LoRA")
    train.add_argument("--train", default="poc/artifacts/train.json")
    train.add_argument("--dev", default="poc/artifacts/dev.json")
    train.add_argument("--output", type=Path, default=Path("poc/artifacts/qwen-router-lora"))
    train.add_argument("--seed", type=int, default=42)
    add_training_args(train)
    train.set_defaults(func=cmd_train_qwen)

    evaluate = subparsers.add_parser("evaluate-qwen", help="evaluate a fine-tuned Qwen router")
    evaluate.add_argument("--data", default="poc/artifacts/test.json")
    evaluate.add_argument("--train", default="poc/artifacts/train.json")
    evaluate.add_argument("--dev", default="poc/artifacts/dev.json")
    evaluate.add_argument("--adapter", default="poc/artifacts/qwen-router-lora")
    evaluate.add_argument("--predictions", default="poc/artifacts/predictions_test.json")
    evaluate.add_argument("--report", default="poc/artifacts/report.md")
    evaluate.add_argument("--base-model", default=ROUTER_BASE_MODEL)
    evaluate.add_argument("--max-length", type=int, default=256)
    evaluate.set_defaults(func=cmd_evaluate_qwen)

    predict = subparsers.add_parser("predict-qwen", help="route one question with a fine-tuned Qwen router")
    predict.add_argument("--adapter", default="poc/artifacts/qwen-router-lora")
    predict.add_argument("--base-model", default=ROUTER_BASE_MODEL)
    predict.add_argument("--max-length", type=int, default=256)
    predict.add_argument("text")
    predict.set_defaults(func=cmd_predict_qwen)

    run = subparsers.add_parser("run", help="prepare data, fine-tune Qwen, and evaluate")
    add_oss_data_args(run)
    add_training_args(run)
    run.set_defaults(func=cmd_run)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
