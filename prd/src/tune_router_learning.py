from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tune_constants import LABELS
from tune_training import import_training_dependencies


ROUTER_DATA_FORMAT = "tune-router-training-data-v1"
ROUTER_PROTOTYPE_FORMAT = "tune-router-prototype-v1"
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+|[\u3040-\u30ff\u3400-\u9fff]+")


@dataclass(frozen=True)
class RouterDataConfig:
    dev_ratio: float = 0.15
    test_ratio: float = 0.15
    max_per_label: int | None = None
    min_text_chars: int = 8
    seed: int = 42


@dataclass(frozen=True)
class RouterContinualConfig:
    min_rating: int = 4
    include_failed_corrections: bool = True
    max_per_label: int | None = None
    seed: int = 42


@dataclass(frozen=True)
class RouterMergeConfig:
    continual_ratio: float = 0.35
    max_per_label: int | None = None
    seed: int = 42


def build_router_pretrain_dataset(records: list[dict[str, Any]], config: RouterDataConfig | None = None) -> dict[str, Any]:
    cfg = config or RouterDataConfig()
    normalized = _dedupe_records(_normalize_source_record(record, "pretrain", cfg.min_text_chars) for record in records)
    if cfg.max_per_label is not None:
        normalized = _cap_per_label(normalized, cfg.max_per_label, cfg.seed)
    splits = _split_records(normalized, cfg.dev_ratio, cfg.test_ratio, cfg.seed)
    return _dataset_document("pretrain", splits, cfg)


def build_router_continual_dataset(traces: list[dict[str, Any]], config: RouterContinualConfig | None = None) -> dict[str, Any]:
    cfg = config or RouterContinualConfig()
    records = []
    for trace in traces:
        records.extend(_examples_from_trace(trace, cfg))
    normalized = _dedupe_records(records)
    if cfg.max_per_label is not None:
        normalized = _cap_per_label(normalized, cfg.max_per_label, cfg.seed)
    return _dataset_document("continual", {"train": normalized, "dev": [], "test": []}, cfg)


def merge_router_training_datasets(
    *,
    base: dict[str, Any],
    continual: dict[str, Any],
    config: RouterMergeConfig | None = None,
) -> dict[str, Any]:
    cfg = config or RouterMergeConfig()
    base_train = list(_records_from_document(base, "train"))
    continual_train = list(_records_from_document(continual, "train"))
    if cfg.continual_ratio <= 0:
        merged_train = base_train
    else:
        max_continual = math.ceil(len(base_train) * cfg.continual_ratio / max(1e-9, 1.0 - cfg.continual_ratio))
        merged_train = base_train + _stable_sample(continual_train, max_continual, cfg.seed)
    if cfg.max_per_label is not None:
        merged_train = _cap_per_label(merged_train, cfg.max_per_label, cfg.seed)
    splits = {
        "train": _dedupe_records(merged_train),
        "dev": list(_records_from_document(base, "dev")) + list(_records_from_document(continual, "dev")),
        "test": list(_records_from_document(base, "test")),
    }
    return _dataset_document("merged", splits, cfg)


def train_router_prototype(records: list[dict[str, Any]], *, min_token_count: int = 1) -> dict[str, Any]:
    by_label: dict[str, Counter[str]] = {label: Counter() for label in LABELS}
    doc_freq: Counter[str] = Counter()
    examples = 0
    for record in records:
        label = str(record.get("gold_label", ""))
        if label not in by_label:
            continue
        tokens = _tokens(str(record.get("text", "")))
        if not tokens:
            continue
        examples += 1
        counts = Counter(tokens)
        by_label[label].update(counts)
        doc_freq.update(counts)
    if examples == 0:
        raise ValueError("router prototype training data is empty")
    vocabulary = sorted(token for token, count in doc_freq.items() if count >= min_token_count)
    idf = {token: math.log((examples + 1) / (doc_freq[token] + 1)) + 1.0 for token in vocabulary}
    centroids = {}
    label_counts = {}
    for label, counts in by_label.items():
        label_total = sum(counts.values())
        label_counts[label] = label_total
        if label_total <= 0:
            centroids[label] = {}
            continue
        raw = {token: (counts[token] / label_total) * idf[token] for token in vocabulary if counts[token] > 0}
        norm = math.sqrt(sum(value * value for value in raw.values())) or 1.0
        centroids[label] = {token: value / norm for token, value in raw.items()}
    return {
        "format": ROUTER_PROTOTYPE_FORMAT,
        "created_from": {
            "examples": examples,
            "labels": list(LABELS),
            "min_token_count": min_token_count,
            "training_digest": _records_digest(records),
        },
        "idf": idf,
        "centroids": centroids,
        "label_token_counts": label_counts,
    }


def predict_router_prototype(model: dict[str, Any], text: str) -> dict[str, Any]:
    vector = _tfidf_vector(text, model.get("idf", {}))
    raw_scores = {}
    for label in LABELS:
        centroid = model.get("centroids", {}).get(label, {})
        raw_scores[label] = sum(vector.get(token, 0.0) * float(weight) for token, weight in centroid.items())
    return {"scores": _softmax_scores(raw_scores), "raw_scores": raw_scores}


def evaluate_router_prototype(model: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {label: {candidate: 0 for candidate in LABELS} for label in LABELS}
    correct = 0
    total = 0
    for record in records:
        actual = str(record.get("gold_label", ""))
        if actual not in LABELS:
            continue
        scores = predict_router_prototype(model, str(record.get("text", "")))["scores"]
        predicted = max(scores.items(), key=lambda item: (item[1], item[0]))[0]
        confusion[actual][predicted] += 1
        correct += int(actual == predicted)
        total += 1
    per_label = {}
    f1_values = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[actual][label] for actual in LABELS if actual != label)
        fn = sum(confusion[label][pred] for pred in LABELS if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "total": total,
        "correct": correct,
        "per_label": per_label,
        "confusion": confusion,
    }


def train_router_lora(args: Any) -> None:
    deps = _router_training_deps()
    deps["set_seed"](args.seed)
    train_records = load_router_records(Path(args.train))
    dev_records = load_router_records(Path(args.dev)) if args.dev and Path(args.dev).exists() else []
    if not train_records:
        raise ValueError("router training dataset is empty")
    label2id = {label: index for index, label in enumerate(LABELS)}
    id2label = {index: label for label, index in label2id.items()}
    tokenizer = deps["AutoTokenizer"].from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = deps["AutoModelForSequenceClassification"].from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        torch_dtype=_training_dtype(deps["torch"], args),
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    lora = deps["LoraConfig"](
        task_type=deps["TaskType"].SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["score"],
    )
    model = deps["get_peft_model"](model, lora)
    train_dataset = _router_tokenized_dataset(train_records, tokenizer, deps["torch"], args.max_length)
    dev_dataset = _router_tokenized_dataset(dev_records, tokenizer, deps["torch"], args.max_length) if dev_records else None
    training_args = deps["TrainingArguments"](**_router_training_args_kwargs(deps["TrainingArguments"], args, bool(dev_dataset)))
    trainer = deps["Trainer"](
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=deps["DataCollatorWithPadding"](tokenizer=tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    write_json(
        Path(args.output) / "router_config.json",
        {
            "format": "tune-router-adapter-v1",
            "base_model": args.base_model,
            "labels": list(LABELS),
            "training_method": "lora_sequence_classification",
            "train_examples": len(train_records),
            "dev_examples": len(dev_records),
            "max_length": args.max_length,
            "training_digest": _records_digest(train_records),
        },
    )


def load_router_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return list(value["records"])
    if isinstance(value, list):
        return list(value)
    raise ValueError(f"unsupported router data file: {path}")


def write_router_dataset(out_dir: Path, dataset: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "dataset.json", dataset)
    for split in ("train", "dev", "test"):
        write_json(out_dir / f"{split}.json", {"format": ROUTER_DATA_FORMAT, "split": split, "records": dataset["splits"][split]})
    write_json(out_dir / "metadata.json", dataset["metadata"])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def _normalize_source_record(record: dict[str, Any], source: str, min_text_chars: int) -> dict[str, Any] | None:
    text = str(record.get("text") or record.get("input") or record.get("question") or "").strip()
    label = str(record.get("gold_label") or record.get("label") or "").strip()
    if label not in LABELS or len(text) < min_text_chars:
        return None
    return _router_record(text=text, label=label, source=source, weight=float(record.get("weight", 1.0)), metadata=record.get("metadata", {}))


def _examples_from_trace(trace: dict[str, Any], cfg: RouterContinualConfig) -> list[dict[str, Any]]:
    text = str(trace.get("input", {}).get("text") or trace.get("text") or "").strip()
    if not text:
        return []
    evaluation = trace.get("evaluation", {}) if isinstance(trace.get("evaluation"), dict) else {}
    rating = evaluation.get("user_rating")
    review = str(evaluation.get("review_label") or "").lower()
    accepted = review in {"preferred", "approved", "success"} or (isinstance(rating, (int, float)) and rating >= cfg.min_rating)
    labels = _trace_selected_labels(trace)
    if not labels or not accepted:
        return []
    weight = 1.0
    if review == "preferred" or rating == 5:
        weight = 1.5
    if trace.get("graph", {}).get("fallback_reason"):
        weight *= 0.7
    return [
        _router_record(
            text=text,
            label=label,
            source="continual_trace",
            weight=weight,
            metadata={
                "trace_id": trace.get("trace_id"),
                "review_label": review or None,
                "user_rating": rating,
                "router_scores": trace.get("router", {}).get("scores"),
            },
        )
        for label in labels
    ]


def _trace_selected_labels(trace: dict[str, Any]) -> list[str]:
    graph = trace.get("graph", {}) if isinstance(trace.get("graph"), dict) else {}
    router = trace.get("router", {}) if isinstance(trace.get("router"), dict) else {}
    labels = graph.get("selected_labels") or router.get("primary_labels") or []
    if not labels:
        labels = [item.get("label") for item in graph.get("delegations", []) if isinstance(item, dict)]
    return [str(label) for label in labels if str(label) in LABELS]


def _router_record(*, text: str, label: str, source: str, weight: float, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    record_id = hashlib.sha256(f"{label}\n{text}".encode("utf-8")).hexdigest()[:20]
    return {
        "record_id": record_id,
        "text": text,
        "gold_label": label,
        "weight": weight,
        "source": source,
        "metadata": metadata or {},
    }


def _dedupe_records(records: Any) -> list[dict[str, Any]]:
    deduped = {}
    for record in records:
        if not record:
            continue
        key = record["record_id"]
        if key not in deduped or float(record.get("weight", 1.0)) > float(deduped[key].get("weight", 1.0)):
            deduped[key] = record
    return sorted(deduped.values(), key=lambda item: item["record_id"])


def _cap_per_label(records: list[dict[str, Any]], cap: int, seed: int) -> list[dict[str, Any]]:
    capped = []
    for label in LABELS:
        items = [record for record in records if record["gold_label"] == label]
        capped.extend(_stable_sample(items, cap, seed))
    return sorted(capped, key=lambda item: item["record_id"])


def _stable_sample(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return sorted(records, key=lambda item: hashlib.sha256(f"{seed}:{item['record_id']}".encode("utf-8")).hexdigest())[:limit]


def _split_records(records: list[dict[str, Any]], dev_ratio: float, test_ratio: float, seed: int) -> dict[str, list[dict[str, Any]]]:
    splits = {"train": [], "dev": [], "test": []}
    for record in records:
        bucket = int(hashlib.sha256(f"{seed}:{record['record_id']}".encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        if bucket < test_ratio:
            split = "test"
        elif bucket < test_ratio + dev_ratio:
            split = "dev"
        else:
            split = "train"
        splits[split].append({**record, "split": split})
    return splits


def _dataset_document(kind: str, splits: dict[str, list[dict[str, Any]]], config: Any) -> dict[str, Any]:
    counts = {split: _label_counts(records) for split, records in splits.items()}
    return {
        "format": ROUTER_DATA_FORMAT,
        "kind": kind,
        "metadata": {
            "labels": list(LABELS),
            "config": config.__dict__ if hasattr(config, "__dict__") else {},
            "counts": counts,
            "total": sum(len(records) for records in splits.values()),
            "digest": _records_digest([record for records in splits.values() for record in records]),
        },
        "splits": splits,
    }


def _records_from_document(document: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if "splits" in document:
        return list(document.get("splits", {}).get(split, []))
    if document.get("split") == split:
        return list(document.get("records", []))
    return list(document.get("records", [])) if split == "train" else []


def _label_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for record in records:
        label = record.get("gold_label")
        if label in counts:
            counts[label] += 1
    return counts


def _records_digest(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _tfidf_vector(text: str, idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(token for token in _tokens(text) if token in idf)
    total = sum(counts.values()) or 1
    raw = {token: (count / total) * float(idf[token]) for token, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in raw.values())) or 1.0
    return {token: value / norm for token, value in raw.items()}


def _softmax_scores(raw_scores: dict[str, float]) -> dict[str, float]:
    if not raw_scores:
        return {label: 1.0 / len(LABELS) for label in LABELS}
    top = max(raw_scores.values())
    exps = {label: math.exp(value - top) for label, value in raw_scores.items()}
    total = sum(exps.values()) or 1.0
    return {label: exps.get(label, 0.0) / total for label in LABELS}


def _router_training_deps() -> dict[str, Any]:
    deps = import_training_dependencies()
    try:
        from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding
    except ImportError as exc:
        raise SystemExit(
            "Router training dependencies are missing. Install the training extra:\n"
            "  uv sync --project .\\prd --extra training --system-certs\n"
            f"Missing import: {exc}"
        ) from exc
    deps["AutoModelForSequenceClassification"] = AutoModelForSequenceClassification
    deps["DataCollatorWithPadding"] = DataCollatorWithPadding
    return deps


def _router_tokenized_dataset(records: list[dict[str, Any]], tokenizer: Any, torch: Any, max_length: int) -> Any:
    label2id = {label: index for index, label in enumerate(LABELS)}
    items = []
    for record in records:
        label = str(record.get("gold_label", ""))
        if label not in label2id:
            continue
        encoded = tokenizer(str(record.get("text", "")), truncation=True, max_length=max_length)
        encoded["labels"] = label2id[label]
        items.append(encoded)

    class RouterDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(items)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return items[index]

    return RouterDataset()


def _router_training_args_kwargs(training_args_cls: Any, args: Any, has_dev: bool) -> dict[str, Any]:
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
        "lr_scheduler_type": "cosine",
        "save_total_limit": 2,
        "fp16": args.fp16,
        "bf16": args.bf16,
    }
    signature = __import__("inspect").signature(training_args_cls.__init__)
    kwargs["eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"] = "epoch" if has_dev else "no"
    return kwargs
