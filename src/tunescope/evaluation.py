from __future__ import annotations

import json
import math
import random
import re
import time
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

from tunescope.artifacts import ensure_dir, resolve_output_dir, run_metadata, write_json, write_yaml
from tunescope.config import ConfigError, get_experiment, load_all
from tunescope.dataset_setup import TODO_REVISION


INTERNAL_PROMPTS = [
    {
        "id": "ja_instruction",
        "prompt": "日本語で、QLoRA と LoRA の違いを3点で簡潔に説明してください。",
        "expect_json": False,
    },
    {
        "id": "json_format",
        "prompt": "次の情報をJSONだけで返してください。名前: TuneScope、目的: FT手法比較",
        "expect_json": True,
    },
    {
        "id": "summary",
        "prompt": "次の文を一文で要約してください。日本語Instruction Tuningでは、同じデータを使って手法差を比較することで、性能とコストの関係を調べられます。",
        "expect_json": False,
    },
]


@dataclass(frozen=True)
class GenerationResult:
    text: str
    new_tokens: int
    elapsed_seconds: float


class Generator:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int,
        generation_config: dict[str, Any],
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            revision=revision,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.generation_config = generation_config

    def format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt

    def generate(self, prompt: str) -> GenerationResult:
        prompt_text = self.format_prompt(prompt)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_length = int(inputs["input_ids"].shape[-1])
        kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": bool(self.generation_config.get("do_sample", False)),
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if kwargs["do_sample"]:
            kwargs["temperature"] = float(self.generation_config.get("temperature", 1.0))
            kwargs["top_p"] = float(self.generation_config.get("top_p", 1.0))

        started = time.perf_counter()
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, **kwargs)
        elapsed = max(time.perf_counter() - started, 1e-9)
        new_ids = output_ids[0][input_length:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        return GenerationResult(text=text, new_tokens=int(new_ids.shape[-1]), elapsed_seconds=elapsed)

    def max_vram_gb(self) -> float | None:
        if self.torch.cuda.is_available():
            return self.torch.cuda.max_memory_allocated() / (1024**3)
        return None


def _reject_placeholder_model(model: str, dry_run: bool, allow_placeholder_model: bool) -> None:
    if model.startswith("TODO_") and not dry_run and not allow_placeholder_model:
        raise ConfigError(
            f"Model is still a placeholder: {model}. "
            "Pin configs/experiments/*.yaml base_model or pass --allow-placeholder-model."
        )


def _model_path(root: Path, experiment: dict[str, Any], output_dir: Path, reuse_result_from: str | None) -> str:
    if reuse_result_from:
        source = get_experiment(reuse_result_from, root)
        source_dir = resolve_output_dir(root, source, None)
        model_dir = source_dir / "model"
        return str(model_dir) if model_dir.exists() else str(source.get("base_model"))

    model_dir = output_dir / "model"
    if model_dir.exists():
        return str(model_dir)

    reused = experiment.get("reuses_result_from")
    if reused:
        source = get_experiment(str(reused), root)
        source_dir = resolve_output_dir(root, source, None)
        source_model_dir = source_dir / "model"
        return str(source_model_dir) if source_model_dir.exists() else str(source.get("base_model"))

    return str(experiment.get("base_model"))


def _model_revision(experiment: dict[str, Any], model_path: str) -> str | None:
    if Path(model_path).exists():
        return None
    revision = experiment.get("base_model_revision")
    return str(revision) if revision else None


def _is_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False


def _refusal(text: str) -> bool:
    lowered = text.lower()
    markers = ["できません", "申し訳", "回答でき", "cannot", "sorry"]
    return any(marker in lowered for marker in markers)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _first_text(record: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = _as_text(record.get(field))
        if value:
            return value
    return ""


def _sample(records: list[dict[str, Any]], sample_count: int | None, seed: int) -> list[dict[str, Any]]:
    if sample_count is None or sample_count >= len(records):
        return records
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    selected = sorted(indices[:sample_count])
    return [records[index] for index in selected]


def _load_hf_dataset(
    dataset_config: dict[str, Any],
    sample_count: int | None,
    seed: int,
    allow_floating_revision: bool,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ConfigError("datasets is required. Run: uv sync --group dev") from exc

    revision = dataset_config.get("revision")
    if revision == TODO_REVISION:
        if not allow_floating_revision:
            raise ConfigError(
                f"Dataset {dataset_config['id']!r} has no pinned revision. "
                "Set configs/datasets/*.yaml revision or pass --allow-floating-revision."
            )
        revision = None

    split = str(dataset_config.get("split", "test"))
    if dataset_config.get("load_mode") == "hf_parquet_api":
        return _load_hf_parquet_api_dataset(dataset_config, split, sample_count, seed)

    kwargs: dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision
    if dataset_config.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True

    subsets = dataset_config.get("subsets")
    subset = dataset_config.get("subset")
    loaded: list[dict[str, Any]] = []
    if isinstance(subsets, list):
        for item in subsets:
            dataset = load_dataset(dataset_config["name"], str(item), **kwargs)
            loaded.extend([{"_subset": str(item), **dict(record)} for record in dataset])
    elif subset:
        dataset = load_dataset(dataset_config["name"], str(subset), **kwargs)
        loaded = [dict(record) for record in dataset]
    else:
        dataset = load_dataset(dataset_config["name"], **kwargs)
        loaded = [dict(record) for record in dataset]
    return _sample(loaded, sample_count, seed)


def _hf_parquet_api_urls(dataset_name: str) -> dict[tuple[str, str], list[str]]:
    api_url = "https://datasets-server.huggingface.co/parquet?" + urlencode({"dataset": dataset_name})
    try:
        with urlopen(api_url, timeout=60) as response:
            payload = json.load(response)
    except OSError as exc:
        raise ConfigError(f"Could not fetch Hugging Face parquet metadata for {dataset_name!r}: {exc}") from exc

    files = payload.get("parquet_files")
    if not isinstance(files, list):
        raise ConfigError(f"Unexpected Hugging Face parquet metadata for {dataset_name!r}.")

    urls: dict[tuple[str, str], list[str]] = {}
    for item in files:
        if not isinstance(item, dict):
            continue
        config = _as_text(item.get("config"))
        split = _as_text(item.get("split"))
        url = _as_text(item.get("url"))
        if config and split and url:
            urls.setdefault((config, split), []).append(url)
    return urls


def _load_hf_parquet_api_dataset(
    dataset_config: dict[str, Any],
    split: str,
    sample_count: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ConfigError("datasets is required. Run: uv sync --group dev") from exc

    subsets = dataset_config.get("subsets")
    subset = dataset_config.get("subset")
    parquet_urls = _hf_parquet_api_urls(str(dataset_config["name"]))
    loaded: list[dict[str, Any]] = []
    if isinstance(subsets, list):
        for item in subsets:
            subset_name = str(item)
            urls = parquet_urls.get((subset_name, split))
            if not urls:
                raise ConfigError(
                    f"No parquet files found for dataset {dataset_config.get('id')!r}, "
                    f"subset {subset_name!r}, split {split!r}."
                )
            dataset = load_dataset("parquet", data_files={split: urls}, split=split)
            loaded.extend([{"_subset": subset_name, **dict(record)} for record in dataset])
    elif subset:
        urls = parquet_urls.get((str(subset), split))
        if not urls:
            raise ConfigError(
                f"No parquet files found for dataset {dataset_config.get('id')!r}, "
                f"subset {subset!r}, split {split!r}."
            )
        dataset = load_dataset("parquet", data_files={split: urls}, split=split)
        loaded = [dict(record) for record in dataset]
    else:
        raise ConfigError(f"Dataset {dataset_config.get('id')!r} uses hf_parquet_api but has no subset list.")
    return _sample(loaded, sample_count, seed)


def _load_task_records(
    root: Path,
    task: dict[str, Any],
    allow_floating_revision: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    dataset_id = task.get("dataset")
    sample_count = limit or task.get("sample_count")
    seed = int(task.get("seed", 42))
    if dataset_id == "internal_prompts":
        records = [dict(item) for item in INTERNAL_PROMPTS]
        return _sample(records, int(sample_count) if sample_count else None, seed)

    configs = load_all(root)
    dataset_config = configs["datasets"].get(str(dataset_id))
    if dataset_config is None:
        raise ConfigError(f"Evaluation task {task.get('id')!r} references unknown dataset {dataset_id!r}.")
    return _load_hf_dataset(
        dataset_config,
        int(sample_count) if isinstance(sample_count, int) else None,
        seed,
        allow_floating_revision,
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _normalize_answer(text: str) -> str:
    return re.sub(r"\s+", "", text).strip().lower()


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(reference) else 0.0


def _tokens(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if re.search(r"\s", stripped):
        return stripped.lower().split()
    return list(stripped.lower())


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def macro_f1(predictions: list[str], references: list[str]) -> float:
    labels = sorted(set(predictions) | set(references))
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(1 for pred, ref in zip(predictions, references, strict=True) if pred == label and ref == label)
        fp = sum(1 for pred, ref in zip(predictions, references, strict=True) if pred == label and ref != label)
        fn = sum(1 for pred, ref in zip(predictions, references, strict=True) if pred != label and ref == label)
        if tp == 0:
            scores.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    pred = _tokens(prediction)
    ref = _tokens(reference)
    return {
        "rouge1": _rouge_n(pred, ref, 1),
        "rouge2": _rouge_n(pred, ref, 2),
        "rougeL": _rouge_l(pred, ref),
    }


def _rouge_n(pred: list[str], ref: list[str], n: int) -> float:
    if len(pred) < n or len(ref) < n:
        return 0.0
    pred_ngrams = Counter(tuple(pred[index : index + n]) for index in range(len(pred) - n + 1))
    ref_ngrams = Counter(tuple(ref[index : index + n]) for index in range(len(ref) - n + 1))
    overlap = sum((pred_ngrams & ref_ngrams).values())
    if overlap == 0:
        return 0.0
    precision = overlap / sum(pred_ngrams.values())
    recall = overlap / sum(ref_ngrams.values())
    return 2 * precision * recall / (precision + recall)


def _rouge_l(pred: list[str], ref: list[str]) -> float:
    if not pred or not ref:
        return 0.0
    lengths = [[0] * (len(ref) + 1) for _ in range(len(pred) + 1)]
    for i, pred_token in enumerate(pred, start=1):
        for j, ref_token in enumerate(ref, start=1):
            if pred_token == ref_token:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])
    lcs = lengths[-1][-1]
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _classification_prompt(record: dict[str, Any]) -> tuple[str, str]:
    label = _as_text(record.get("label_text") or record.get("label"))
    choices = record.get("choices")
    choice_values = []
    if isinstance(choices, list) and choices:
        choice_values = [_as_text(choice) for choice in choices]
    else:
        choice_values = [_as_text(record.get(f"choice{index}")) for index in range(10)]
        choice_values = [choice for choice in choice_values if choice]
    choice_text = ""
    if choice_values:
        choice_text = "選択肢: " + " / ".join(f"{index}: {choice}" for index, choice in enumerate(choice_values))
    fields = [
        ("文", "sentence"),
        ("文1", "sentence1"),
        ("文2", "sentence2"),
        ("前提", "premise"),
        ("仮説", "hypothesis"),
        ("質問", "question"),
        ("本文", "context"),
        ("本文", "paragraph"),
    ]
    body = [f"{label_name}: {_as_text(record.get(field))}" for label_name, field in fields if _as_text(record.get(field))]
    prompt = "\n".join(
        [
            "次の日本語言語理解タスクに答えてください。ラベルまたは選択肢だけを出力してください。",
            choice_text,
            *body,
        ]
    ).strip()
    return prompt, label


def _sts_prompt(record: dict[str, Any]) -> tuple[str, float]:
    sentence1 = _first_text(record, ["sentence1", "sentence_a", "text1"])
    sentence2 = _first_text(record, ["sentence2", "sentence_b", "text2"])
    label = float(record.get("label"))
    prompt = (
        "次の2文の意味的類似度を0から5の数値だけで答えてください。\n"
        f"文1: {sentence1}\n"
        f"文2: {sentence2}\n"
        "類似度:"
    )
    return prompt, label


def _extract_float(response: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", response)
    if not match:
        return None
    return float(match.group(0))


def pearsonr(predictions: list[float], references: list[float]) -> float | None:
    if len(predictions) < 2 or len(predictions) != len(references):
        return None
    pred_mean = sum(predictions) / len(predictions)
    ref_mean = sum(references) / len(references)
    numerator = sum((pred - pred_mean) * (ref - ref_mean) for pred, ref in zip(predictions, references, strict=True))
    pred_var = sum((pred - pred_mean) ** 2 for pred in predictions)
    ref_var = sum((ref - ref_mean) ** 2 for ref in references)
    denominator = math.sqrt(pred_var * ref_var)
    if denominator == 0:
        return None
    return numerator / denominator


def _extract_label(response: str, reference: str) -> str:
    first_line = response.strip().splitlines()[0] if response.strip() else ""
    if reference and reference in first_line:
        return reference
    match = re.search(r"-?\d+(?:\.\d+)?", first_line)
    if match:
        return match.group(0)
    return first_line.strip()


def _qa_prompt(record: dict[str, Any]) -> tuple[str, str]:
    question = _first_text(record, ["question", "query", "instruction"])
    context = _first_text(record, ["context", "paragraph", "article", "text"])
    answers = record.get("answers")
    reference = ""
    if isinstance(answers, dict):
        text = answers.get("text")
        if isinstance(text, list) and text:
            reference = _as_text(text[0])
        else:
            reference = _as_text(text)
    elif isinstance(answers, list) and answers:
        reference = _as_text(answers[0])
    else:
        reference = _first_text(record, ["answer", "output", "summary"])
    prompt = f"次の本文を根拠に質問に短く答えてください。\n\n本文: {context}\n\n質問: {question}\n\n回答:"
    return prompt, reference


def _summary_prompt(record: dict[str, Any]) -> tuple[str, str]:
    article = _first_text(record, ["text", "document", "article", "body", "content"])
    reference = _first_text(record, ["summary", "target", "output", "headline"])
    prompt = f"次の記事を日本語で簡潔に要約してください。\n\n{article}\n\n要約:"
    return prompt, reference


def _elyza_prompt(record: dict[str, Any]) -> str:
    return _first_text(record, ["input", "instruction", "prompt", "question", "text"])


def _run_internal(task: dict[str, Any], generator: Generator, output_dir: Path) -> dict[str, Any]:
    rows = []
    for item in INTERNAL_PROMPTS:
        generated = generator.generate(item["prompt"])
        rows.append(
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "response": generated.text,
                "new_tokens": generated.new_tokens,
                "elapsed_seconds": generated.elapsed_seconds,
                "json_valid": _is_json(generated.text) if item["expect_json"] else None,
                "refusal": _refusal(generated.text),
            }
        )
    _write_jsonl(output_dir / "predictions" / f"{task['id']}.jsonl", rows)
    json_checks = [row["json_valid"] for row in rows if row["json_valid"] is not None]
    elapsed = sum(float(row["elapsed_seconds"]) for row in rows)
    tokens = sum(int(row["new_tokens"]) for row in rows)
    return {
        "status": "completed",
        "sample_count": len(rows),
        "mean_output_tokens": tokens / len(rows),
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "json_valid_rate": sum(1 for value in json_checks if value) / len(json_checks) if json_checks else None,
        "refusal_rate": sum(1 for row in rows if row["refusal"]) / len(rows),
    }


def _run_elyza(records: list[dict[str, Any]], task: dict[str, Any], generator: Generator, output_dir: Path) -> dict[str, Any]:
    rows = []
    for index, record in enumerate(records):
        prompt = _elyza_prompt(record)
        generated = generator.generate(prompt)
        rows.append(
            {
                "id": record.get("id", index),
                "prompt": prompt,
                "response": generated.text,
                "new_tokens": generated.new_tokens,
                "elapsed_seconds": generated.elapsed_seconds,
                "refusal": _refusal(generated.text),
            }
        )
    _write_jsonl(output_dir / "predictions" / f"{task['id']}.jsonl", rows)
    tokens = sum(int(row["new_tokens"]) for row in rows)
    elapsed = sum(float(row["elapsed_seconds"]) for row in rows)
    return {
        "status": "completed_needs_judge",
        "sample_count": len(rows),
        "mean_output_tokens": tokens / len(rows) if rows else None,
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "refusal_rate": sum(1 for row in rows if row["refusal"]) / len(rows) if rows else None,
        "judge_score": None,
    }


def _run_xlsum(records: list[dict[str, Any]], task: dict[str, Any], generator: Generator, output_dir: Path) -> dict[str, Any]:
    rows = []
    rouge1: list[float] = []
    rouge2: list[float] = []
    rouge_l: list[float] = []
    for index, record in enumerate(records):
        prompt, reference = _summary_prompt(record)
        generated = generator.generate(prompt)
        scores = rouge_scores(generated.text, reference)
        rouge1.append(scores["rouge1"])
        rouge2.append(scores["rouge2"])
        rouge_l.append(scores["rougeL"])
        rows.append(
            {
                "id": record.get("id", index),
                "prompt": prompt,
                "reference": reference,
                "response": generated.text,
                "new_tokens": generated.new_tokens,
                "elapsed_seconds": generated.elapsed_seconds,
                **scores,
            }
        )
    _write_jsonl(output_dir / "predictions" / f"{task['id']}.jsonl", rows)
    return {
        "status": "completed",
        "sample_count": len(rows),
        "rouge1": _mean(rouge1),
        "rouge2": _mean(rouge2),
        "rougeL": _mean(rouge_l),
    }


def _run_jglue(records: list[dict[str, Any]], task: dict[str, Any], generator: Generator, output_dir: Path) -> dict[str, Any]:
    rows = []
    cls_predictions: list[str] = []
    cls_references: list[str] = []
    exact_scores: list[float] = []
    f1_scores: list[float] = []
    sts_predictions: list[float] = []
    sts_references: list[float] = []
    sts_abs_errors: list[float] = []
    unsupported = 0
    for index, record in enumerate(records):
        subset = _as_text(record.get("_subset"))
        if subset == "JSTS" or (
            "label" in record and isinstance(record.get("label"), float) and record.get("sentence1") and record.get("sentence2")
        ):
            prompt, reference = _sts_prompt(record)
            generated = generator.generate(prompt)
            prediction = _extract_float(generated.text)
            if prediction is None:
                unsupported += 1
                continue
            sts_predictions.append(prediction)
            sts_references.append(reference)
            sts_abs_errors.append(abs(prediction - reference))
            rows.append(
                {
                    "id": record.get("id", index),
                    "subset": record.get("_subset"),
                    "task_type": "sts",
                    "prompt": prompt,
                    "reference": reference,
                    "prediction": prediction,
                    "response": generated.text,
                    "absolute_error": abs(prediction - reference),
                }
            )
            continue

        is_qa = bool(record.get("answers") or record.get("answer"))
        if is_qa:
            prompt, reference = _qa_prompt(record)
            if not prompt or not reference:
                unsupported += 1
                continue
            generated = generator.generate(prompt)
            exact = exact_match(generated.text, reference)
            f1 = token_f1(generated.text, reference)
            exact_scores.append(exact)
            f1_scores.append(f1)
            rows.append(
                {
                    "id": record.get("id", index),
                    "subset": record.get("_subset"),
                    "task_type": "qa",
                    "prompt": prompt,
                    "reference": reference,
                    "response": generated.text,
                    "exact_match": exact,
                    "f1": f1,
                }
            )
            continue

        if "label" not in record and "label_text" not in record:
            unsupported += 1
            continue
        prompt, reference = _classification_prompt(record)
        generated = generator.generate(prompt)
        prediction = _extract_label(generated.text, reference)
        cls_predictions.append(prediction)
        cls_references.append(reference)
        rows.append(
            {
                "id": record.get("id", index),
                "subset": record.get("_subset"),
                "task_type": "classification",
                "prompt": prompt,
                "reference": reference,
                "prediction": prediction,
                "response": generated.text,
                "correct": prediction == reference,
            }
        )

    _write_jsonl(output_dir / "predictions" / f"{task['id']}.jsonl", rows)
    accuracy = (
        sum(1 for pred, ref in zip(cls_predictions, cls_references, strict=True) if pred == ref) / len(cls_references)
        if cls_references
        else None
    )
    return {
        "status": "completed",
        "sample_count": len(rows),
        "unsupported_count": unsupported,
        "accuracy": accuracy,
        "macro_f1": macro_f1(cls_predictions, cls_references) if cls_references else None,
        "exact_match": _mean(exact_scores),
        "qa_f1": _mean(f1_scores),
        "sts_pearson": pearsonr(sts_predictions, sts_references),
        "sts_mae": _mean(sts_abs_errors),
    }


def _run_task(
    root: Path,
    task: dict[str, Any],
    generator: Generator,
    output_dir: Path,
    allow_floating_revision: bool,
    limit: int | None,
) -> dict[str, Any]:
    task_id = str(task["id"])
    if task.get("dataset") == "internal_prompts":
        return _run_internal(task, generator, output_dir)

    records = _load_task_records(root, task, allow_floating_revision, limit)
    if task_id == "jglue":
        return _run_jglue(records, task, generator, output_dir)
    if task_id == "xlsum_ja":
        return _run_xlsum(records, task, generator, output_dir)
    if task_id == "elyza_tasks_100":
        return _run_elyza(records, task, generator, output_dir)
    raise ConfigError(f"Unknown evaluation task {task_id!r}.")


def _aggregate(task_metrics: dict[str, dict[str, Any]], max_vram_gb: float | None) -> dict[str, Any]:
    flattened: dict[str, Any] = {
        "status": "completed",
        "tasks": task_metrics,
    }
    cost_tasks = [metrics for metrics in task_metrics.values() if metrics.get("tokens_per_second") is not None]
    if cost_tasks:
        flattened["tokens_per_second"] = _mean([float(item["tokens_per_second"]) for item in cost_tasks])
    token_tasks = [metrics for metrics in task_metrics.values() if metrics.get("mean_output_tokens") is not None]
    if token_tasks:
        flattened["mean_output_tokens"] = _mean([float(item["mean_output_tokens"]) for item in token_tasks])
    refusal_tasks = [metrics for metrics in task_metrics.values() if metrics.get("refusal_rate") is not None]
    if refusal_tasks:
        flattened["refusal_rate"] = _mean([float(item["refusal_rate"]) for item in refusal_tasks])
    if max_vram_gb is not None and not math.isnan(max_vram_gb):
        flattened["max_vram_gb"] = max_vram_gb
    return flattened


def evaluate(
    root: Path,
    experiment_id: str,
    output_dir_arg: str | None = None,
    dry_run: bool = False,
    max_new_tokens: int | None = None,
    allow_placeholder_model: bool = False,
    reuse_result_from: str | None = None,
    task_ids: list[str] | None = None,
    limit: int | None = None,
    allow_floating_revision: bool = False,
) -> Path:
    experiment = get_experiment(experiment_id, root)
    output_dir = ensure_dir(resolve_output_dir(root, experiment, output_dir_arg))
    model_path = _model_path(root, experiment, output_dir, reuse_result_from)
    model_revision = _model_revision(experiment, model_path)
    _reject_placeholder_model(model_path, dry_run, allow_placeholder_model)

    configs = load_all(root)
    evaluation_config = configs["evaluation"].get(str(experiment.get("evaluation_config", "default")))
    if evaluation_config is None:
        raise ConfigError(f"Unknown evaluation config {experiment.get('evaluation_config')!r}.")
    tasks = list(evaluation_config.get("tasks", []))
    if task_ids:
        selected = set(task_ids)
        tasks = [task for task in tasks if task.get("id") in selected]
    if not tasks:
        raise ConfigError("No evaluation tasks selected.")

    generation = dict(evaluation_config.get("generation", {}))
    generation_limit = max_new_tokens or int(generation.get("max_new_tokens", 256))

    run = run_metadata(root, experiment, "evaluate", output_dir)
    run["resolved_model"] = model_path
    run["resolved_model_revision"] = model_revision
    run["evaluation_config"] = evaluation_config
    run["selected_tasks"] = [task["id"] for task in tasks]
    if reuse_result_from:
        run["reuse_result_from"] = reuse_result_from
    write_yaml(output_dir / "run.yaml", run)

    if dry_run:
        metrics = {
            "status": "dry_run",
            "experiment_id": experiment_id,
            "planned_tasks": [task["id"] for task in tasks],
            "max_new_tokens": generation_limit,
        }
        write_json(output_dir / "eval_metrics.json", metrics)
        return output_dir

    generator = Generator(model_path, generation_limit, generation, revision=model_revision)
    task_metrics: dict[str, dict[str, Any]] = {}
    for task in tasks:
        metrics = _run_task(root, task, generator, output_dir, allow_floating_revision, limit)
        task_metrics[str(task["id"])] = metrics
        write_json(output_dir / "metrics" / f"{task['id']}.json", metrics)

    aggregate = _aggregate(task_metrics, generator.max_vram_gb())
    aggregate["experiment_id"] = experiment_id
    write_json(output_dir / "eval_metrics.json", aggregate)
    write_json(
        output_dir / "cost.json",
        {
            "tokens_per_second": aggregate.get("tokens_per_second"),
            "mean_output_tokens": aggregate.get("mean_output_tokens"),
            "max_vram_gb": aggregate.get("max_vram_gb"),
        },
    )
    return output_dir


def _read_score_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ConfigError(f"{path}:{line_number} must contain a JSON object.")
            rows.append(value)
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must contain a JSON object.")
    return value


def apply_elyza_judge_scores(
    root: Path,
    experiment_id: str,
    scores_path_arg: str,
    output_dir_arg: str | None = None,
) -> Path:
    experiment = get_experiment(experiment_id, root)
    output_dir = ensure_dir(resolve_output_dir(root, experiment, output_dir_arg))
    scores_path = Path(scores_path_arg)
    if not scores_path.is_absolute():
        scores_path = root / scores_path
    if not scores_path.exists():
        raise ConfigError(f"Judge score file not found: {scores_path}")

    score_rows = _read_score_rows(scores_path)
    scores_by_id: dict[str, dict[str, Any]] = {}
    ordered_scores: list[dict[str, Any]] = []
    for row in score_rows:
        raw_score = row.get("score") or row.get("judge_score") or row.get("elyza_score")
        if raw_score is None:
            raise ConfigError(f"Judge row missing score: {row}")
        try:
            score = float(raw_score)
        except ValueError as exc:
            raise ConfigError(f"Judge score must be numeric: {raw_score!r}") from exc
        normalized = {
            "id": _as_text(row.get("id")),
            "score": score,
            "comment": _as_text(row.get("comment") or row.get("reason") or row.get("rationale")),
        }
        ordered_scores.append(normalized)
        if normalized["id"]:
            scores_by_id[normalized["id"]] = normalized

    predictions_path = output_dir / "predictions" / "elyza_tasks_100.jsonl"
    predictions = []
    if predictions_path.exists():
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    predictions.append(json.loads(line))

    merged = []
    used_scores: list[float] = []
    for index, prediction in enumerate(predictions):
        prediction_id = _as_text(prediction.get("id"))
        score_row = scores_by_id.get(prediction_id)
        if score_row is None and index < len(ordered_scores):
            score_row = ordered_scores[index]
        if score_row is None:
            merged.append(prediction)
            continue
        updated = dict(prediction)
        updated["judge_score"] = score_row["score"]
        if score_row["comment"]:
            updated["judge_comment"] = score_row["comment"]
        used_scores.append(float(score_row["score"]))
        merged.append(updated)

    if predictions:
        _write_jsonl(predictions_path, merged)
    else:
        used_scores = [float(row["score"]) for row in ordered_scores]

    judge_score = _mean(used_scores)
    metric_path = output_dir / "metrics" / "elyza_tasks_100.json"
    metrics = _load_json(metric_path)
    metrics.update(
        {
            "status": "completed_judged",
            "sample_count": len(used_scores),
            "judge_score": judge_score,
            "judge_scores_path": str(scores_path),
        }
    )
    write_json(metric_path, metrics)

    aggregate_path = output_dir / "eval_metrics.json"
    aggregate = _load_json(aggregate_path)
    tasks = aggregate.setdefault("tasks", {})
    if isinstance(tasks, dict):
        tasks["elyza_tasks_100"] = metrics
    aggregate["elyza_judge_score"] = judge_score
    write_json(aggregate_path, aggregate)
    return output_dir
