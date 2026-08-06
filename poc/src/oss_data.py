from __future__ import annotations

import json
import random
import re
import time
from html import unescape

from config import LABELS, OSS_SOURCES
from utils import enable_system_cert_store, stable_int


IAC_GENERIC_INSTRUCTION_MARKERS = [
    "you are a terraform security expert",
    "analyze the following terraform configuration",
    "provide remediation guidance",
]

IAC_SOURCE_HINTS = ["terraform", "iac", "kubernetes", "k8s", "cloudformation"]
IAC_CODE_FIELDS = {"terraform_code", "code", "input"}

QUESTION_INTENT_MARKERS = [
    "?",
    "？",
    "教えて",
    "説明して",
    "整理して",
    "分析して",
    "確認して",
    "答えて",
    "ください",
    "切り分け",
    "作って",
    "実装",
    "how ",
    "what ",
    "why ",
    "which ",
    "explain",
    "analyze",
    "review",
    "write",
    "implement",
]

GENERIC_QUESTION_TEMPLATE = "以下の内容について、ユーザーの依頼として回答してください:\n{text}"


def is_generic_iac_instruction(text: str) -> bool:
    lowered = text.lower()
    return all(marker in lowered for marker in IAC_GENERIC_INSTRUCTION_MARKERS)


def is_iac_source(source_config: dict) -> bool:
    source_blob = " ".join(
        stringify_field(source_config.get(field))
        for field in ("dataset", "url", "template_id")
    ).lower()
    prompt_fields = set(source_config.get("prompt_fields", []))
    return any(hint in source_blob for hint in IAC_SOURCE_HINTS) or bool(
        prompt_fields & {"terraform_code", "kubernetes_yaml"}
    )


def source_configs_for_label(label: str) -> list[dict]:
    sources = OSS_SOURCES[label]
    if isinstance(sources, list):
        return sources
    return [sources]


def source_name(source_config: dict) -> str:
    return source_config.get("name") or source_config["dataset"]


def preferred_prompt_fields(source_config: dict, label: str) -> list[str]:
    if label == "Coding" and is_iac_source(source_config):
        return [
            "terraform_code",
            "kubernetes_yaml",
            "code",
            "input",
            "text",
            "instruction",
            "prompt",
            "question",
        ]
    return source_config["prompt_fields"]


def import_dataset_deps():
    enable_system_cert_store()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "OSS dataset loading requires the datasets package. Run with uv:\n"
            "  uv run --project .\\poc python .\\poc\\src\\cli.py prepare-data\n"
            f"Missing import: {exc}"
        ) from exc
    return load_dataset


def stringify_field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(part.strip() for part in parts if part and part.strip())
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return unescape(text)


def normalize_raw_text(text: str) -> str:
    text = strip_html(text)
    return re.sub(r"\s+", " ", text).strip()


def extract_stackexchange_question(text: str) -> str:
    text = re.sub(r"^user\d+:\s*", "", text.strip())
    text = re.split(r"\n\s*\nuser\d+:", text, maxsplit=1)[0]
    return text.strip()


def is_question_like(text: str) -> bool:
    lowered = f" {text.lower()} "
    return any(marker in lowered for marker in QUESTION_INTENT_MARKERS)


def format_as_question(text: str, label: str, source_config: dict) -> str:
    text = normalize_raw_text(text)
    template = source_config.get("question_template")
    if template:
        return template.format(text=text).strip()
    if is_question_like(text):
        return text
    return GENERIC_QUESTION_TEMPLATE.format(text=text).strip()


def metadata_matches_source(row: dict, source_config: dict) -> bool:
    terms = source_config.get("metadata_terms")
    if not terms:
        return True
    metadata_blob = " ".join(
        stringify_field(row.get(field))
        for field in ("metadata", "url", "source", "site", "tags")
    ).lower()
    return any(term.lower() in metadata_blob for term in terms)


def extract_user_text(row: dict, source_config: dict, label: str) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                text = stringify_field(message.get("content"))
                if text:
                    return text

    combine_fields = source_config.get("combine_fields")
    if combine_fields:
        parts = [normalize_raw_text(stringify_field(row.get(field))) for field in combine_fields]
        text = "\n".join(part for part in parts if part).strip()
        if text:
            return format_as_question(text, label, source_config)

    source_is_iac = is_iac_source(source_config)
    for field in preferred_prompt_fields(source_config, label):
        text = stringify_field(row.get(field))
        if text:
            if source_config.get("extract_stackexchange_question"):
                text = extract_stackexchange_question(text)
            text = normalize_raw_text(text)
            if (
                label == "Coding"
                and source_is_iac
                and field in {"instruction", "prompt", "question"}
                and is_generic_iac_instruction(text)
            ):
                continue
            if label == "Coding" and source_is_iac and field in IAC_CODE_FIELDS:
                return format_as_question(
                    f"Review this configuration and explain the operational risk:\n{text}",
                    label,
                    source_config,
                )
            if label == "Security" and field in {"text", "input"} and "log" not in text.lower():
                return format_as_question(f"Analyze this event or alert:\n{text}", label, source_config)
            return format_as_question(text, label, source_config)

    ignored = {"answer", "response", "completion", "output", "labels", "label"}
    fallback_parts = [
        stringify_field(value)
        for key, value in row.items()
        if key not in ignored and stringify_field(value)
    ]
    return format_as_question("\n".join(fallback_parts[:3]).strip(), label, source_config)


def normalize_extracted_text(text: str, label: str) -> str:
    text = normalize_raw_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    if label == "Security":
        marker = "principle of defense only."
        marker_index = text.find(marker)
        if marker_index >= 0:
            text = text[marker_index + len(marker) :].strip()
        question_index = text.find("?")
        if question_index >= 0:
            text = text[: question_index + 1].strip()
    return text


def is_usable_for_label(text: str, label: str) -> bool:
    lowered = text.lower()
    if len(text) < 12:
        return False
    if label == "Coding" and is_generic_iac_instruction(text):
        return False
    if label == "Storage":
        positive_terms = [
            "ceph",
            "zfs",
            "raid",
            "nas",
            "san",
            "iscsi",
            "nfs",
            "smb",
            "storage",
            "disk",
            "volume",
            "snapshot",
            "backup",
        ]
        return any(term in lowered for term in positive_terms)
    if label == "Network":
        positive_terms = [
            "bgp",
            "ospf",
            "vlan",
            "dns",
            "dhcp",
            "mtu",
            "vpn",
            "l2",
            "l3",
            "ping",
            "network",
            "routing",
            "packet",
            "switch",
            "firewall",
        ]
        return any(term in lowered for term in positive_terms)
    if label == "Coding":
        negative_intents = [
            "summarize",
            "classify the following",
            "gather information",
            "write a review",
            "translate",
            "sentiment",
            "one sentence",
        ]
        positive_terms = [
            "code",
            "script",
            "function",
            "program",
            "algorithm",
            "debug",
            "sql",
            "api",
            "terraform",
            "kubernetes",
            "dockerfile",
            "docker compose",
            "yaml",
            "ansible",
            "cloudformation",
            "helm",
            "python",
            "javascript",
            "typescript",
            "java",
            "rust",
            "go ",
            "c++",
            "class",
            "data structure",
        ]
        if any(term in lowered for term in negative_intents):
            return False
        return any(term in lowered for term in positive_terms)
    if label == "Security":
        positive_terms = [
            "security",
            "log",
            "cve",
            "vulnerability",
            "threat",
            "incident",
            "attack",
            "malware",
            "metasploit",
            "credential",
            "mitre",
            "nist",
            "detection",
            "countermeasure",
        ]
        return any(term in lowered for term in positive_terms)
    if label == "Database":
        positive_terms = [
            "postgres",
            "postgresql",
            "mysql",
            "oracle",
            "redis",
            "mongodb",
            "database",
            "sql",
            "query",
            "index",
            "replication",
            "transaction",
            "deadlock",
        ]
        return any(term in lowered for term in positive_terms)
    return True


def iter_source_rows(source_config: dict, *, seed: int, streaming: bool, cache_dir: str | None):
    if source_config.get("loader") == "stackexchange_api":
        return iter_stackexchange_api_rows(source_config)

    load_dataset = import_dataset_deps()
    dataset_name = source_config.get("loader") or source_config["dataset"]
    dataset_config = source_config.get("config")
    load_args = [dataset_name]
    if dataset_config:
        load_args.append(dataset_config)
    load_kwargs = {
        "split": source_config["split"],
        "streaming": streaming,
        "cache_dir": cache_dir,
    }
    if source_config.get("data_files"):
        load_kwargs["data_files"] = {"train": source_config["data_files"]}
    dataset = load_dataset(
        *load_args,
        **load_kwargs,
    )
    if streaming and source_config.get("shuffle", source_config.get("loader") != "parquet"):
        return dataset.shuffle(seed=seed, buffer_size=10_000)
    rows = list(dataset)
    random.Random(seed).shuffle(rows)
    return rows


def stackexchange_request(params: dict) -> dict:
    enable_system_cert_store()
    try:
        import requests
    except ImportError as exc:
        raise SystemExit(
            "Stack Exchange API loading requires requests, which is normally installed with datasets."
        ) from exc
    response = requests.get(
        "https://api.stackexchange.com/2.3/questions",
        params=params,
        headers={"User-Agent": "tune-router-poc/1.0"},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def iter_stackexchange_api_rows(source_config: dict):
    for site_config in source_config.get("stackexchange_sites", []):
        site = site_config["site"]
        for tag in site_config.get("tags", []):
            for page in range(1, int(source_config.get("max_pages_per_tag", 4)) + 1):
                payload = stackexchange_request(
                    {
                        "site": site,
                        "tagged": tag,
                        "page": page,
                        "pagesize": int(source_config.get("pagesize", 100)),
                        "order": "desc",
                        "sort": source_config.get("sort", "activity"),
                        "filter": "withbody",
                    }
                )
                for item in payload.get("items", []):
                    row = dict(item)
                    row["site"] = site
                    row["source_tag"] = tag
                    yield row
                backoff = payload.get("backoff")
                if backoff:
                    time.sleep(float(backoff))
                if not payload.get("has_more"):
                    break


def build_oss_dataset(
    per_label: int,
    seed: int,
    *,
    streaming: bool,
    cache_dir: str | None,
    max_source_scan: int,
) -> list[dict]:
    records, counts = build_partial_oss_dataset(
        per_label,
        seed,
        streaming=streaming,
        cache_dir=cache_dir,
        max_source_scan=max_source_scan,
    )
    for label, count in counts.items():
        if count < per_label:
            source_names = ", ".join(source_name(source) for source in source_configs_for_label(label))
            raise RuntimeError(
                f"{source_names} produced {count} usable records for {label}; "
                f"requested {per_label}. Increase --max-source-scan or choose another source."
            )
    return records


def build_partial_oss_dataset(
    per_label: int,
    seed: int,
    *,
    streaming: bool,
    cache_dir: str | None,
    max_source_scan: int,
) -> tuple[list[dict], dict[str, int]]:
    records: list[dict] = []
    label_names = list(LABELS)
    counts: dict[str, int] = {}
    seen_texts_by_label: dict[str, set[str]] = {label: set() for label in label_names}
    for label in label_names:
        count = 0
        for source_index, source_config in enumerate(source_configs_for_label(label)):
            if count >= per_label:
                break
            seen = 0
            for row in iter_source_rows(
                source_config,
                seed=seed + label_names.index(label) + (source_index * 10_000),
                streaming=streaming,
                cache_dir=cache_dir,
            ):
                if count >= per_label:
                    break
                seen += 1
                if seen > max_source_scan:
                    break
                if not isinstance(row, dict):
                    row = dict(row)
                if not metadata_matches_source(row, source_config):
                    continue
                text = normalize_extracted_text(extract_user_text(row, source_config, label), label)
                if not is_usable_for_label(text, label):
                    continue
                dedupe_key = text.lower()
                if dedupe_key in seen_texts_by_label[label]:
                    continue
                seen_texts_by_label[label].add(dedupe_key)
                source_record_id = stringify_field(
                    row.get("id")
                    or row.get("question_id")
                    or row.get("qid")
                    or row.get("task_id")
                    or row.get("raw_index")
                    or row.get("idx")
                    or row.get("index")
                    or stable_int(json.dumps(row, ensure_ascii=False, sort_keys=True))
                )
                source_label = source_name(source_config)
                count += 1
                records.append(
                    {
                        "question_id": f"oss-{label}-{count:06d}",
                        "text": text[:4000],
                        "gold_label": label,
                        "label_id": label_names.index(label),
                        "target_model": LABELS[label]["target_model"],
                        "tags": [label, "oss", source_label],
                        "importance": "normal",
                        "source": source_config["dataset"],
                        "source_name": source_label,
                        "source_config": source_config.get("config", ""),
                        "source_url": source_config["url"],
                        "source_license": source_config["license"],
                        "source_record_id": source_record_id,
                        "source_record_url": stringify_field(row.get("link")),
                        "source_original_type": "question_or_prompt",
                        "question_format": "question_or_request",
                        "template_id": source_label,
                        "split_group": f"{source_label}:{source_record_id}",
                        "split": "",
                        "rubric": {
                            "minimum_quality": 0.65 if label == "General" else 0.75,
                            "requires_human_review": False,
                            "question_like": is_question_like(text),
                        },
                    }
                )
        counts[label] = count
    random.Random(seed).shuffle(records)
    return records, counts
