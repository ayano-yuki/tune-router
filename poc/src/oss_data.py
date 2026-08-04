from __future__ import annotations

import json
import random
import re

from config import LABELS, OSS_SOURCES
from utils import enable_system_cert_store, stable_int


IAC_GENERIC_INSTRUCTION_MARKERS = [
    "you are a terraform security expert",
    "analyze the following terraform configuration",
    "provide remediation guidance",
]

IAC_SOURCE_HINTS = ["terraform", "iac", "kubernetes", "k8s", "cloudformation"]
IAC_CODE_FIELDS = {"terraform_code", "code", "input"}


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


def extract_user_text(row: dict, source_config: dict, label: str) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                text = stringify_field(message.get("content"))
                if text:
                    return text

    source_is_iac = is_iac_source(source_config)
    for field in preferred_prompt_fields(source_config, label):
        text = stringify_field(row.get(field))
        if text:
            if (
                label == "Coding"
                and source_is_iac
                and field in {"instruction", "prompt", "question"}
                and is_generic_iac_instruction(text)
            ):
                continue
            if label == "Coding" and source_is_iac and field in IAC_CODE_FIELDS:
                return f"Review this Terraform/IaC configuration and explain the operational risk:\n{text}"
            if label == "Security" and field in {"text", "input"} and "log" not in text.lower():
                return f"Analyze this security event or alert:\n{text}"
            return text

    ignored = {"answer", "response", "completion", "output", "labels", "label"}
    fallback_parts = [
        stringify_field(value)
        for key, value in row.items()
        if key not in ignored and stringify_field(value)
    ]
    return "\n".join(fallback_parts[:3]).strip()


def normalize_extracted_text(text: str, label: str) -> str:
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
    load_dataset = import_dataset_deps()
    dataset = load_dataset(
        source_config["dataset"],
        split=source_config["split"],
        streaming=streaming,
        cache_dir=cache_dir,
    )
    if streaming:
        return dataset.shuffle(seed=seed, buffer_size=10_000)
    rows = list(dataset)
    random.Random(seed).shuffle(rows)
    return rows


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
            source_config = OSS_SOURCES[label]
            raise RuntimeError(
                f"{source_config['dataset']} produced {count} usable records for {label}; "
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
        source_config = OSS_SOURCES[label]
        count = 0
        seen = 0
        for row in iter_source_rows(
            source_config,
            seed=seed + label_names.index(label),
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
            text = normalize_extracted_text(extract_user_text(row, source_config, label), label)
            if not is_usable_for_label(text, label):
                continue
            dedupe_key = text.lower()
            if dedupe_key in seen_texts_by_label[label]:
                continue
            seen_texts_by_label[label].add(dedupe_key)
            source_record_id = stringify_field(
                row.get("id")
                or row.get("task_id")
                or row.get("idx")
                or row.get("index")
                or stable_int(json.dumps(row, ensure_ascii=False, sort_keys=True))
            )
            count += 1
            records.append(
                {
                    "question_id": f"oss-{label}-{count:06d}",
                    "text": text[:4000],
                    "gold_label": label,
                    "label_id": label_names.index(label),
                    "target_model": LABELS[label]["target_model"],
                    "tags": [label, "oss"],
                    "importance": "normal",
                    "source": source_config["dataset"],
                    "source_url": source_config["url"],
                    "source_license": source_config["license"],
                    "source_record_id": source_record_id,
                    "template_id": source_config["dataset"],
                    "split_group": f"{source_config['dataset']}:{source_record_id}",
                    "split": "",
                    "rubric": {
                        "minimum_quality": 0.65 if label == "General" else 0.75,
                        "requires_human_review": False,
                    },
                }
            )
        counts[label] = count
    random.Random(seed).shuffle(records)
    return records, counts
