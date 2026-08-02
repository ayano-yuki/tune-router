from __future__ import annotations

import json
import random
import re

from config import LABELS, OSS_SOURCES
from utils import enable_system_cert_store, stable_int


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

    for field in source_config["prompt_fields"]:
        text = stringify_field(row.get(field))
        if text:
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
    records: list[dict] = []
    label_names = list(LABELS)
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
        if count < per_label:
            raise RuntimeError(
                f"{source_config['dataset']} produced {count} usable records for {label}; "
                f"requested {per_label}. Increase --max-source-scan or choose another source."
            )
    random.Random(seed).shuffle(records)
    return records
