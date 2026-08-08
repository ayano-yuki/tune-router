from __future__ import annotations

import json
from pathlib import Path

from config import LABELS, OSS_SOURCES, ROUTER_BASE_MODEL


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_records(path: Path) -> list[dict]:
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    return payload["records"]


def write_json(path: Path, payload: dict | list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dataset_payload(
    records: list[dict],
    *,
    split: str | None,
    per_label: int,
    seed: int,
    data_origin: str,
    extra_metadata: dict | None = None,
) -> dict:
    metadata = {
        "format": "tune-router-json-v1",
        "data_origin": data_origin,
        "router_base_model": ROUTER_BASE_MODEL,
        "labels": list(LABELS),
        "target_models": {
            label: config["target_model"] for label, config in LABELS.items()
        },
        "oss_sources": OSS_SOURCES if data_origin.startswith("oss_only") else {},
        "split": split,
        "requested_per_label": per_label,
        "seed": seed,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return {
        "metadata": metadata,
        "records": records,
    }


def write_dataset_files(
    out_dir: Path,
    splits: dict[str, list[dict]],
    per_label: int,
    seed: int,
    data_origin: str,
    extra_metadata: dict | None = None,
) -> None:
    records = [record for split in ["train", "dev", "test"] for record in splits[split]]
    write_json(
        out_dir / "dataset.json",
        dataset_payload(
            records,
            split=None,
            per_label=per_label,
            seed=seed,
            data_origin=data_origin,
            extra_metadata=extra_metadata,
        ),
    )
    for split, rows in splits.items():
        write_json(
            out_dir / f"{split}.json",
            dataset_payload(
                rows,
                split=split,
                per_label=per_label,
                seed=seed,
                data_origin=data_origin,
                extra_metadata=extra_metadata,
            ),
        )
