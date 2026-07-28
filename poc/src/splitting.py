from __future__ import annotations

from utils import stable_int


def split_name(record: dict) -> str:
    bucket = stable_int(f"{record['gold_label']}:{record['split_group']}") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "dev"
    return "test"


def split_dataset(records: list[dict]) -> dict[str, list[dict]]:
    splits = {"train": [], "dev": [], "test": []}
    for record in records:
        row = dict(record)
        row["split"] = split_name(row)
        splits[row["split"]].append(row)
    return splits
