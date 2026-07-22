from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from tunescope.artifacts import ensure_dir, resolve_output_dir
from tunescope.config import ConfigError, get_experiment


def checkpoint_dirs(output_dir: Path) -> list[Path]:
    model_dir = output_dir / "model"
    if not model_dir.exists():
        return []
    checkpoints = [path for path in model_dir.glob("checkpoint-*") if path.is_dir()]
    return sorted(checkpoints, key=_checkpoint_step)


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = checkpoint_dirs(output_dir)
    return checkpoints[-1] if checkpoints else None


def prune_checkpoints(root: Path, experiment_id: str, output_dir_arg: str | None, keep_last: int) -> dict[str, Any]:
    if keep_last < 1:
        raise ConfigError("--keep-last must be at least 1.")
    experiment = get_experiment(experiment_id, root)
    output_dir = ensure_dir(resolve_output_dir(root, experiment, output_dir_arg))
    checkpoints = checkpoint_dirs(output_dir)
    keep = set(checkpoints[-keep_last:])
    removed: list[str] = []
    kept: list[str] = []

    for path in checkpoints:
        if path in keep:
            kept.append(str(path))
            continue
        shutil.rmtree(path)
        removed.append(str(path))

    return {
        "experiment_id": experiment_id,
        "output_dir": str(output_dir),
        "keep_last": keep_last,
        "kept": kept,
        "removed": removed,
    }

