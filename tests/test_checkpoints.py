from pathlib import Path

from tunescope.checkpoints import latest_checkpoint, prune_checkpoints


ROOT = Path(__file__).resolve().parents[1]


def test_prune_checkpoints_keeps_latest(tmp_path) -> None:
    model_dir = tmp_path / "Q2" / "model"
    for step in [1, 10, 2]:
        checkpoint = model_dir / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "marker.txt").write_text(str(step), encoding="utf-8")

    result = prune_checkpoints(ROOT, "Q2", str(tmp_path / "Q2"), keep_last=2)

    assert len(result["removed"]) == 1
    assert not (model_dir / "checkpoint-1").exists()
    assert (model_dir / "checkpoint-2").exists()
    assert (model_dir / "checkpoint-10").exists()
    assert latest_checkpoint(tmp_path / "Q2") == model_dir / "checkpoint-10"

