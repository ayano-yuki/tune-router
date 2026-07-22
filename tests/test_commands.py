import shutil
from pathlib import Path

from tunescope.cli import main


ROOT = Path(__file__).resolve().parents[1]


def copy_runtime_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "configs", root / "configs")
    shutil.copytree(ROOT / "experiments", root / "experiments")
    (root / "datasets" / "manifests").mkdir(parents=True)
    (root / "reports").mkdir()
    (root / "src" / "tunescope").mkdir(parents=True)
    return root


def test_training_evaluation_and_report_dry_run(tmp_path) -> None:
    root = copy_runtime_root(tmp_path)
    q2_dir = tmp_path / "Q2"
    b0_dir = tmp_path / "B0"
    report_path = tmp_path / "report.md"

    assert main(["--root", str(root), "train-sft", "--experiment-id", "Q2", "--output-dir", str(q2_dir), "--dry-run"]) == 0
    assert (q2_dir / "run.yaml").exists()
    assert (q2_dir / "train_metrics.json").exists()
    assert (root / "experiments" / "manifests" / "artifacts.yaml").exists()

    assert main(["--root", str(root), "evaluate", "--experiment-id", "B0", "--output-dir", str(b0_dir), "--dry-run"]) == 0
    assert (b0_dir / "eval_metrics.json").exists()

    assert (
        main(
            [
                "--root",
                str(root),
                "report",
                "--experiment-id",
                "B0",
                "--experiment-id",
                "Q2",
                "--results-dir",
                str(tmp_path),
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    assert report_path.exists()
    assert report_path.with_suffix(".json").exists()

    assert main(["--root", str(root), "list-artifacts"]) == 0


def test_dpo_dry_run(tmp_path) -> None:
    root = copy_runtime_root(tmp_path)
    d1_dir = tmp_path / "D1"

    assert main(["--root", str(root), "train-dpo", "--experiment-id", "D1", "--output-dir", str(d1_dir), "--dry-run"]) == 0
    assert (d1_dir / "train_metrics.json").exists()
