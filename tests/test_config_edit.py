import shutil
from pathlib import Path

from tunescope.config import load_all
from tunescope.config_edit import pin_dataset_revisions, set_base_model


ROOT = Path(__file__).resolve().parents[1]


def copy_config_tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "configs", root / "configs")
    shutil.copytree(ROOT / "experiments", root / "experiments")
    (root / "datasets" / "manifests").mkdir(parents=True)
    (root / "reports").mkdir()
    (root / "src" / "tunescope").mkdir(parents=True)
    return root


def test_set_base_model_updates_selected_method(tmp_path, monkeypatch) -> None:
    root = copy_config_tree(tmp_path)
    monkeypatch.setattr("tunescope.config_edit._hf_model_sha", lambda repo_id: "model-sha")

    changes = set_base_model(root, "llm-jp/llm-jp-3-1.8b", methods=["qlora_sft"], pin_revision=True)

    configs = load_all(root)["experiments"]
    assert any(change["status"] == "updated" for change in changes)
    assert configs["Q2"]["base_model"] == "llm-jp/llm-jp-3-1.8b"
    assert configs["Q2"]["base_model_revision"] == "model-sha"
    assert configs["B0"]["base_model"] == "llm-jp/llm-jp-3-1.8b"


def test_pin_dataset_revisions_updates_todo(tmp_path, monkeypatch) -> None:
    root = copy_config_tree(tmp_path)
    monkeypatch.setattr("tunescope.config_edit._hf_dataset_sha", lambda repo_id: "dataset-sha")

    changes = pin_dataset_revisions(root, dataset_ids=["llm_jp_instructions"], force=True)

    configs = load_all(root)["datasets"]
    assert changes == [{"id": "llm_jp_instructions", "status": "pinned", "revision": "dataset-sha"}]
    assert configs["llm_jp_instructions"]["revision"] == "dataset-sha"
