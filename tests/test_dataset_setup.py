import json

from tunescope.dataset_setup import normalize_record, prepare_dataset, sample_records, write_jsonl


def test_normalize_messages_combines_instruction_and_input() -> None:
    config = {
        "id": "example_sft",
        "normalization": {
            "target_format": "messages",
            "fields": {"user": "instruction", "assistant": "output"},
        },
    }
    record = {"instruction": "次を要約してください。", "input": "長い本文", "output": "短い要約"}

    normalized = normalize_record(record, config)

    assert normalized == {
        "messages": [
            {"role": "user", "content": "次を要約してください。\n\n長い本文"},
            {"role": "assistant", "content": "短い要約"},
        ]
    }


def test_normalize_preference_record() -> None:
    config = {
        "id": "example_dpo",
        "normalization": {
            "target_format": "preference",
            "fields": {"prompt": "prompt", "chosen": "chosen", "rejected": "rejected"},
        },
    }
    record = {"prompt": "質問", "chosen": "よい回答", "rejected": "悪い回答"}

    assert normalize_record(record, config) == record


def test_normalize_preference_record_from_conversations() -> None:
    config = {
        "id": "hh_rlhf_12k_ja",
        "normalization": {
            "target_format": "preference",
            "fields": {"prompt": "conversations", "chosen": "chosen", "rejected": "rejected"},
        },
    }
    record = {
        "conversations": [{"from": "human", "value": "説明してください。"}],
        "chosen": "よい回答",
        "rejected": "悪い回答",
        "source": "helpful-base",
    }

    assert normalize_record(record, config) == {
        "prompt": "説明してください。",
        "chosen": "よい回答",
        "rejected": "悪い回答",
    }


def test_normalize_preference_record_from_multi_turn_conversations() -> None:
    config = {
        "id": "hh_rlhf_12k_ja",
        "normalization": {
            "target_format": "preference",
            "fields": {"prompt": "conversations", "chosen": "chosen", "rejected": "rejected"},
        },
    }
    record = {
        "conversations": [
            {"from": "human", "value": "最初の質問"},
            {"from": "gpt", "value": "途中の回答"},
            {"from": "human", "value": "追加の質問"},
        ],
        "chosen": "よい回答",
        "rejected": "悪い回答",
    }

    assert normalize_record(record, config)["prompt"] == (
        "user: 最初の質問\nassistant: 途中の回答\nuser: 追加の質問"
    )


def test_prepare_dataset_skips_invalid_preference_records(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "configs" / "datasets"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "hh.yaml").write_text(
        """
id: hh
name: example/hh
revision: abc
split: train
normalization:
  target_format: preference
  skip_invalid: true
  fields:
    prompt: conversations
    chosen: chosen
    rejected: rejected
""".lstrip(),
        encoding="utf-8",
    )

    def fake_load_hf_records(dataset_config, allow_floating_revision):
        return [
            {
                "conversations": [{"from": "human", "value": "質問"}],
                "chosen": "よい回答",
                "rejected": "悪い回答",
            },
            {
                "conversations": [{"from": "human", "value": "空の回答"}],
                "chosen": "",
                "rejected": "悪い回答",
            },
        ]

    monkeypatch.setattr("tunescope.dataset_setup._load_hf_records", fake_load_hf_records)

    result = prepare_dataset(tmp_path, "hh", "all", seed=42)

    assert result.record_count == 1
    assert result.invalid_record_count == 1
    lines = result.output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"prompt": "質問", "chosen": "よい回答", "rejected": "悪い回答"}
    manifest = result.manifest_path.read_text(encoding="utf-8")
    assert "skipped_invalid_records: 1" in manifest


def test_sample_records_is_deterministic() -> None:
    records = [{"id": index} for index in range(10)]

    assert sample_records(records, 3, seed=42) == sample_records(records, 3, seed=42)
    assert len(sample_records(records, 3, seed=42)) == 3


def test_write_jsonl_uses_utf8_json_lines(tmp_path) -> None:
    path = tmp_path / "out.jsonl"

    count = write_jsonl(path, [{"text": "日本語"}, {"text": "ok"}])

    assert count == 2
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"text": "日本語"}
