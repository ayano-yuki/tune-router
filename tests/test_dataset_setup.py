import json

from tunescope.dataset_setup import normalize_record, sample_records, write_jsonl


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

