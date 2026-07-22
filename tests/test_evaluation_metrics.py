import json

from tunescope.evaluation import _load_hf_dataset, exact_match, macro_f1, pearsonr, rouge_scores, token_f1


def test_exact_match_ignores_spaces() -> None:
    assert exact_match("東京 都", "東京都") == 1.0


def test_token_f1_for_japanese_characters() -> None:
    assert token_f1("東京都", "京都") > 0.0


def test_macro_f1_handles_multiple_labels() -> None:
    score = macro_f1(["positive", "negative", "positive"], ["positive", "positive", "positive"])
    assert 0.0 < score < 1.0


def test_rouge_scores_are_bounded() -> None:
    scores = rouge_scores("今日は晴れです", "今日は雨です")
    assert set(scores) == {"rouge1", "rouge2", "rougeL"}
    assert all(0.0 <= value <= 1.0 for value in scores.values())


def test_pearsonr_for_similarity_scores() -> None:
    assert pearsonr([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_load_hf_dataset_can_use_parquet_api(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return json.dumps(
                {
                    "parquet_files": [
                        {
                            "config": "MARC-ja",
                            "split": "validation",
                            "url": "https://huggingface.co/datasets/shunk031/JGLUE/resolve/refs%2Fconvert%2Fparquet/MARC-ja/jglue-validation.parquet",
                        }
                    ]
                }
            ).encode()

    def fake_urlopen(url, timeout):
        assert url == "https://datasets-server.huggingface.co/parquet?dataset=shunk031/JGLUE"
        assert timeout == 60
        return FakeResponse()

    def fake_load_dataset(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        return [{"label": 1, "sentence": "テスト"}]

    monkeypatch.setattr("tunescope.evaluation.urlopen", fake_urlopen)
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    records = _load_hf_dataset(
        {
            "id": "jglue",
            "name": "shunk031/JGLUE",
            "split": "validation",
            "load_mode": "hf_parquet_api",
            "subsets": ["MARC-ja"],
        },
        sample_count=None,
        seed=42,
        allow_floating_revision=False,
    )

    assert records == [{"_subset": "MARC-ja", "label": 1, "sentence": "テスト"}]
    assert calls[0][0] == "parquet"
    assert calls[0][2]["split"] == "validation"
    assert calls[0][2]["data_files"] == {
        "validation": [
            "https://huggingface.co/datasets/shunk031/JGLUE/resolve/refs%2Fconvert%2Fparquet/MARC-ja/jglue-validation.parquet"
        ]
    }


def test_load_hf_dataset_falls_back_to_repo_parquet_files(monkeypatch) -> None:
    calls = []

    def fake_urlopen(url, timeout):
        raise OSError("metadata API unavailable")

    def fake_list_repo_files(repo_id, repo_type, revision):
        assert repo_id == "shunk031/JGLUE"
        assert repo_type == "dataset"
        assert revision == "refs/convert/parquet"
        return [
            "MARC-ja/jglue-validation.parquet",
            "MARC-ja/jglue-train.parquet",
        ]

    def fake_load_dataset(name, *args, **kwargs):
        calls.append((name, args, kwargs))
        return [{"label": 1, "sentence": "テスト"}]

    monkeypatch.setattr("tunescope.evaluation.urlopen", fake_urlopen)
    monkeypatch.setattr("huggingface_hub.list_repo_files", fake_list_repo_files)
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    records = _load_hf_dataset(
        {
            "id": "jglue",
            "name": "shunk031/JGLUE",
            "split": "validation",
            "load_mode": "hf_parquet_api",
            "subsets": ["MARC-ja"],
        },
        sample_count=None,
        seed=42,
        allow_floating_revision=False,
    )

    assert records == [{"_subset": "MARC-ja", "label": 1, "sentence": "テスト"}]
    assert calls[0][2]["data_files"] == {
        "validation": [
            "https://huggingface.co/datasets/shunk031/JGLUE/resolve/refs%2Fconvert%2Fparquet/MARC-ja/jglue-validation.parquet"
        ]
    }
