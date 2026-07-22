from tunescope.evaluation import exact_match, macro_f1, pearsonr, rouge_scores, token_f1


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
