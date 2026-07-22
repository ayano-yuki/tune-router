from pathlib import Path

from tunescope.cli import render_run_card
from tunescope.config import validate_workspace


ROOT = Path(__file__).resolve().parents[1]


def test_workspace_has_no_validation_errors() -> None:
    messages = validate_workspace(ROOT)
    errors = [message.message for message in messages if message.level == "error"]
    assert errors == []


def test_run_card_mentions_reused_result() -> None:
    card = render_run_card("R2", ROOT)
    assert "reuses_result_from: Q3" in card

