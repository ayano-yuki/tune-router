from __future__ import annotations

import json
import unittest

from training import _tokenized_dataset, evaluate_adapter


class FakeDatasetBase:
    pass


class FakeTorch:
    class utils:
        class data:
            Dataset = FakeDatasetBase


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        del messages, tokenize
        return "prompt" if add_generation_prompt else "full"

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [1, 2] if text == "prompt" else [1, 2, 3, 4]}


class StaticClient:
    def __init__(self, plan):
        self.plan = plan

    def generate(self, messages):
        del messages
        return json.dumps(self.plan)


class TrainingTests(unittest.TestCase):
    def test_sft_masks_system_and_user_tokens(self) -> None:
        record = {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "request"},
                {"role": "assistant", "content": "plan"},
            ]
        }
        dataset = _tokenized_dataset([record], FakeTokenizer(), FakeTorch(), max_length=32)
        self.assertEqual([-100, -100, 3, 4], dataset[0]["labels"])

    def test_adapter_evaluation_scores_structured_plan(self) -> None:
        plan = {"graph_id": "single_specialist", "selected_labels": ["Storage"]}
        records = [
            {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "request"},
                    {"role": "assistant", "content": json.dumps(plan)},
                ],
                "metadata": {"query_id": "q1"},
            }
        ]
        metrics, predictions = evaluate_adapter(StaticClient(plan), records)
        self.assertEqual(1.0, metrics["valid_plan_rate"])
        self.assertEqual(1.0, metrics["graph_accuracy"])
        self.assertEqual(1.0, metrics["selected_labels_exact_match"])
        self.assertIsNone(predictions[0]["error"])


if __name__ == "__main__":
    unittest.main()
