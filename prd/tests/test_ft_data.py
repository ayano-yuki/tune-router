from __future__ import annotations

import json
import unittest
from pathlib import Path

from tune_orchestrator.ft_data import FTDataConfig, build_ft_datasets


class FTDataTests(unittest.TestCase):
    def test_builds_sft_and_preference_examples_without_outcome_leakage(self) -> None:
        fixture = Path(__file__).resolve().parents[1] / "eval" / "candidate-results.example.json"
        records = json.loads(fixture.read_text(encoding="utf-8"))["records"]
        datasets = build_ft_datasets(records, config=FTDataConfig(dev_ratio=0.0))
        self.assertEqual(2, len(datasets["train"]))
        self.assertEqual(0, len(datasets["dev"]))
        self.assertEqual(2, len(datasets["preferences"]))

        first = datasets["train"][0]
        prompt = first["messages"][1]["content"]
        self.assertNotIn('"quality"', prompt)
        self.assertNotIn('"candidate_results"', prompt)
        plan = json.loads(first["messages"][-1]["content"])
        self.assertEqual("single_specialist", plan["graph_id"])
        self.assertEqual("storage-specialist", plan["delegations"][0]["model"])

        second_plan = json.loads(datasets["train"][1]["messages"][-1]["content"])
        self.assertEqual("parallel_experts", second_plan["graph_id"])
        self.assertEqual(["Database", "Storage"], second_plan["selected_labels"][:2])

    def test_only_approved_traces_enter_sft_data(self) -> None:
        approved = {
            "trace_id": "trace-approved",
            "input": {"text": "Ceph is degraded"},
            "router": {
                "scores": {"Storage": 0.8, "General": 0.2},
                "primary_labels": ["Storage"],
                "secondary_labels": [],
                "confidence": 0.8,
            },
            "policy": {"risk_level": "normal"},
            "graph": {"id": "single_specialist", "selection_reason": "high confidence"},
            "evaluation": {"user_rating": 5, "review_label": None},
        }
        rejected = {**approved, "trace_id": "trace-rejected", "evaluation": {"user_rating": 2}}
        datasets = build_ft_datasets([], traces=[approved, rejected], config=FTDataConfig(dev_ratio=0.0))
        self.assertEqual(1, len(datasets["train"]))
        self.assertEqual("approved_execution_trace", datasets["train"][0]["metadata"]["source"])

    def test_high_risk_outcome_without_verified_candidate_is_skipped(self) -> None:
        record = {
            "query_id": "security-1",
            "query": "Review our defensive TLS controls",
            "router_scores": {"Security": 0.8, "General": 0.05, "Network": 0.05, "Coding": 0.04, "Storage": 0.03, "Database": 0.03},
            "candidate_results": [
                {"candidate_id": "security-specialist", "candidate_type": "model", "quality": 0.9, "cost": 0.02, "latency_ms": 2000}
            ],
        }
        datasets = build_ft_datasets([record], config=FTDataConfig(dev_ratio=0.0))
        self.assertEqual(0, len(datasets["train"]))
        self.assertEqual(1, datasets["summary"]["skipped_examples"])


if __name__ == "__main__":
    unittest.main()
