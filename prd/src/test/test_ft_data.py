from __future__ import annotations

import json
import unittest
from pathlib import Path

from ft_data import FTDataConfig, build_ft_datasets


class FTDataTests(unittest.TestCase):
    def test_builds_sft_and_preference_examples_without_outcome_leakage(self) -> None:
        fixture = Path(__file__).resolve().parents[2] / "eval" / "candidate-results.example.json"
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

    def test_approved_bounded_graph_trace_enters_sft_data(self) -> None:
        trace = {
            "trace_id": "trace-bounded-approved",
            "input": {"text": "PostgreSQL on NFS is slow"},
            "router": {
                "scores": {"Database": 0.46, "Storage": 0.36, "Network": 0.1, "General": 0.01, "Coding": 0.04, "Security": 0.03},
                "primary_labels": ["Database"],
                "secondary_labels": ["Storage"],
                "confidence": 0.46,
            },
            "policy": {"risk_level": "normal"},
            "graph": {
                "id": "bounded_graph",
                "selection_reason": "reviewed bounded graph",
                "generated_graph": {
                    "plan_type": "bounded_graph",
                    "version": "generated-v1",
                    "max_steps": 3,
                    "nodes": [
                        {
                            "id": "database_1",
                            "role": "specialist",
                            "label": "Database",
                            "model": "database-specialist",
                            "dependencies": [],
                            "objective": "Check waits.",
                        },
                        {
                            "id": "storage_1",
                            "role": "specialist",
                            "label": "Storage",
                            "model": "storage-specialist",
                            "dependencies": [],
                            "objective": "Check NFS.",
                        },
                        {
                            "id": "synthesis_1",
                            "role": "synthesizer",
                            "model": "general-synthesizer",
                            "dependencies": ["database_1", "storage_1"],
                            "objective": "Merge findings.",
                        },
                    ],
                    "final_node": "synthesis_1",
                },
            },
            "evaluation": {"review_label": "approved"},
        }
        datasets = build_ft_datasets([], traces=[trace], config=FTDataConfig(dev_ratio=0.0))
        plan = json.loads(datasets["train"][0]["messages"][-1]["content"])
        self.assertEqual("bounded_graph", plan["plan_type"])
        self.assertEqual("synthesis_1", plan["final_node"])

    def test_reviewed_traces_create_trajectory_preferences(self) -> None:
        base = {
            "input": {"text": "PostgreSQL on NFS is slow"},
            "router": {
                "scores": {"Database": 0.46, "Storage": 0.36, "Network": 0.1, "General": 0.01, "Coding": 0.04, "Security": 0.03},
                "primary_labels": ["Database"],
                "secondary_labels": ["Storage"],
                "confidence": 0.46,
            },
            "policy": {"risk_level": "normal"},
            "usage": {"steps": 2, "cost_usd": 0.02, "latency_ms": 3000},
        }
        chosen = {
            **base,
            "trace_id": "trace-chosen",
            "graph": {
                "id": "parallel_experts",
                "selector_type": "learned_orchestrator",
                "selection_reason": "reviewed better trajectory",
                "stop_reason": "completed",
                "delegations": [
                    {"label": "Database", "model": "database-specialist", "objective": "Check waits."},
                    {"label": "Storage", "model": "storage-specialist", "objective": "Check NFS."},
                ],
                "synthesis_strategy": "Merge findings.",
            },
            "nodes": [
                {"id": "expert_database", "role": "specialist", "model": "database-specialist", "status": "completed", "attempts": 1},
                {"id": "expert_storage", "role": "specialist", "model": "storage-specialist", "status": "completed", "attempts": 1},
                {"id": "synthesizer", "role": "synthesizer", "model": "general-synthesizer", "status": "completed", "attempts": 1},
            ],
            "evaluation": {"query_id": "incident-1", "user_rating": 5},
        }
        rejected = {
            **base,
            "trace_id": "trace-rejected",
            "graph": {
                "id": "single_specialist",
                "selector_type": "deterministic",
                "selection_reason": "baseline",
                "stop_reason": "completed",
                "delegations": [
                    {"label": "Database", "model": "database-specialist", "objective": "Check waits."}
                ],
            },
            "nodes": [
                {"id": "specialist", "role": "specialist", "model": "database-specialist", "status": "completed", "attempts": 1},
            ],
            "evaluation": {"query_id": "incident-1", "user_rating": 2},
        }
        datasets = build_ft_datasets([], traces=[chosen, rejected], config=FTDataConfig(dev_ratio=0.0))
        self.assertEqual(1, len(datasets["trajectory_preferences"]))
        preference = datasets["trajectory_preferences"][0]
        self.assertEqual("trace-chosen", preference["metadata"]["chosen_trace_id"])
        self.assertEqual("trace-rejected", preference["metadata"]["rejected_trace_id"])
        self.assertNotIn("user_rating", json.dumps(preference["prompt"], ensure_ascii=False))
        self.assertEqual("parallel_experts", json.loads(preference["chosen"])["graph_id"])
        self.assertEqual("single_specialist", json.loads(preference["rejected"])["graph_id"])

    def test_high_risk_trajectory_preference_requires_verifier_trace(self) -> None:
        trace = {
            "trace_id": "high-risk-unverified",
            "input": {"text": "Review defensive controls"},
            "router": {
                "scores": {"Security": 0.8, "General": 0.05, "Network": 0.05, "Coding": 0.04, "Storage": 0.03, "Database": 0.03},
                "primary_labels": ["Security"],
                "secondary_labels": [],
                "confidence": 0.8,
            },
            "policy": {"risk_level": "high"},
            "graph": {
                "id": "specialist_with_verifier",
                "selection_reason": "high risk",
                "stop_reason": "completed",
            },
            "nodes": [
                {"id": "specialist", "role": "specialist", "model": "security-specialist", "status": "completed"},
            ],
            "evaluation": {"query_id": "security-trajectory", "user_rating": 5},
        }
        datasets = build_ft_datasets([], traces=[trace, {**trace, "trace_id": "low", "evaluation": {"query_id": "security-trajectory", "user_rating": 1}}])
        self.assertEqual(0, len(datasets["trajectory_preferences"]))

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

    def test_model_catalog_allows_non_default_domain_model(self) -> None:
        record = {
            "query_id": "storage-catalog-1",
            "query": "PVC on NFS is slow",
            "domain_labels": ["Storage"],
            "router_scores": {"Storage": 0.8, "Database": 0.1, "Network": 0.05, "Coding": 0.02, "Security": 0.02, "General": 0.01},
            "model_catalog": [
                {
                    "alias": "storage-long-context",
                    "domains": ["Storage"],
                    "strengths": ["long-log-analysis"],
                    "context_window": 65536,
                    "cost_tier": "medium",
                },
                {"alias": "general-fallback", "domains": ["*"], "cost_tier": "low"},
            ],
            "candidate_results": [
                {
                    "candidate_id": "storage-long-context",
                    "candidate_type": "model",
                    "quality": 0.93,
                    "cost": 0.03,
                    "latency_ms": 3500,
                    "domains": ["Storage"],
                    "strengths": ["long-log-analysis"],
                    "context_window": 65536,
                },
                {
                    "candidate_id": "general-fallback",
                    "candidate_type": "model",
                    "quality": 0.55,
                    "cost": 0.004,
                    "latency_ms": 900,
                    "domains": ["*"],
                },
            ],
        }
        datasets = build_ft_datasets([record], config=FTDataConfig(dev_ratio=0.0))
        plan = json.loads(datasets["train"][0]["messages"][-1]["content"])
        prompt = json.loads(datasets["train"][0]["messages"][1]["content"])
        self.assertEqual("storage-long-context", plan["delegations"][0]["model"])
        self.assertIn("model_catalog", prompt)
        self.assertEqual("storage-long-context", prompt["model_catalog"][1]["alias"])


if __name__ == "__main__":
    unittest.main()
