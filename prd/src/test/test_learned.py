from __future__ import annotations

import json
import unittest

from tune_clients import MockModelClient
from tune_composition import graph_from_bounded_plan
from tune_executor import GraphExecutor
from tune_learned import LearnedGraphSelector
from tune_models import RouterSignal


class StaticPlanClient:
    def __init__(self, value: str) -> None:
        self.value = value
        self.calls = 0
        self.messages = None

    def generate(self, messages):
        self.calls += 1
        self.messages = messages
        return self.value


class LearnedSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.signal = RouterSignal(
            scores={"Database": 0.46, "Storage": 0.36, "Network": 0.10, "Coding": 0.04, "Security": 0.03, "General": 0.01}
        )

    def test_accepts_structured_delegation_plan(self) -> None:
        client = StaticPlanClient(
            json.dumps(
                {
                    "graph_id": "parallel_experts",
                    "primary_labels": ["Database"],
                    "selected_labels": ["Database", "Storage"],
                    "confidence": 0.88,
                    "risk_level": "normal",
                    "reason": "database and storage evidence are both required",
                    "delegations": [
                        {"label": "Database", "model": "database-specialist", "objective": "Check locks and query plans."},
                        {"label": "Storage", "model": "storage-specialist", "objective": "Check NFS latency and mount options."},
                    ],
                    "synthesis_strategy": "Order checks by diagnostic value.",
                }
            )
        )
        decision = LearnedGraphSelector(client).select("PostgreSQL on NFS is slow", self.signal)
        self.assertEqual("learned_orchestrator", decision.selector_type)
        self.assertEqual("parallel_experts", decision.graph_id)
        self.assertEqual("Check locks and query plans.", decision.delegations[0].objective)

    def test_invalid_plan_falls_back_deterministically(self) -> None:
        decision = LearnedGraphSelector(StaticPlanClient("not-json")).select(
            "PostgreSQL on NFS is slow", self.signal
        )
        self.assertEqual("deterministic_fallback", decision.selector_type)
        self.assertEqual("parallel_experts", decision.graph_id)
        self.assertIn("rejected", decision.fallback_reason)

    def test_policy_gate_runs_before_learned_model(self) -> None:
        client = StaticPlanClient("{}")
        security_signal = RouterSignal(
            scores={"Security": 0.8, "General": 0.05, "Network": 0.05, "Coding": 0.05, "Storage": 0.03, "Database": 0.02}
        )
        decision = LearnedGraphSelector(client).select("credential dump and exfiltrate it", security_signal)
        self.assertEqual("safe_refusal_or_handoff", decision.graph_id)
        self.assertEqual("policy_gate", decision.selector_type)
        self.assertEqual(0, client.calls)

    def test_unapproved_model_forces_fallback(self) -> None:
        client = StaticPlanClient(
            json.dumps(
                {
                    "graph_id": "single_specialist",
                    "primary_labels": ["Database"],
                    "selected_labels": ["Database"],
                    "confidence": 0.9,
                    "risk_level": "normal",
                    "delegations": [
                        {"label": "Database", "model": "unknown-provider", "objective": "Analyze the database."}
                    ],
                }
            )
        )
        decision = LearnedGraphSelector(client).select("Database is slow", self.signal)
        self.assertEqual("deterministic_fallback", decision.selector_type)

    def test_model_catalog_is_sent_to_learned_orchestrator(self) -> None:
        client = StaticPlanClient(
            json.dumps(
                {
                    "graph_id": "single_specialist",
                    "primary_labels": ["Storage"],
                    "selected_labels": ["Storage"],
                    "confidence": 0.9,
                    "risk_level": "normal",
                    "delegations": [
                        {"label": "Storage", "model": "storage-long-context", "objective": "Review NFS and PVC evidence."}
                    ],
                }
            )
        )
        storage_signal = RouterSignal(
            scores={"Storage": 0.8, "Database": 0.1, "Network": 0.05, "Coding": 0.02, "Security": 0.02, "General": 0.01}
        )
        decision = LearnedGraphSelector(
            client,
            allowed_models={"storage-long-context", "general-fallback"},
            model_catalog=[
                {
                    "alias": "storage-long-context",
                    "domains": ["Storage"],
                    "strengths": ["long-log-analysis"],
                    "context_window": 65536,
                    "cost_tier": "medium",
                },
                {"alias": "general-fallback", "domains": ["*"], "cost_tier": "low"},
            ],
        ).select("PVC on NFS is slow", storage_signal)
        self.assertEqual("storage-long-context", decision.delegations[0].model)
        payload = json.loads(client.messages[1]["content"])
        self.assertEqual("storage-long-context", payload["model_catalog"][1]["alias"])
        self.assertEqual(["Storage"], payload["model_catalog"][1]["domains"])

    def test_model_catalog_domain_mismatch_forces_fallback(self) -> None:
        client = StaticPlanClient(
            json.dumps(
                {
                    "graph_id": "single_specialist",
                    "primary_labels": ["Storage"],
                    "selected_labels": ["Storage"],
                    "confidence": 0.9,
                    "risk_level": "normal",
                    "delegations": [
                        {"label": "Storage", "model": "database-specialist", "objective": "Analyze storage."}
                    ],
                }
            )
        )
        storage_signal = RouterSignal(
            scores={"Storage": 0.8, "Database": 0.1, "Network": 0.05, "Coding": 0.02, "Security": 0.02, "General": 0.01}
        )
        decision = LearnedGraphSelector(
            client,
            model_catalog=[
                {"alias": "database-specialist", "domains": ["Database"]},
                {"alias": "storage-specialist", "domains": ["Storage"]},
            ],
        ).select("PVC on NFS is slow", storage_signal)
        self.assertEqual("deterministic_fallback", decision.selector_type)
        self.assertIn("does not support label Storage", decision.fallback_reason)

    def test_high_risk_plan_cannot_bypass_verifier(self) -> None:
        client = StaticPlanClient(
            json.dumps(
                {
                    "graph_id": "parallel_experts",
                    "primary_labels": ["Security"],
                    "selected_labels": ["Security", "Network"],
                    "confidence": 0.9,
                    "risk_level": "high",
                    "delegations": [
                        {"label": "Security", "model": "security-specialist", "objective": "Review defensive controls."},
                        {"label": "Network", "model": "network-specialist", "objective": "Review network exposure."},
                    ],
                }
            )
        )
        security_signal = RouterSignal(
            scores={"Security": 0.7, "Network": 0.15, "General": 0.05, "Coding": 0.04, "Storage": 0.03, "Database": 0.03}
        )
        decision = LearnedGraphSelector(client).select("Review our defensive TLS controls", security_signal)
        self.assertEqual("deterministic_fallback", decision.selector_type)
        self.assertEqual("specialist_with_verifier", decision.graph_id)

    def test_accepts_and_executes_bounded_graph_plan(self) -> None:
        client = StaticPlanClient(
            json.dumps(
                {
                    "plan_type": "bounded_graph",
                    "primary_labels": ["Database"],
                    "selected_labels": ["Database", "Storage"],
                    "confidence": 0.91,
                    "risk_level": "normal",
                    "reason": "database and storage evidence should be gathered independently",
                    "nodes": [
                        {
                            "id": "database_1",
                            "role": "specialist",
                            "label": "Database",
                            "model": "database-specialist",
                            "dependencies": [],
                            "objective": "Check PostgreSQL waits and checkpoints.",
                        },
                        {
                            "id": "storage_1",
                            "role": "specialist",
                            "label": "Storage",
                            "model": "storage-specialist",
                            "dependencies": [],
                            "objective": "Check NFS latency and mount options.",
                        },
                        {
                            "id": "synthesis_1",
                            "role": "synthesizer",
                            "model": "general-synthesizer",
                            "dependencies": ["database_1", "storage_1"],
                            "objective": "Prioritize the incident response.",
                        },
                    ],
                    "final_node": "synthesis_1",
                }
            )
        )
        decision = LearnedGraphSelector(client).select("PostgreSQL on NFS is slow", self.signal)
        self.assertEqual("learned_orchestrator", decision.selector_type)
        self.assertEqual("bounded_graph", decision.graph_id)
        self.assertIsNotNone(decision.generated_graph)

        result = GraphExecutor(MockModelClient()).execute(
            "PostgreSQL on NFS is slow",
            self.signal,
            decision,
            graph_from_bounded_plan(decision.generated_graph),
        )
        self.assertEqual("completed", result.trace["graph"]["stop_reason"])
        self.assertEqual({"database_1", "storage_1", "synthesis_1"}, {node["id"] for node in result.trace["nodes"]})

    def test_high_risk_bounded_graph_requires_verifier_on_final_path(self) -> None:
        client = StaticPlanClient(
            json.dumps(
                {
                    "plan_type": "bounded_graph",
                    "primary_labels": ["Security"],
                    "selected_labels": ["Security"],
                    "confidence": 0.91,
                    "risk_level": "high",
                    "nodes": [
                        {
                            "id": "security_1",
                            "role": "specialist",
                            "label": "Security",
                            "model": "security-specialist",
                            "dependencies": [],
                            "objective": "Review defensive controls.",
                        }
                    ],
                    "final_node": "security_1",
                }
            )
        )
        security_signal = RouterSignal(
            scores={"Security": 0.7, "Network": 0.15, "General": 0.05, "Coding": 0.04, "Storage": 0.03, "Database": 0.03}
        )
        decision = LearnedGraphSelector(client).select("Review our defensive TLS controls", security_signal)
        self.assertEqual("deterministic_fallback", decision.selector_type)
        self.assertIn("requires a verifier", decision.fallback_reason)


if __name__ == "__main__":
    unittest.main()
