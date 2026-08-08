from __future__ import annotations

import json
import unittest

from tune_orchestrator.learned import LearnedGraphSelector
from tune_orchestrator.models import RouterSignal


class StaticPlanClient:
    def __init__(self, value: str) -> None:
        self.value = value
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
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


if __name__ == "__main__":
    unittest.main()
