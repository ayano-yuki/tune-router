from __future__ import annotations

import unittest
from pathlib import Path

from clients import MockModelClient
from graphs import load_graphs
from models import Delegation, PolicyDecision, RouteDecision, RouterSignal
from selector import GraphSelector
from shadow import ShadowConfig, build_shadow_decisions, execute_shadow_decisions


class ShadowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graphs = load_graphs(Path(__file__).resolve().parents[2] / "graphs")
        self.signal = RouterSignal(
            scores={"Database": 0.46, "Storage": 0.36, "Network": 0.1, "General": 0.01, "Coding": 0.04, "Security": 0.03}
        )

    def test_learned_served_plan_shadows_deterministic_baseline(self) -> None:
        served = RouteDecision(
            graph_id="single_specialist",
            primary_labels=("Database",),
            secondary_labels=(),
            selected_labels=("Database",),
            confidence=0.8,
            margin=0.1,
            reason="learned chose a cheaper single specialist",
            policy=PolicyDecision(risk_level="normal", action="allow"),
            delegations=(Delegation("Database", "database-specialist", "Check waits."),),
            selector_type="learned_orchestrator",
        )
        shadows = build_shadow_decisions(
            text="PostgreSQL on NFS is slow",
            signal=self.signal,
            served_decision=served,
            risk_level="auto",
            selector=GraphSelector(),
            config=ShadowConfig(mode="deterministic-baseline", max_count=1),
        )
        self.assertEqual(1, len(shadows))
        self.assertEqual("deterministic_baseline", shadows[0][0])
        self.assertEqual("parallel_experts", shadows[0][1].graph_id)

        results = execute_shadow_decisions(
            text="PostgreSQL on NFS is slow",
            signal=self.signal,
            model_client=MockModelClient(),
            graphs=self.graphs,
            shadows=shadows,
            config=ShadowConfig(mode="deterministic-baseline", max_count=1),
            max_workers=2,
        )
        self.assertEqual("completed", results[0]["status"])
        self.assertEqual("parallel_experts", results[0]["trace"]["graph"]["id"])

    def test_shadow_skips_high_risk_by_default(self) -> None:
        signal = RouterSignal(
            scores={"Security": 0.8, "General": 0.05, "Network": 0.05, "Coding": 0.04, "Storage": 0.03, "Database": 0.03}
        )
        served = GraphSelector().select("Review our defensive TLS controls", signal)
        shadows = build_shadow_decisions(
            text="Review our defensive TLS controls",
            signal=signal,
            served_decision=served,
            risk_level="auto",
            selector=GraphSelector(),
            config=ShadowConfig(mode="alternatives", max_count=1),
        )
        self.assertEqual([], shadows)


if __name__ == "__main__":
    unittest.main()
