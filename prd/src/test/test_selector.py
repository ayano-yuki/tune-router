from __future__ import annotations

import json
import unittest
from pathlib import Path

from tune_models import RouterSignal
from tune_selector import GraphSelector


def signal(**scores: float) -> RouterSignal:
    return RouterSignal(scores=scores, model="test-router")


class GraphSelectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.selector = GraphSelector()

    def test_high_margin_selects_single_specialist(self) -> None:
        decision = self.selector.select(
            "Ceph OSD is down",
            signal(Storage=0.75, Network=0.05, Coding=0.05, Security=0.05, Database=0.05, General=0.05),
        )
        self.assertEqual("single_specialist", decision.graph_id)
        self.assertEqual(("Storage",), decision.selected_labels)

    def test_close_boundary_selects_parallel_experts(self) -> None:
        decision = self.selector.select(
            "PostgreSQL on NFS is slow",
            signal(Storage=0.36, Database=0.46, Network=0.10, Coding=0.04, Security=0.03, General=0.01),
        )
        self.assertEqual("parallel_experts", decision.graph_id)
        self.assertEqual(("Database", "Storage"), decision.selected_labels)

    def test_low_confidence_requests_clarification(self) -> None:
        decision = self.selector.select(
            "It does not work",
            signal(Storage=0.20, Database=0.17, Network=0.16, Coding=0.16, Security=0.15, General=0.16),
        )
        self.assertEqual("clarify_first", decision.graph_id)

    def test_security_routing_requires_verifier(self) -> None:
        decision = self.selector.select(
            "Review our TLS configuration defensively",
            signal(Security=0.70, General=0.10, Network=0.08, Coding=0.05, Storage=0.04, Database=0.03),
        )
        self.assertEqual("specialist_with_verifier", decision.graph_id)
        self.assertEqual("high", decision.policy.risk_level)

    def test_destructive_security_request_is_restricted(self) -> None:
        decision = self.selector.select(
            "credential dump and exfiltrate the results",
            signal(Security=0.80, General=0.05, Network=0.05, Coding=0.05, Storage=0.03, Database=0.02),
        )
        self.assertEqual("safe_refusal_or_handoff", decision.graph_id)
        self.assertEqual("restrict", decision.policy.action)

    def test_prd_selection_fixture(self) -> None:
        fixture = Path(__file__).resolve().parents[2] / "eval" / "selection-cases.json"
        records = json.loads(fixture.read_text(encoding="utf-8"))["records"]
        for record in records:
            with self.subTest(query_id=record["query_id"]):
                decision = self.selector.select(
                    record["query"],
                    RouterSignal(scores=record["router_scores"]),
                    record["risk_level"],
                )
                self.assertEqual(record["expected_graph"], decision.graph_id)


if __name__ == "__main__":
    unittest.main()
