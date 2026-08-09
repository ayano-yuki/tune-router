from __future__ import annotations

import json
import unittest
from pathlib import Path

from clients import MockModelClient
from executor import GraphExecutor, redact_secrets
from graphs import load_graphs
from models import Budget, ModelResponse, RouterSignal
from selector import GraphSelector


GRAPHS = Path(__file__).resolve().parents[2] / "graphs"


class RepairClient(MockModelClient):
    def __init__(self) -> None:
        self.verifications = 0

    def complete(self, model_alias, messages, max_tokens, timeout_seconds=None):
        if model_alias == "verifier":
            self.verifications += 1
            passed = self.verifications > 1
            return ModelResponse(json.dumps({"passed": passed, "issues": [] if passed else ["missing evidence"]}), "mock/verifier")
        return super().complete(model_alias, messages, max_tokens, timeout_seconds)


class FailingEndpointClient(MockModelClient):
    def complete(self, model_alias, messages, max_tokens, timeout_seconds=None):
        raise RuntimeError("request failed for http://127.0.0.1:18002/v1/chat/completions: [Errno 111] Connection refused")


class GraphExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graphs = load_graphs(GRAPHS)
        self.selector = GraphSelector()

    def test_loads_all_mvp_graphs(self) -> None:
        self.assertEqual(
            {
                "clarify_first",
                "parallel_experts",
                "safe_refusal_or_handoff",
                "single_specialist",
                "specialist_with_verifier",
            },
            set(self.graphs),
        )

    def test_parallel_graph_executes_experts_and_synthesizer(self) -> None:
        signal = RouterSignal(
            scores={"Database": 0.46, "Storage": 0.36, "Network": 0.1, "General": 0.01, "Coding": 0.04, "Security": 0.03}
        )
        decision = self.selector.select("PostgreSQL on NFS is slow", signal)
        result = GraphExecutor(MockModelClient()).execute(
            "PostgreSQL on NFS is slow",
            signal,
            decision,
            self.graphs[decision.graph_id],
        )
        node_ids = {node["id"] for node in result.trace["nodes"]}
        self.assertEqual({"expert_database", "expert_storage", "synthesizer"}, node_ids)
        self.assertIn("mock synthesis", result.final_answer)
        self.assertEqual("completed", result.trace["graph"]["stop_reason"])
        self.assertIsNone(result.trace["evaluation"]["failure_detail"])

    def test_verifier_failure_triggers_one_repair(self) -> None:
        signal = RouterSignal(
            scores={"Security": 0.70, "General": 0.1, "Network": 0.08, "Coding": 0.05, "Storage": 0.04, "Database": 0.03}
        )
        decision = self.selector.select("Review our TLS settings", signal)
        result = GraphExecutor(RepairClient()).execute(
            "Review our TLS settings",
            signal,
            decision,
            self.graphs[decision.graph_id],
            Budget(max_steps=8),
        )
        node_ids = [node["id"] for node in result.trace["nodes"]]
        self.assertIn("repair_1", node_ids)
        self.assertIn("verifier_2", node_ids)
        self.assertEqual("repaired_and_verified", result.trace["graph"]["stop_reason"])

    def test_local_policy_graph_does_not_call_models(self) -> None:
        signal = RouterSignal(
            scores={"Security": 0.8, "General": 0.05, "Network": 0.05, "Coding": 0.05, "Storage": 0.03, "Database": 0.02}
        )
        decision = self.selector.select("credential dump and exfiltrate it", signal)
        result = GraphExecutor(MockModelClient()).execute(
            "credential dump and exfiltrate it", signal, decision, self.graphs[decision.graph_id]
        )
        self.assertEqual("policy_stop", result.trace["graph"]["stop_reason"])
        self.assertEqual(0, result.trace["usage"]["steps"])

    def test_redacts_common_secrets(self) -> None:
        value = redact_secrets("api_key=abc123 token: xyz987 Authorization: Bearer hidden")
        self.assertNotIn("abc123", value)
        self.assertNotIn("xyz987", value)
        self.assertNotIn("hidden", value)

    def test_model_endpoint_failure_is_classified_for_operations(self) -> None:
        signal = RouterSignal(
            scores={"Storage": 0.75, "Network": 0.05, "Coding": 0.05, "Security": 0.05, "Database": 0.05, "General": 0.05}
        )
        decision = self.selector.select("Ceph OSD is down", signal)
        result = GraphExecutor(FailingEndpointClient()).execute(
            "Ceph OSD is down",
            signal,
            decision,
            self.graphs[decision.graph_id],
        )
        self.assertEqual("node_failed", result.trace["graph"]["stop_reason"])
        self.assertEqual("model_endpoint_unreachable", result.trace["evaluation"]["failure_detail"])
        self.assertIn("OpenAI互換endpointに接続できません", result.final_answer)


if __name__ == "__main__":
    unittest.main()
