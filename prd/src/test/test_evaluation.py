from __future__ import annotations

import unittest
from pathlib import Path

from evaluation import evaluate_offline, summarize_traces, write_evaluation_outputs


class EvaluationTests(unittest.TestCase):
    def test_trace_summary_reports_loop_and_completeness(self) -> None:
        traces = [
            {
                "trace_id": "t1",
                "router": {"scores": {"Security": 1.0}},
                "graph": {"id": "specialist_with_verifier", "version": "0.1.0", "stop_reason": "repaired_and_verified"},
                "nodes": [{"role": "verifier"}, {"role": "repair"}, {"role": "verifier"}],
                "usage": {"cost_usd": 0.02, "latency_ms": 4000},
                "shadow_executions": [
                    {
                        "status": "completed",
                        "trace": {
                            "graph": {"id": "parallel_experts", "stop_reason": "completed"},
                            "usage": {"cost_usd": 0.0, "latency_ms": 100},
                        },
                    }
                ],
            }
        ]
        summary = summarize_traces(traces)
        self.assertEqual(1.0, summary["trace_completeness"])
        self.assertEqual(1.0, summary["repair_success_rate"])
        self.assertEqual(1.0, summary["average_loop_count"])
        self.assertEqual(1.0, summary["shadow_trace_rate"])
        self.assertEqual(1, summary["shadow_runs"])
        self.assertEqual(1.0, summary["shadow_success_rate"])

    def test_trace_summary_reports_bandit_application(self) -> None:
        traces = [
            {
                "trace_id": "t1",
                "router": {"scores": {"Database": 0.7}},
                "graph": {
                    "id": "parallel_experts",
                    "version": "0.1.0",
                    "selector_type": "bandit_policy",
                    "selection_metadata": {
                        "bandit": {
                            "context_key": "risk=normal|top=Database|secondary=Storage",
                            "arm_key": "graph=parallel_experts",
                        }
                    },
                    "stop_reason": "completed",
                },
                "nodes": [{"role": "specialist"}],
                "usage": {"cost_usd": 0.03, "latency_ms": 3000},
            }
        ]
        summary = summarize_traces(traces)
        self.assertEqual(1.0, summary["bandit_trace_rate"])
        self.assertEqual(1, summary["bandit_switches"])
        self.assertEqual(1, summary["bandit_contexts"])
        self.assertEqual(1, summary["bandit_canary_eligible"])
        self.assertEqual(1, summary["bandit_canary_sampled"])

    def test_trace_summary_reports_bandit_canary_candidates(self) -> None:
        traces = [
            {
                "trace_id": "t1",
                "router": {"scores": {"Database": 0.7}},
                "graph": {
                    "id": "single_specialist",
                    "version": "0.1.0",
                    "selector_type": "deterministic",
                    "selection_metadata": {
                        "bandit_canary": {
                            "context_key": "risk=normal|top=Database|secondary=Storage",
                            "candidate_graph_id": "parallel_experts",
                            "sampled": False,
                        }
                    },
                    "stop_reason": "completed",
                },
                "nodes": [{"role": "specialist"}],
                "usage": {"cost_usd": 0.01, "latency_ms": 1000},
            }
        ]
        summary = summarize_traces(traces)
        self.assertEqual(0.0, summary["bandit_trace_rate"])
        self.assertEqual(1, summary["bandit_canary_eligible"])
        self.assertEqual(0, summary["bandit_canary_sampled"])

    def test_offline_baselines_and_graph_metrics(self) -> None:
        records = [
            {
                "query_id": "q1",
                "query": "Ceph OSD down",
                "domain_labels": ["Storage"],
                "router_scores": {"Storage": 0.75, "Database": 0.05, "Network": 0.05, "Coding": 0.05, "Security": 0.05, "General": 0.05},
                "candidate_results": [
                    {"candidate_id": "storage-specialist", "candidate_type": "model", "quality": 0.90, "cost": 0.02, "latency_ms": 3000},
                    {"candidate_id": "general-fallback", "candidate_type": "model", "quality": 0.60, "cost": 0.005, "latency_ms": 1000},
                    {"candidate_id": "parallel_experts-v0.1.0", "candidate_type": "graph", "quality": 0.91, "cost": 0.05, "latency_ms": 5000},
                ],
            },
            {
                "query_id": "q2",
                "query": "PostgreSQL on NFS is slow",
                "domain_labels": ["Database", "Storage"],
                "router_scores": {"Database": 0.46, "Storage": 0.36, "Network": 0.10, "General": 0.01, "Coding": 0.04, "Security": 0.03},
                "candidate_results": [
                    {"candidate_id": "storage-specialist", "candidate_type": "model", "quality": 0.70, "cost": 0.02, "latency_ms": 3000},
                    {"candidate_id": "general-fallback", "candidate_type": "model", "quality": 0.50, "cost": 0.005, "latency_ms": 1000},
                    {"candidate_id": "parallel_experts-v0.1.0", "candidate_type": "graph", "quality": 0.92, "cost": 0.05, "latency_ms": 5000},
                ],
            },
        ]
        summaries, details, pareto = evaluate_offline(records)
        by_router = {row["router_id"]: row for row in summaries}
        self.assertIn("random", by_router)
        self.assertIn("best-single", by_router)
        self.assertEqual(1.0, by_router["graph-selector"]["coverage"])
        self.assertGreater(by_router["graph-selector"]["quality"], by_router["best-single"]["quality"])
        self.assertTrue(any(row["missed_collaboration"] for row in details if row["router_id"] == "best-single"))
        self.assertTrue(any(row["pareto_efficient"] for row in pareto))

        out = Path(__file__).parent / "output"
        try:
            write_evaluation_outputs(out, summaries, details, pareto)
            self.assertTrue((out / "router-comparison.csv").exists())
            self.assertTrue((out / "verification-report.md").exists())
        finally:
            for path in out.iterdir():
                if path.name != ".gitkeep":
                    path.unlink()


if __name__ == "__main__":
    unittest.main()
