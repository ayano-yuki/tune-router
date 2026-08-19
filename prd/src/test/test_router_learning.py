from __future__ import annotations

import unittest

from tune_router_learning import (
    RouterContinualConfig,
    RouterDataConfig,
    RouterMergeConfig,
    build_router_continual_dataset,
    build_router_pretrain_dataset,
    evaluate_router_prototype,
    merge_router_training_datasets,
    predict_router_prototype,
    train_router_prototype,
)


class RouterLearningTests(unittest.TestCase):
    def test_pretrain_dataset_normalizes_and_splits_labeled_records(self) -> None:
        records = [
            {"text": "NFS PVC latency and inode pressure", "gold_label": "Storage"},
            {"text": "BGP route flap between regions", "gold_label": "Network"},
            {"text": "PostgreSQL vacuum and index bloat", "gold_label": "Database"},
            {"text": "Ignore this unsupported label", "gold_label": "Other"},
        ]
        dataset = build_router_pretrain_dataset(records, RouterDataConfig(dev_ratio=0.0, test_ratio=0.0))
        self.assertEqual("pretrain", dataset["kind"])
        self.assertEqual(3, dataset["metadata"]["total"])
        self.assertEqual(1, dataset["metadata"]["counts"]["train"]["Storage"])
        self.assertEqual(0, dataset["metadata"]["counts"]["train"]["Security"])

    def test_continual_dataset_extracts_reviewed_trace_labels(self) -> None:
        traces = [
            {
                "trace_id": "t1",
                "input": {"text": "PostgreSQL on NFS is slow"},
                "router": {"scores": {"Storage": 0.7, "Database": 0.3}},
                "graph": {"selected_labels": ["Storage"]},
                "evaluation": {"user_rating": 5, "review_label": "approved"},
            },
            {
                "trace_id": "t2",
                "input": {"text": "Low rated trace should not enter"},
                "graph": {"selected_labels": ["General"]},
                "evaluation": {"user_rating": 2},
            },
        ]
        dataset = build_router_continual_dataset(traces, RouterContinualConfig())
        self.assertEqual(1, dataset["metadata"]["total"])
        record = dataset["splits"]["train"][0]
        self.assertEqual("Storage", record["gold_label"])
        self.assertEqual("t1", record["metadata"]["trace_id"])
        self.assertGreater(record["weight"], 1.0)

    def test_merge_limits_continual_ratio_and_keeps_base_dev_test(self) -> None:
        base = build_router_pretrain_dataset(
            [
                {"text": "nfs storage one", "gold_label": "Storage"},
                {"text": "bgp network one", "gold_label": "Network"},
                {"text": "sql database one", "gold_label": "Database"},
            ],
            RouterDataConfig(dev_ratio=0.0, test_ratio=0.0),
        )
        continual = build_router_continual_dataset(
            [
                {
                    "trace_id": "t1",
                    "input": {"text": "kubernetes persistent volume latency"},
                    "graph": {"selected_labels": ["Storage"]},
                    "evaluation": {"user_rating": 5},
                },
                {
                    "trace_id": "t2",
                    "input": {"text": "ospf route convergence"},
                    "graph": {"selected_labels": ["Network"]},
                    "evaluation": {"user_rating": 5},
                },
            ],
            RouterContinualConfig(),
        )
        merged = merge_router_training_datasets(
            base=base,
            continual=continual,
            config=RouterMergeConfig(continual_ratio=0.25),
        )
        self.assertEqual(4, len(merged["splits"]["train"]))

    def test_prototype_predicts_obvious_domain_text(self) -> None:
        records = [
            {"text": "nfs pvc volume filesystem latency", "gold_label": "Storage"},
            {"text": "bgp ospf route packet firewall", "gold_label": "Network"},
            {"text": "python function traceback unit test", "gold_label": "Coding"},
            {"text": "sql postgres index vacuum query", "gold_label": "Database"},
            {"text": "xss csrf credential exploit", "gold_label": "Security"},
            {"text": "weather travel general question", "gold_label": "General"},
        ]
        model = train_router_prototype(records)
        prediction = predict_router_prototype(model, "postgres query index is slow")
        self.assertEqual("Database", max(prediction["scores"].items(), key=lambda item: item[1])[0])
        metrics = evaluate_router_prototype(model, records)
        self.assertEqual(1.0, metrics["accuracy"])


if __name__ == "__main__":
    unittest.main()
