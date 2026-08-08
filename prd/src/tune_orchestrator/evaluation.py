from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .constants import LABEL_TO_MODEL, MULTI_AGENT_GRAPHS
from .models import RouterSignal
from .selector import GraphSelector


@dataclass(frozen=True)
class EvaluationConfig:
    oracle_quality_tolerance: float = 0.02
    missed_collaboration_regret: float = 0.05
    success_quality: float = 0.8
    random_seed: int = 42


def load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        records = value.get("records") if isinstance(value, dict) else value
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError(f"expected an array of objects: {path}")
    return records


def evaluate_offline(
    candidate_records: list[dict[str, Any]],
    predictions: list[dict[str, Any]] | None = None,
    config: EvaluationConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = config or EvaluationConfig()
    _validate_candidate_records(candidate_records)
    result_maps = {
        _query_id(record): {str(item["candidate_id"]): item for item in record["candidate_results"]}
        for record in candidate_records
    }
    oracles = {
        query_id: _oracle(results, cfg.oracle_quality_tolerance)
        for query_id, results in result_maps.items()
    }

    selectors: dict[str, Callable[[dict[str, Any]], str]] = {}
    model_stats = _candidate_averages(candidate_records, candidate_type="model")
    all_stats = _candidate_averages(candidate_records)
    if not all_stats:
        raise ValueError("candidate results are empty")
    if model_stats:
        selectors["always-small"] = _constant_selector(min(model_stats, key=lambda key: model_stats[key]["cost"]))
        selectors["always-large"] = _constant_selector(max(model_stats, key=lambda key: model_stats[key]["cost"]))
        selectors["best-single"] = _constant_selector(max(model_stats, key=lambda key: model_stats[key]["quality"]))
    selectors["random"] = _random_selector(cfg.random_seed)
    selectors["rule-based"] = _rule_selector
    selectors["graph-selector"] = _graph_selector

    by_router: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for prediction in predictions or []:
        query_id = str(prediction.get("query_id") or prediction.get("question_id") or "")
        router_id = str(prediction.get("router_id") or "provided-router")
        candidate_id = str(prediction.get("selected_candidate_id") or prediction.get("target_model") or "")
        if query_id and candidate_id:
            by_router[router_id][query_id] = prediction

    details: list[dict[str, Any]] = []
    for router_id, selector in selectors.items():
        for record in candidate_records:
            query_id = _query_id(record)
            try:
                selected_id = selector(record)
            except (KeyError, ValueError):
                continue
            if selected_id not in result_maps[query_id]:
                continue
            details.append(
                _detail(
                    router_id,
                    record,
                    result_maps[query_id][selected_id],
                    oracles[query_id],
                    cfg,
                    router_cost=float(record.get("router_cost", 0.0)) if router_id == "graph-selector" else 0.0,
                    router_latency_ms=float(record.get("router_latency_ms", 0.0)) if router_id == "graph-selector" else 0.0,
                )
            )
    for router_id, mapping in by_router.items():
        for record in candidate_records:
            query_id = _query_id(record)
            prediction = mapping.get(query_id, {})
            selected_id = str(prediction.get("selected_candidate_id") or prediction.get("target_model") or "")
            if selected_id in result_maps[query_id]:
                details.append(
                    _detail(
                        router_id,
                        record,
                        result_maps[query_id][selected_id],
                        oracles[query_id],
                        cfg,
                        router_cost=float(prediction.get("router_cost", 0.0)),
                        router_latency_ms=float(prediction.get("router_latency_ms", 0.0)),
                    )
                )

    summaries = _summaries(details, len(candidate_records), cfg)
    pareto = _pareto_rows(summaries)
    return summaries, details, pareto


def write_evaluation_outputs(
    out_dir: Path,
    summaries: list[dict[str, Any]],
    details: list[dict[str, Any]],
    pareto: list[dict[str, Any]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "router-comparison.csv", summaries)
    _write_csv(out_dir / "router-evaluation-details.csv", details)
    _write_csv(out_dir / "pareto-quality-cost.csv", [row for row in pareto if row["axis"] == "cost"])
    _write_csv(out_dir / "pareto-quality-latency.csv", [row for row in pareto if row["axis"] == "latency"])
    (out_dir / "verification-report.md").write_text(_report_markdown(summaries), encoding="utf-8", newline="\n")
    (out_dir / "failure-analysis.md").write_text(_failure_markdown(details), encoding="utf-8", newline="\n")


def summarize_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    if not traces:
        raise ValueError("at least one trace is required")
    required_paths = (
        ("trace_id",),
        ("router", "scores"),
        ("graph", "id"),
        ("graph", "version"),
        ("graph", "stop_reason"),
        ("nodes",),
        ("usage", "cost_usd"),
        ("usage", "latency_ms"),
    )
    complete = sum(all(_has_path(trace, path) for path in required_paths) for trace in traces)
    verifier_traces = [trace for trace in traces if any(node.get("role") == "verifier" for node in trace.get("nodes", []))]
    repair_traces = [trace for trace in traces if any(node.get("role") == "repair" for node in trace.get("nodes", []))]
    repair_successes = sum(trace.get("graph", {}).get("stop_reason") == "repaired_and_verified" for trace in repair_traces)
    latencies = [float(trace.get("usage", {}).get("latency_ms", 0.0)) for trace in traces]
    stop_reasons: dict[str, int] = defaultdict(int)
    graph_counts: dict[str, int] = defaultdict(int)
    for trace in traces:
        stop_reasons[str(trace.get("graph", {}).get("stop_reason", "missing"))] += 1
        graph_counts[str(trace.get("graph", {}).get("id", "missing"))] += 1
    return {
        "traces": len(traces),
        "trace_completeness": complete / len(traces),
        "mean_cost": statistics.fmean(float(trace.get("usage", {}).get("cost_usd", 0.0)) for trace in traces),
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "verifier_trace_rate": len(verifier_traces) / len(traces),
        "verifier_pass_rate": (
            sum(trace.get("graph", {}).get("stop_reason") in {"verifier_passed", "repaired_and_verified"} for trace in verifier_traces)
            / len(verifier_traces)
            if verifier_traces
            else 0.0
        ),
        "repair_trigger_rate": len(repair_traces) / len(verifier_traces) if verifier_traces else 0.0,
        "repair_success_rate": repair_successes / len(repair_traces) if repair_traces else 0.0,
        "average_loop_count": statistics.fmean(
            sum(node.get("role") == "repair" for node in trace.get("nodes", [])) for trace in traces
        ),
        "graph_counts": dict(sorted(graph_counts.items())),
        "stop_reasons": dict(sorted(stop_reasons.items())),
    }


def write_trace_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Orchestration Trace Report",
        "",
        f"- Traces: {summary['traces']}",
        f"- Trace completeness: {summary['trace_completeness']:.1%}",
        f"- Mean cost: ${summary['mean_cost']:.6f}",
        f"- Latency P50 / P95: {summary['latency_p50_ms']:.0f} / {summary['latency_p95_ms']:.0f} ms",
        f"- Verifier pass rate: {summary['verifier_pass_rate']:.1%}",
        f"- Repair trigger / success rate: {summary['repair_trigger_rate']:.1%} / {summary['repair_success_rate']:.1%}",
        f"- Average repair loops: {summary['average_loop_count']:.3f}",
        "",
        "## Graphs",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in summary["graph_counts"].items())
    lines.extend(["", "## Stop Reasons", ""])
    lines.extend(f"- `{name}`: {count}" for name, count in summary["stop_reasons"].items())
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _has_path(value: dict[str, Any], path: tuple[str, ...]) -> bool:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    return current is not None


def _validate_candidate_records(records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for record in records:
        query_id = _query_id(record)
        if not query_id or query_id in seen:
            raise ValueError(f"query ids must be non-empty and unique: {query_id!r}")
        seen.add(query_id)
        results = record.get("candidate_results")
        if not isinstance(results, list) or not results:
            raise ValueError(f"candidate_results must be non-empty: {query_id}")
        ids = set()
        for result in results:
            candidate_id = str(result.get("candidate_id", ""))
            if not candidate_id or candidate_id in ids:
                raise ValueError(f"candidate ids must be non-empty and unique: {query_id}/{candidate_id}")
            ids.add(candidate_id)
            for field in ("quality", "cost", "latency_ms"):
                value = float(result.get(field, 0.0))
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"invalid {field}: {query_id}/{candidate_id}")


def _query_id(record: dict[str, Any]) -> str:
    return str(record.get("query_id") or record.get("question_id") or "")


def _oracle(results: dict[str, dict[str, Any]], tolerance: float) -> dict[str, Any]:
    best_quality = max(float(item["quality"]) for item in results.values())
    close = [item for item in results.values() if best_quality - float(item["quality"]) <= tolerance]
    return min(close, key=lambda item: (float(item["cost"]), float(item["latency_ms"]), str(item["candidate_id"])))


def _candidate_averages(
    records: list[dict[str, Any]],
    candidate_type: str | None = None,
) -> dict[str, dict[str, float]]:
    values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for result in record["candidate_results"]:
            if candidate_type and result.get("candidate_type", "model") != candidate_type:
                continue
            values[str(result["candidate_id"])].append(result)
    complete = {candidate_id: rows for candidate_id, rows in values.items() if len(rows) == len(records)}
    return {
        candidate_id: {
            "quality": statistics.fmean(float(row["quality"]) for row in rows),
            "cost": statistics.fmean(float(row["cost"]) for row in rows),
            "latency": statistics.fmean(float(row["latency_ms"]) for row in rows),
        }
        for candidate_id, rows in complete.items()
    }


def _constant_selector(candidate_id: str) -> Callable[[dict[str, Any]], str]:
    return lambda record: candidate_id


def _random_selector(seed: int) -> Callable[[dict[str, Any]], str]:
    def select(record: dict[str, Any]) -> str:
        rng = random.Random(f"{seed}:{_query_id(record)}")
        return str(rng.choice(record["candidate_results"])["candidate_id"])

    return select


def _rule_selector(record: dict[str, Any]) -> str:
    labels = record.get("domain_labels") or [record.get("gold_label", "General")]
    candidate_ids = {str(item["candidate_id"]) for item in record["candidate_results"]}
    for label in labels:
        model = LABEL_TO_MODEL.get(str(label))
        if model in candidate_ids:
            return model
    return "general-fallback" if "general-fallback" in candidate_ids else sorted(candidate_ids)[0]


def _graph_selector(record: dict[str, Any]) -> str:
    scores = record.get("router_scores")
    if not isinstance(scores, dict):
        raise ValueError("router_scores are required")
    text = str(record.get("query") or record.get("text") or "")
    risk = str(record.get("risk_level", "auto"))
    decision = GraphSelector().select(text, RouterSignal(scores=scores), risk)
    candidate_ids = {str(item["candidate_id"]) for item in record["candidate_results"]}
    matches = sorted(item for item in candidate_ids if item == decision.graph_id or item.startswith(decision.graph_id + "-v"))
    if matches:
        return matches[-1]
    if decision.graph_id == "single_specialist":
        model = LABEL_TO_MODEL[decision.primary_labels[0]]
        if model in candidate_ids:
            return model
    raise KeyError(f"no candidate for selected graph {decision.graph_id}")


def _detail(
    router_id: str,
    record: dict[str, Any],
    selected: dict[str, Any],
    oracle: dict[str, Any],
    cfg: EvaluationConfig,
    router_cost: float = 0.0,
    router_latency_ms: float = 0.0,
) -> dict[str, Any]:
    quality = float(selected["quality"])
    oracle_quality = float(oracle["quality"])
    regret = max(0.0, oracle_quality - quality)
    candidate_type = str(selected.get("candidate_type", "model"))
    selected_id = str(selected["candidate_id"])
    oracle_id = str(oracle["candidate_id"])
    model_results = [item for item in record["candidate_results"] if item.get("candidate_type", "model") == "model"]
    best_model_quality = max((float(item["quality"]) for item in model_results), default=0.0)
    selected_multi = selected_id.split("-v", 1)[0] in MULTI_AGENT_GRAPHS
    oracle_multi = oracle_id.split("-v", 1)[0] in MULTI_AGENT_GRAPHS
    return {
        "router_id": router_id,
        "query_id": _query_id(record),
        "selected_candidate_id": selected_id,
        "oracle_candidate_id": oracle_id,
        "quality": quality,
        "oracle_quality": oracle_quality,
        "regret": regret,
        "router_cost": router_cost,
        "router_latency_ms": router_latency_ms,
        "cost": float(selected["cost"]) + router_cost,
        "latency_ms": float(selected["latency_ms"]) + router_latency_ms,
        "routing_correct": selected_id == oracle_id,
        "success": quality >= cfg.success_quality,
        "safety_violation": bool(selected.get("safety_violation", False)),
        "unnecessary_multi_agent": selected_multi and oracle_quality - best_model_quality <= cfg.oracle_quality_tolerance,
        "missed_collaboration": oracle_multi and candidate_type == "model" and regret > cfg.missed_collaboration_regret,
    }


def _summaries(details: list[dict[str, Any]], total_queries: int, cfg: EvaluationConfig) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[str(row["router_id"])].append(row)
    summaries = []
    for router_id, rows in grouped.items():
        latencies = [float(row["latency_ms"]) for row in rows]
        successes = [row for row in rows if row["success"]]
        summaries.append(
            {
                "router_id": router_id,
                "coverage": len(rows) / total_queries,
                "queries": len(rows),
                "quality": statistics.fmean(float(row["quality"]) for row in rows),
                "routing_accuracy": statistics.fmean(float(row["routing_correct"]) for row in rows),
                "mean_regret": statistics.fmean(float(row["regret"]) for row in rows),
                "mean_cost": statistics.fmean(float(row["cost"]) for row in rows),
                "cost_per_success": sum(float(row["cost"]) for row in rows) / len(successes) if successes else None,
                "latency_p50_ms": _percentile(latencies, 0.50),
                "latency_p95_ms": _percentile(latencies, 0.95),
                "task_success_rate": len(successes) / len(rows),
                "safety_violations": sum(int(row["safety_violation"]) for row in rows),
                "unnecessary_multi_agent_rate": statistics.fmean(float(row["unnecessary_multi_agent"]) for row in rows),
                "missed_collaboration_rate": statistics.fmean(float(row["missed_collaboration"]) for row in rows),
            }
        )
    return sorted(summaries, key=lambda row: str(row["router_id"]))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _pareto_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for axis, metric in (("cost", "mean_cost"), ("latency", "latency_p95_ms")):
        for candidate in summaries:
            dominated = any(
                other is not candidate
                and float(other["quality"]) >= float(candidate["quality"])
                and float(other[metric]) <= float(candidate[metric])
                and (
                    float(other["quality"]) > float(candidate["quality"])
                    or float(other[metric]) < float(candidate[metric])
                )
                for other in summaries
            )
            rows.append(
                {
                    "axis": axis,
                    "router_id": candidate["router_id"],
                    "quality": candidate["quality"],
                    metric: candidate[metric],
                    "pareto_efficient": not dominated,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _report_markdown(summaries: list[dict[str, Any]]) -> str:
    lines = [
        "# Model Router Verification Report",
        "",
        "| Router | Coverage | Quality | Routing Acc | Mean Regret | Mean Cost | P95 Latency | Success | Safety |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['router_id']} | {row['coverage']:.1%} | {row['quality']:.3f} | "
            f"{row['routing_accuracy']:.3f} | {row['mean_regret']:.3f} | ${row['mean_cost']:.4f} | "
            f"{row['latency_p95_ms']:.0f} ms | {row['task_success_rate']:.1%} | {row['safety_violations']} |"
        )
    lines.extend(["", "Oracle ties use the lower-cost candidate when quality differs by at most 0.02.", ""])
    return "\n".join(lines)


def _failure_markdown(details: list[dict[str, Any]]) -> str:
    failures = sorted(
        [row for row in details if float(row["regret"]) > 0 or row["safety_violation"]],
        key=lambda row: (-float(row["regret"]), str(row["router_id"]), str(row["query_id"])),
    )
    lines = ["# Failure Analysis", ""]
    if not failures:
        return "# Failure Analysis\n\nNo routing regret or safety violations were observed.\n"
    lines.extend([
        "| Router | Query | Selected | Oracle | Regret | Safety |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ])
    for row in failures:
        lines.append(
            f"| {row['router_id']} | {row['query_id']} | {row['selected_candidate_id']} | "
            f"{row['oracle_candidate_id']} | {row['regret']:.3f} | {int(row['safety_violation'])} |"
        )
    lines.append("")
    return "\n".join(lines)
