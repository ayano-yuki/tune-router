from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from composition import graph_from_bounded_plan
from executor import GraphExecutor
from graphs import GraphDefinition
from models import Budget, RouteDecision, RouterSignal
from selector import GraphSelector, SelectionPolicy


@dataclass(frozen=True)
class ShadowConfig:
    mode: str = "off"
    max_count: int = 1
    max_cost_usd: float = 0.0
    max_latency_ms: int = 30_000
    low_risk_only: bool = True


def build_shadow_decisions(
    *,
    text: str,
    signal: RouterSignal,
    served_decision: RouteDecision,
    risk_level: str,
    selector: GraphSelector,
    config: ShadowConfig,
) -> list[tuple[str, RouteDecision]]:
    if config.mode == "off" or config.max_count <= 0:
        return []
    if served_decision.policy.action != "allow":
        return []
    if config.low_risk_only and served_decision.policy.risk_level == "high":
        return []

    candidates: list[tuple[str, RouteDecision]] = []
    deterministic = selector.select(text, signal, risk_level)
    if _materially_different(served_decision, deterministic):
        candidates.append(("deterministic_baseline", deterministic))

    if config.mode == "alternatives":
        candidates.extend(_alternative_decisions(text, signal, risk_level, served_decision, selector))

    deduped: list[tuple[str, RouteDecision]] = []
    seen: set[tuple[Any, ...]] = set()
    for reason, decision in candidates:
        key = _decision_key(decision)
        if key in seen or key == _decision_key(served_decision):
            continue
        seen.add(key)
        deduped.append((reason, decision))
        if len(deduped) >= config.max_count:
            break
    return deduped


def execute_shadow_decisions(
    *,
    text: str,
    signal: RouterSignal,
    model_client: Any,
    graphs: dict[str, GraphDefinition],
    shadows: list[tuple[str, RouteDecision]],
    config: ShadowConfig,
    max_workers: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for reason, decision in shadows:
        try:
            graph = graph_from_bounded_plan(decision.generated_graph) if decision.generated_graph else graphs[decision.graph_id]
            result = GraphExecutor(model_client, max_workers=max_workers).execute(
                text=text,
                signal=signal,
                decision=decision,
                graph=graph,
                budget=Budget(
                    max_cost_usd=config.max_cost_usd,
                    max_latency_ms=config.max_latency_ms,
                    max_steps=min(12, graph.max_steps),
                ),
            )
            results.append(
                {
                    "reason": reason,
                    "status": "completed",
                    "decision": decision.to_dict(),
                    "trace": result.trace,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "reason": reason,
                    "status": "failed",
                    "decision": decision.to_dict(),
                    "error": str(exc),
                }
            )
    return results


def _alternative_decisions(
    text: str,
    signal: RouterSignal,
    risk_level: str,
    served_decision: RouteDecision,
    selector: GraphSelector,
) -> list[tuple[str, RouteDecision]]:
    del selector
    alternatives: list[tuple[str, RouteDecision]] = []
    if served_decision.graph_id == "single_specialist":
        parallel = GraphSelector(
            SelectionPolicy(
                minimum_confidence=0.0,
                single_specialist_margin=1.0,
                secondary_score=0.05,
                max_selected_labels=3,
            )
        ).select(text, signal, risk_level)
        if parallel.graph_id == "parallel_experts":
            alternatives.append(("forced_parallel_experts", parallel))
    elif served_decision.graph_id == "parallel_experts":
        single = GraphSelector(
            SelectionPolicy(
                minimum_confidence=0.0,
                single_specialist_margin=0.0,
                secondary_score=1.0,
                max_selected_labels=1,
            )
        ).select(text, signal, risk_level)
        if single.graph_id == "single_specialist":
            alternatives.append(("forced_single_specialist", single))
    return alternatives


def _materially_different(left: RouteDecision, right: RouteDecision) -> bool:
    return _decision_key(left) != _decision_key(right)


def _decision_key(decision: RouteDecision) -> tuple[Any, ...]:
    generated = decision.generated_graph
    generated_key = None
    if generated:
        generated_key = (
            generated.get("plan_type"),
            tuple((node.get("id"), node.get("role"), node.get("model"), tuple(node.get("dependencies", ()))) for node in generated.get("nodes", ())),
            generated.get("final_node"),
        )
    return (
        decision.graph_id,
        tuple(decision.selected_labels),
        tuple((item.label, item.model) for item in decision.delegations),
        generated_key,
    )
