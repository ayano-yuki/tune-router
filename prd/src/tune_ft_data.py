from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tune_composition import GENERATED_GRAPH_ID
from tune_constants import LABELS, LABEL_TO_MODEL
from tune_learned import ALLOWED_GRAPHS, build_plan_messages
from tune_models import Budget, RouterSignal
from tune_selector import GraphSelector


@dataclass(frozen=True)
class FTDataConfig:
    quality_tolerance: float = 0.02
    cost_weight: float = 0.25
    latency_weight: float = 0.001
    dev_ratio: float = 0.1
    seed: int = 42


def build_ft_datasets(
    candidate_records: list[dict[str, Any]],
    traces: list[dict[str, Any]] | None = None,
    config: FTDataConfig | None = None,
) -> dict[str, Any]:
    cfg = config or FTDataConfig()
    train: list[dict[str, Any]] = []
    dev: list[dict[str, Any]] = []
    preferences: list[dict[str, Any]] = []
    trajectory_preferences: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for record in candidate_records:
        query_id = str(record.get("query_id") or record.get("question_id") or "")
        try:
            example, preference = _candidate_example(record, cfg)
        except ValueError as exc:
            skipped.append({"query_id": query_id, "reason": str(exc)})
            continue
        target = dev if _is_dev(query_id, cfg) else train
        target.append(example)
        if preference:
            preferences.append(preference)

    for trace in traces or []:
        example = _approved_trace_example(trace)
        if example is None:
            continue
        query_id = str(example["metadata"]["query_id"])
        target = dev if _is_dev(query_id, cfg) else train
        target.append(example)
    trajectory_preferences.extend(_trajectory_preferences(traces or []))

    return {
        "train": train,
        "dev": dev,
        "preferences": preferences,
        "trajectory_preferences": trajectory_preferences,
        "summary": {
            "format": "tune-orchestrator-ft-v1",
            "train_examples": len(train),
            "dev_examples": len(dev),
            "preference_examples": len(preferences),
            "trajectory_preference_examples": len(trajectory_preferences),
            "skipped_examples": len(skipped),
            "config": asdict(cfg),
            "skipped": skipped,
        },
    }


def write_ft_datasets(out_dir: Path, datasets: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "preferences", "trajectory_preferences"):
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in datasets[split]:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    (out_dir / "metadata.json").write_text(
        json.dumps(datasets["summary"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


def _candidate_example(
    record: dict[str, Any],
    cfg: FTDataConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    query_id = str(record.get("query_id") or record.get("question_id") or "")
    query = str(record.get("query") or record.get("text") or "").strip()
    scores = record.get("router_scores")
    results = record.get("candidate_results")
    if not query_id or not query or not isinstance(scores, dict) or not isinstance(results, list):
        raise ValueError("query_id, query, router_scores, and candidate_results are required")
    signal = RouterSignal(scores={str(key): float(value) for key, value in scores.items()})
    requested_risk = str(record.get("risk_level", "auto"))
    policy_decision = GraphSelector().select(query, signal, requested_risk)
    risk = policy_decision.policy.risk_level

    candidates = [item for item in results if not item.get("safety_violation", False)]
    if policy_decision.policy.action == "restrict":
        candidates = [item for item in candidates if _candidate_graph(item) == "safe_refusal_or_handoff"]
    elif risk == "high":
        candidates = [
            item
            for item in candidates
            if _candidate_graph(item) in {"specialist_with_verifier", "clarify_first", "safe_refusal_or_handoff"}
        ]
    if not candidates:
        raise ValueError("no policy-compliant candidates")

    ranked = sorted(candidates, key=lambda item: _preference_key(item, candidates, cfg))
    chosen = ranked[0]
    chosen_plan = _plan_for_candidate(record, chosen, risk)
    allowed_models = {
        str(item["candidate_id"])
        for item in results
        if str(item.get("candidate_type", "model")) == "model"
    } or set(LABEL_TO_MODEL.values())
    model_catalog = _model_catalog_for_record(record, results, allowed_models)
    messages = build_plan_messages(
        text=query,
        signal=signal,
        risk_level=risk,
        budget=_budget(record.get("budget")),
        allowed_models=allowed_models,
        model_catalog=model_catalog,
    )
    assistant = json.dumps(chosen_plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    example = {
        "messages": [*messages, {"role": "assistant", "content": assistant}],
        "metadata": {
            "query_id": query_id,
            "source": "offline_outcome_oracle",
            "chosen_candidate_id": chosen["candidate_id"],
            "quality": float(chosen.get("quality", 0.0)),
            "cost": float(chosen.get("cost", 0.0)),
            "latency_ms": float(chosen.get("latency_ms", 0.0)),
            "utility": _utility(chosen, cfg),
        },
    }

    rejected = next(
        (item for item in ranked[1:] if _candidate_graph(item) != chosen_plan["graph_id"] or item["candidate_id"] != chosen["candidate_id"]),
        None,
    )
    preference = None
    if rejected:
        rejected_plan = _plan_for_candidate(record, rejected, risk)
        preference = {
            "prompt": messages,
            "chosen": assistant,
            "rejected": json.dumps(rejected_plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            "metadata": {
                "query_id": query_id,
                "chosen_candidate_id": chosen["candidate_id"],
                "rejected_candidate_id": rejected["candidate_id"],
                "utility_margin": _utility(chosen, cfg) - _utility(rejected, cfg),
            },
        }
    return example, preference


def _preference_key(
    candidate: dict[str, Any],
    candidates: list[dict[str, Any]],
    cfg: FTDataConfig,
) -> tuple[float, float, float, str]:
    best_quality = max(float(item.get("quality", 0.0)) for item in candidates)
    outside_tolerance = max(0.0, best_quality - float(candidate.get("quality", 0.0)) - cfg.quality_tolerance)
    return (
        outside_tolerance,
        cfg.cost_weight * float(candidate.get("cost", 0.0))
        + cfg.latency_weight * float(candidate.get("latency_ms", 0.0)) / 1000,
        -float(candidate.get("quality", 0.0)),
        str(candidate.get("candidate_id", "")),
    )


def _utility(candidate: dict[str, Any], cfg: FTDataConfig) -> float:
    return (
        float(candidate.get("quality", 0.0))
        - cfg.cost_weight * float(candidate.get("cost", 0.0))
        - cfg.latency_weight * float(candidate.get("latency_ms", 0.0)) / 1000
    )


def _plan_for_candidate(record: dict[str, Any], candidate: dict[str, Any], risk: str) -> dict[str, Any]:
    graph_id = _candidate_graph(candidate)
    if graph_id not in ALLOWED_GRAPHS:
        raise ValueError(f"candidate cannot be mapped to an allowed graph: {candidate.get('candidate_id')}")
    scores = {str(key): float(value) for key, value in record["router_scores"].items()}
    ranked_labels = [label for label, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0])) if label in LABELS]
    domain_labels = [str(label) for label in record.get("domain_labels", ()) if str(label) in LABELS]
    primary = domain_labels[:1] or ranked_labels[:1] or ["General"]

    if graph_id == "single_specialist":
        candidate_label = next(
            (label for label, model in LABEL_TO_MODEL.items() if model == str(candidate["candidate_id"])),
            primary[0],
        )
        selected = [candidate_label]
        primary = [candidate_label]
    elif graph_id == "parallel_experts":
        selected = list(dict.fromkeys(domain_labels + ranked_labels))[:3]
        if len(selected) < 2:
            selected = ranked_labels[:2]
    else:
        selected = primary

    available_models = _available_models(record["candidate_results"])
    catalog_by_alias = {
        str(item["alias"]): item
        for item in _model_catalog_for_record(record, record["candidate_results"], available_models)
    }
    delegations = []
    for label in selected:
        default_model = LABEL_TO_MODEL[label]
        model = _model_for_label(label, available_models, catalog_by_alias, default_model)
        delegations.append(
            {
                "label": label,
                "model": model,
                "objective": f"Investigate the {label} aspects, provide evidence, and identify safe next actions.",
            }
        )
    return {
        "graph_id": graph_id,
        "primary_labels": primary,
        "selected_labels": selected,
        "confidence": max(scores.values(), default=0.5),
        "risk_level": risk,
        "reason": f"Outcome supervision preferred {candidate['candidate_id']} for this request.",
        "delegations": delegations,
        "synthesis_strategy": (
            "Reconcile expert conflicts and produce a prioritized evidence-based answer."
            if graph_id == "parallel_experts"
            else None
        ),
    }


def _candidate_graph(candidate: dict[str, Any]) -> str:
    candidate_id = re.sub(r"-v\d+(?:\.\d+)*$", "", str(candidate.get("candidate_id", "")))
    if str(candidate.get("candidate_type", "model")) == "model":
        return "single_specialist"
    return candidate_id


def _approved_trace_example(trace: dict[str, Any]) -> dict[str, Any] | None:
    evaluation = trace.get("evaluation", {})
    rating = evaluation.get("user_rating")
    approved = evaluation.get("review_label") in {"approved", "success", "preferred"}
    if not approved and not (isinstance(rating, (int, float)) and rating >= 4):
        return None
    text = str(trace.get("input", {}).get("text", "")).strip()
    router = trace.get("router", {})
    graph = trace.get("graph", {})
    if not text or not isinstance(router.get("scores"), dict):
        return None
    if graph.get("id") == GENERATED_GRAPH_ID:
        generated = graph.get("generated_graph")
        if not isinstance(generated, dict):
            return None
        plan = {
            **generated,
            "primary_labels": list(router.get("primary_labels") or [max(router["scores"], key=router["scores"].get)]),
            "selected_labels": list(dict.fromkeys(list(router.get("primary_labels", ())) + list(router.get("secondary_labels", ())))),
            "confidence": float(router.get("confidence", max(router["scores"].values()))),
            "risk_level": str(trace.get("policy", {}).get("risk_level", "normal")),
            "reason": str(graph.get("selection_reason", "successful reviewed bounded graph execution")),
        }
        if not plan["selected_labels"]:
            plan["selected_labels"] = plan["primary_labels"]
        messages = build_plan_messages(
            text,
            RouterSignal(scores=router["scores"]),
            plan["risk_level"],
            Budget(),
            _available_models_from_generated_graph(generated),
            model_catalog=None,
        )
        return {
            "messages": [
                *messages,
                {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))},
            ],
            "metadata": {"query_id": trace.get("trace_id"), "source": "approved_execution_trace"},
        }
    if graph.get("id") not in ALLOWED_GRAPHS:
        return None
    primary = list(router.get("primary_labels") or [max(router["scores"], key=router["scores"].get)])
    selected = list(dict.fromkeys(primary + list(router.get("secondary_labels", ()))))[:3]
    plan = {
        "graph_id": graph["id"],
        "primary_labels": primary,
        "selected_labels": selected,
        "confidence": float(router.get("confidence", max(router["scores"].values()))),
        "risk_level": str(trace.get("policy", {}).get("risk_level", "normal")),
        "reason": str(graph.get("selection_reason", "successful reviewed execution")),
        "delegations": graph.get("delegations") or [
            {
                "label": label,
                "model": LABEL_TO_MODEL[label],
                "objective": f"Analyze the {label} aspects and provide evidence-backed findings.",
            }
            for label in selected
        ],
        "synthesis_strategy": graph.get("synthesis_strategy"),
    }
    messages = build_plan_messages(
        text,
        RouterSignal(scores=router["scores"]),
        plan["risk_level"],
        Budget(),
        set(LABEL_TO_MODEL.values()),
        model_catalog=None,
    )
    return {
        "messages": [
            *messages,
            {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))},
        ],
        "metadata": {"query_id": trace.get("trace_id"), "source": "approved_execution_trace"},
    }


def _trajectory_preferences(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        candidate = _reviewed_trace_candidate(trace)
        if candidate is None:
            continue
        grouped.setdefault(candidate["query_key"], []).append(candidate)

    preferences: list[dict[str, Any]] = []
    for query_key, candidates in grouped.items():
        if len(candidates) < 2:
            continue
        ranked = sorted(candidates, key=lambda item: (-float(item["score"]), float(item["cost"]), float(item["latency_ms"]), str(item["trace_id"])))
        chosen = ranked[0]
        rejected = next((item for item in reversed(ranked) if float(chosen["score"]) > float(item["score"])), None)
        if rejected is None:
            continue
        preferences.append(
            {
                "prompt": chosen["messages"],
                "chosen": json.dumps(chosen["plan"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "rejected": json.dumps(rejected["plan"], ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                "metadata": {
                    "source": "reviewed_execution_trajectory",
                    "query_key": query_key,
                    "chosen_trace_id": chosen["trace_id"],
                    "rejected_trace_id": rejected["trace_id"],
                    "chosen_score": chosen["score"],
                    "rejected_score": rejected["score"],
                    "chosen_trajectory": chosen["trajectory"],
                    "rejected_trajectory": rejected["trajectory"],
                },
            }
        )
    return preferences


def _reviewed_trace_candidate(trace: dict[str, Any]) -> dict[str, Any] | None:
    score = _review_score(trace)
    if score is None:
        return None
    text = str(trace.get("input", {}).get("text", "")).strip()
    router = trace.get("router", {})
    if not text or not isinstance(router.get("scores"), dict):
        return None
    risk = str(trace.get("policy", {}).get("risk_level", "normal"))
    if risk == "high" and not any(node.get("role") == "verifier" for node in trace.get("nodes", [])):
        return None
    plan = _trace_plan(trace)
    if plan is None:
        return None
    messages = build_plan_messages(
        text,
        RouterSignal(scores=router["scores"]),
        risk,
        _trace_budget(trace),
        _available_models_from_plan(plan),
        model_catalog=None,
    )
    usage = trace.get("usage", {})
    return {
        "trace_id": str(trace.get("trace_id", "")),
        "query_key": str(trace.get("evaluation", {}).get("query_id") or _stable_query_key(text)),
        "score": score,
        "cost": float(usage.get("cost_usd", 0.0)),
        "latency_ms": float(usage.get("latency_ms", 0.0)),
        "messages": messages,
        "plan": plan,
        "trajectory": _trajectory_summary(trace),
    }


def _review_score(trace: dict[str, Any]) -> float | None:
    evaluation = trace.get("evaluation", {})
    rating = evaluation.get("user_rating")
    if isinstance(rating, (int, float)):
        return float(rating)
    label = str(evaluation.get("review_label") or "").lower()
    if label in {"preferred", "approved", "success"}:
        return 1.0
    if label in {"rejected", "failed", "failure"}:
        return 0.0
    return None


def _trace_plan(trace: dict[str, Any]) -> dict[str, Any] | None:
    router = trace.get("router", {})
    graph = trace.get("graph", {})
    if graph.get("id") == GENERATED_GRAPH_ID:
        generated = graph.get("generated_graph")
        if not isinstance(generated, dict):
            return None
        primary = list(router.get("primary_labels") or [max(router["scores"], key=router["scores"].get)])
        selected = list(dict.fromkeys(primary + list(router.get("secondary_labels", ()))))[:3]
        return {
            **generated,
            "primary_labels": primary,
            "selected_labels": selected or primary,
            "confidence": float(router.get("confidence", max(router["scores"].values()))),
            "risk_level": str(trace.get("policy", {}).get("risk_level", "normal")),
            "reason": str(graph.get("selection_reason", "reviewed bounded graph execution")),
        }
    if graph.get("id") not in ALLOWED_GRAPHS:
        return None
    primary = list(router.get("primary_labels") or [max(router["scores"], key=router["scores"].get)])
    selected = list(dict.fromkeys(primary + list(router.get("secondary_labels", ()))))[:3]
    return {
        "graph_id": graph["id"],
        "primary_labels": primary,
        "selected_labels": selected,
        "confidence": float(router.get("confidence", max(router["scores"].values()))),
        "risk_level": str(trace.get("policy", {}).get("risk_level", "normal")),
        "reason": str(graph.get("selection_reason", "reviewed execution trajectory")),
        "delegations": graph.get("delegations") or [
            {
                "label": label,
                "model": LABEL_TO_MODEL[label],
                "objective": f"Analyze the {label} aspects and provide evidence-backed findings.",
            }
            for label in selected
        ],
        "synthesis_strategy": graph.get("synthesis_strategy"),
    }


def _trace_budget(trace: dict[str, Any]) -> Budget:
    budget = trace.get("budget")
    if isinstance(budget, dict):
        return _budget(budget)
    usage = trace.get("usage", {})
    return Budget(
        max_cost_usd=max(1.0, float(usage.get("cost_usd", 0.0))),
        max_latency_ms=max(60_000, int(float(usage.get("latency_ms", 0.0)))),
        max_steps=max(12, int(usage.get("steps", 0))),
    )


def _available_models_from_plan(plan: dict[str, Any]) -> set[str]:
    if plan.get("plan_type") == GENERATED_GRAPH_ID:
        return _available_models_from_generated_graph(plan)
    aliases = {
        str(item.get("model"))
        for item in plan.get("delegations", [])
        if isinstance(item, dict) and item.get("model")
    }
    if plan.get("synthesis_strategy"):
        aliases.add("general-synthesizer")
    return aliases or set(LABEL_TO_MODEL.values())


def _trajectory_summary(trace: dict[str, Any]) -> dict[str, Any]:
    graph = trace.get("graph", {})
    usage = trace.get("usage", {})
    return {
        "graph_id": graph.get("id"),
        "selector_type": graph.get("selector_type"),
        "stop_reason": graph.get("stop_reason"),
        "failure_type": trace.get("evaluation", {}).get("failure_type"),
        "failure_detail": trace.get("evaluation", {}).get("failure_detail"),
        "nodes": [
            {
                "id": node.get("id"),
                "role": node.get("role"),
                "model": node.get("model"),
                "status": node.get("status"),
                "attempts": node.get("attempts"),
            }
            for node in trace.get("nodes", [])
            if isinstance(node, dict)
        ],
        "usage": {
            "steps": usage.get("steps"),
            "cost_usd": usage.get("cost_usd"),
            "latency_ms": usage.get("latency_ms"),
        },
    }


def _stable_query_key(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"text:{digest}"


def _budget(value: Any) -> Budget:
    if not isinstance(value, dict):
        return Budget()
    return Budget(
        max_cost_usd=float(value.get("max_cost_usd", 1.0)),
        max_latency_ms=int(value.get("max_latency_ms", 60_000)),
        max_steps=int(value.get("max_steps", 12)),
    )


def _is_dev(query_id: str, cfg: FTDataConfig) -> bool:
    digest = hashlib.sha256(f"{cfg.seed}:{query_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    return bucket < cfg.dev_ratio


def _available_models(results: list[dict[str, Any]]) -> set[str]:
    return {
        str(item["candidate_id"])
        for item in results
        if str(item.get("candidate_type", "model")) == "model"
    } or set(LABEL_TO_MODEL.values())


def _available_models_from_generated_graph(generated: dict[str, Any]) -> set[str]:
    aliases = {
        str(node.get("model"))
        for node in generated.get("nodes", [])
        if isinstance(node, dict) and node.get("model")
    }
    return aliases or set(LABEL_TO_MODEL.values())


def _model_catalog_for_record(
    record: dict[str, Any],
    results: list[dict[str, Any]],
    allowed_models: set[str],
) -> list[dict[str, Any]]:
    raw_catalog = record.get("model_catalog")
    if isinstance(raw_catalog, list):
        catalog = [
            {**item, "alias": str(item.get("alias", ""))}
            for item in raw_catalog
            if isinstance(item, dict) and str(item.get("alias", "")) in allowed_models
        ]
        known = {str(item["alias"]) for item in catalog}
    else:
        catalog = []
        known = set()

    for result in results:
        if str(result.get("candidate_type", "model")) != "model":
            continue
        alias = str(result.get("candidate_id", ""))
        if not alias or alias in known:
            continue
        item: dict[str, Any] = {"alias": alias}
        for key in ("domains", "strengths", "context_window", "latency_tier", "cost_tier", "safety_profile", "supports_json"):
            if key in result:
                item[key] = result[key]
        if "domains" not in item:
            inferred = _label_for_model_alias(alias)
            if inferred:
                item["domains"] = [inferred]
        catalog.append(item)
        known.add(alias)

    catalog.extend({"alias": alias, "domains": []} for alias in sorted(allowed_models - known))
    return sorted(catalog, key=lambda item: str(item["alias"]))


def _model_for_label(
    label: str,
    available_models: set[str],
    catalog_by_alias: dict[str, dict[str, Any]],
    default_model: str,
) -> str:
    if default_model in available_models and _catalog_supports(catalog_by_alias.get(default_model), label):
        return default_model
    compatible = sorted(
        (
            alias
            for alias in available_models
            if _catalog_supports(catalog_by_alias.get(alias), label)
        ),
        key=lambda alias: (
            0 if _catalog_has_exact_domain(catalog_by_alias.get(alias), label) else 1,
            alias,
        ),
    )
    if compatible:
        return compatible[0]
    return sorted(available_models)[0]


def _catalog_supports(item: dict[str, Any] | None, label: str) -> bool:
    if not item:
        return True
    domains = item.get("domains")
    if isinstance(domains, str):
        domains = [domains]
    if not domains:
        return True
    return "*" in domains or label in domains


def _catalog_has_exact_domain(item: dict[str, Any] | None, label: str) -> bool:
    if not item:
        return False
    domains = item.get("domains")
    if isinstance(domains, str):
        domains = [domains]
    return label in (domains or [])


def _label_for_model_alias(alias: str) -> str | None:
    return next((label for label, model in LABEL_TO_MODEL.items() if model == alias), None)
