from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .constants import LABELS, LABEL_TO_MODEL
from .learned import ALLOWED_GRAPHS, build_plan_messages
from .models import Budget, RouterSignal
from .selector import GraphSelector


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

    return {
        "train": train,
        "dev": dev,
        "preferences": preferences,
        "summary": {
            "format": "tune-orchestrator-ft-v1",
            "train_examples": len(train),
            "dev_examples": len(dev),
            "preference_examples": len(preferences),
            "skipped_examples": len(skipped),
            "config": asdict(cfg),
            "skipped": skipped,
        },
    }


def write_ft_datasets(out_dir: Path, datasets: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "preferences"):
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
    messages = build_plan_messages(
        text=query,
        signal=signal,
        risk_level=risk,
        budget=_budget(record.get("budget")),
        allowed_models=allowed_models,
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

    available_models = {
        str(item["candidate_id"])
        for item in record["candidate_results"]
        if str(item.get("candidate_type", "model")) == "model"
    }
    delegations = []
    for label in selected:
        default_model = LABEL_TO_MODEL[label]
        model = default_model if default_model in available_models else sorted(available_models)[0]
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
    if not text or not isinstance(router.get("scores"), dict) or graph.get("id") not in ALLOWED_GRAPHS:
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
    )
    return {
        "messages": [
            *messages,
            {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))},
        ],
        "metadata": {"query_id": trace.get("trace_id"), "source": "approved_execution_trace"},
    }


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
