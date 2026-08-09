from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from typing import Any, Protocol

from clients import _chat_completions_url, _post_json
from composition import GENERATED_GRAPH_ID, validate_bounded_graph_plan
from constants import LABELS, LABEL_TO_MODEL
from models import Budget, Delegation, PolicyDecision, RouteDecision, RouterSignal
from selector import GraphSelector


ALLOWED_GRAPHS = {
    "single_specialist",
    "specialist_with_verifier",
    "parallel_experts",
    "clarify_first",
    "safe_refusal_or_handoff",
}
DEFAULT_RUNTIME_MODELS = set(LABEL_TO_MODEL.values()) | {"verifier", "general-synthesizer"}

ORCHESTRATOR_SYSTEM_PROMPT = """You are the learned orchestrator for a pool of specialist language models.
Choose the smallest execution graph that can solve the request safely and with high quality.
Use parallel experts only when multiple domains materially contribute. Require verification for high-risk work.
Return one JSON object and no prose.
For normal operation, return this fixed-graph shape:
{
  "graph_id": "single_specialist|specialist_with_verifier|parallel_experts|clarify_first|safe_refusal_or_handoff",
  "primary_labels": ["Storage|Network|Coding|Security|Database|General"],
  "selected_labels": ["..."],
  "confidence": 0.0,
  "risk_level": "low|normal|high",
  "reason": "short decision reason",
  "delegations": [{"label": "...", "model": "allowed model alias", "objective": "specific subtask"}],
  "synthesis_strategy": "how to reconcile the delegated outputs or null"
}
When a fixed graph is materially worse and the request can be solved safely within budget, you may return this bounded graph shape:
{
  "plan_type": "bounded_graph",
  "primary_labels": ["Storage|Network|Coding|Security|Database|General"],
  "selected_labels": ["..."],
  "confidence": 0.0,
  "risk_level": "low|normal|high",
  "reason": "short decision reason",
  "nodes": [{"id": "node_id", "role": "specialist|verifier|repair|synthesizer|clarifier|policy", "label": "optional domain label", "model": "allowed model alias", "dependencies": ["node_id"], "objective": "specific subtask"}],
  "final_node": "node_id"
}"""


class PlanClient(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


class OpenAIPlanClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.url = _chat_completions_url(base_url)
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds

    def generate(self, messages: list[dict[str, str]]) -> str:
        headers: dict[str, str] = {}
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"environment variable is not set: {self.api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"
        response = _post_json(
            self.url,
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 1024,
                "response_format": {"type": "json_object"},
            },
            headers,
            self.timeout_seconds,
        )
        try:
            return str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("orchestrator response does not contain choices[0].message.content") from exc


class LearnedGraphSelector:
    def __init__(
        self,
        client: PlanClient,
        fallback: GraphSelector | None = None,
        allowed_models: set[str] | None = None,
        model_catalog: list[dict[str, Any]] | None = None,
    ) -> None:
        self.client = client
        self.fallback = fallback or GraphSelector()
        self.allowed_models = allowed_models or set(DEFAULT_RUNTIME_MODELS)
        self.model_catalog = _normalized_model_catalog(model_catalog, self.allowed_models)
        self.catalog_by_alias = {str(item["alias"]): item for item in self.model_catalog}

    def select(
        self,
        text: str,
        signal: RouterSignal,
        risk_level: str = "auto",
        budget: Budget | None = None,
    ) -> RouteDecision:
        deterministic = self.fallback.select(text, signal, risk_level)
        if deterministic.policy.action == "restrict":
            return replace(deterministic, selector_type="policy_gate")

        messages = build_plan_messages(
            text=text,
            signal=signal,
            risk_level=deterministic.policy.risk_level,
            budget=budget or Budget(),
            allowed_models=self.allowed_models,
            model_catalog=self.model_catalog,
        )
        try:
            raw_plan = self.client.generate(messages)
            return self._decision_from_plan(parse_plan(raw_plan), deterministic, budget or Budget())
        except Exception as exc:
            return replace(
                deterministic,
                selector_type="deterministic_fallback",
                fallback_reason=f"learned orchestrator rejected: {exc}",
            )

    def _decision_from_plan(
        self,
        plan: dict[str, Any],
        deterministic: RouteDecision,
        budget: Budget,
    ) -> RouteDecision:
        if plan.get("plan_type") == GENERATED_GRAPH_ID:
            return self._decision_from_bounded_graph_plan(plan, deterministic, budget)

        graph_id = str(plan.get("graph_id", ""))
        if graph_id not in ALLOWED_GRAPHS:
            raise ValueError(f"unknown graph_id: {graph_id}")

        selected = _labels(plan.get("selected_labels"), "selected_labels")
        primary = _labels(plan.get("primary_labels"), "primary_labels")
        if not primary:
            primary = selected[:1]
        if not selected:
            selected = primary
        if not primary or not selected:
            raise ValueError("plan must select at least one label")
        if primary[0] not in selected:
            raise ValueError("primary label must be included in selected_labels")
        if graph_id == "single_specialist" and len(selected) != 1:
            raise ValueError("single_specialist requires exactly one selected label")
        if graph_id == "parallel_experts" and len(selected) < 2:
            raise ValueError("parallel_experts requires at least two selected labels")

        risk = str(plan.get("risk_level", deterministic.policy.risk_level))
        if risk not in {"low", "normal", "high"}:
            raise ValueError(f"invalid risk_level: {risk}")
        if deterministic.policy.risk_level == "high":
            risk = "high"
        if risk == "high" and graph_id not in {
            "specialist_with_verifier",
            "safe_refusal_or_handoff",
            "clarify_first",
        }:
            raise ValueError("high-risk plans require verifier, clarification, or policy handoff")

        confidence = float(plan.get("confidence", deterministic.confidence))
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        delegations = self._delegations(plan.get("delegations"), selected)
        return RouteDecision(
            graph_id=graph_id,
            primary_labels=primary,
            secondary_labels=tuple(label for label in selected if label not in primary),
            selected_labels=selected,
            confidence=confidence,
            margin=deterministic.margin,
            reason=str(plan.get("reason") or "learned orchestration plan"),
            policy=PolicyDecision(risk_level=risk, action="allow"),
            delegations=delegations,
            synthesis_strategy=(
                str(plan["synthesis_strategy"])
                if plan.get("synthesis_strategy") is not None
                else None
            ),
            selector_type="learned_orchestrator",
        )

    def _decision_from_bounded_graph_plan(
        self,
        plan: dict[str, Any],
        deterministic: RouteDecision,
        budget: Budget,
    ) -> RouteDecision:
        selected = _labels(plan.get("selected_labels"), "selected_labels")
        primary = _labels(plan.get("primary_labels"), "primary_labels")
        if not primary:
            primary = selected[:1]
        if not selected:
            selected = primary
        if not primary or not selected:
            raise ValueError("bounded graph plan must select at least one label")
        if primary[0] not in selected:
            raise ValueError("primary label must be included in selected_labels")

        risk = str(plan.get("risk_level", deterministic.policy.risk_level))
        if risk not in {"low", "normal", "high"}:
            raise ValueError(f"invalid risk_level: {risk}")
        if deterministic.policy.risk_level == "high":
            risk = "high"
        confidence = float(plan.get("confidence", deterministic.confidence))
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

        generated_graph = validate_bounded_graph_plan(
            plan,
            allowed_models=self.allowed_models,
            catalog_by_alias=self.catalog_by_alias,
            risk_level=risk,
            budget=budget,
        )
        delegations = tuple(
            Delegation(
                label=str(node["label"]),
                model=str(node["model"]),
                objective=str(node.get("objective", "")) or f"Analyze the {node['label']} aspects.",
            )
            for node in generated_graph["nodes"]
            if node.get("label") in selected and node.get("model")
        )
        return RouteDecision(
            graph_id=GENERATED_GRAPH_ID,
            primary_labels=primary,
            secondary_labels=tuple(label for label in selected if label not in primary),
            selected_labels=selected,
            confidence=confidence,
            margin=deterministic.margin,
            reason=str(plan.get("reason") or "learned bounded graph plan"),
            policy=PolicyDecision(risk_level=risk, action="allow"),
            delegations=delegations,
            synthesis_strategy=str(plan.get("synthesis_strategy")) if plan.get("synthesis_strategy") is not None else None,
            selector_type="learned_orchestrator",
            generated_graph=generated_graph,
        )

    def _delegations(self, raw: Any, selected: tuple[str, ...]) -> tuple[Delegation, ...]:
        by_label: dict[str, Delegation] = {}
        if raw is not None and not isinstance(raw, list):
            raise ValueError("delegations must be an array")
        for value in raw or []:
            if not isinstance(value, dict):
                raise ValueError("each delegation must be an object")
            label = str(value.get("label", ""))
            model = str(value.get("model", ""))
            objective = str(value.get("objective", "")).strip()
            if label not in selected:
                raise ValueError(f"delegation label is not selected: {label}")
            if model not in self.allowed_models:
                raise ValueError(f"delegation model is not allowed: {model}")
            if not _supports_label(self.catalog_by_alias.get(model), label):
                raise ValueError(f"delegation model {model} does not support label {label}")
            if not objective:
                raise ValueError(f"delegation objective is empty: {label}")
            by_label[label] = Delegation(label=label, model=model, objective=objective)
        delegations = []
        for label in selected:
            if label in by_label:
                delegations.append(by_label[label])
                continue
            default_model = LABEL_TO_MODEL[label]
            if default_model not in self.allowed_models:
                raise ValueError(f"delegation is missing for {label} and its default model is unavailable")
            if not _supports_label(self.catalog_by_alias.get(default_model), label):
                raise ValueError(f"default model {default_model} does not support label {label}")
            delegations.append(
                Delegation(
                    label=label,
                    model=default_model,
                    objective=f"Analyze the {label} aspects and return evidence-backed findings.",
                )
            )
        return tuple(delegations)


def build_plan_messages(
    text: str,
    signal: RouterSignal,
    risk_level: str,
    budget: Budget,
    allowed_models: set[str],
    model_catalog: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    catalog = _normalized_model_catalog(model_catalog, allowed_models)
    payload = {
        "request": text,
        "router_scores": signal.scores,
        "risk_level": risk_level,
        "budget": {
            "max_cost_usd": budget.max_cost_usd,
            "max_latency_ms": budget.max_latency_ms,
            "max_steps": budget.max_steps,
        },
        "available_graphs": sorted(ALLOWED_GRAPHS),
        "available_models": sorted(allowed_models),
        "model_catalog": catalog,
        "composition_constraints": {
            "enabled_plan_type": GENERATED_GRAPH_ID,
            "allowed_node_roles": ["specialist", "verifier", "repair", "synthesizer", "clarifier", "policy"],
            "max_nodes": min(8, max(1, budget.max_steps)),
            "dag_only": True,
            "high_risk_requires_verifier_on_final_path": True,
            "models_must_come_from_catalog": True,
        },
    }
    return [
        {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def parse_plan(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("orchestration plan must be a JSON object")
    return value


def _labels(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    labels = tuple(dict.fromkeys(str(item) for item in value))
    unknown = set(labels) - set(LABELS)
    if unknown:
        raise ValueError(f"{field} contains unknown labels: {sorted(unknown)}")
    if len(labels) > 3:
        raise ValueError(f"{field} cannot contain more than three labels")
    return labels


def _normalized_model_catalog(
    raw_catalog: list[dict[str, Any]] | None,
    allowed_models: set[str],
) -> list[dict[str, Any]]:
    if raw_catalog is None:
        return [{"alias": alias, "domains": []} for alias in sorted(allowed_models)]
    catalog = []
    seen: set[str] = set()
    for raw in raw_catalog:
        if not isinstance(raw, dict):
            raise ValueError("model_catalog entries must be objects")
        alias = str(raw.get("alias", "")).strip()
        if not alias:
            raise ValueError("model_catalog alias is required")
        if alias not in allowed_models:
            continue
        if alias in seen:
            raise ValueError(f"duplicate model_catalog alias: {alias}")
        seen.add(alias)
        domains = _catalog_strings(raw.get("domains", ()), "domains")
        unknown = set(domains) - set(LABELS) - {"*"}
        if unknown:
            raise ValueError(f"model_catalog {alias} has unknown domains: {sorted(unknown)}")
        strengths = _catalog_strings(raw.get("strengths", ()), "strengths")
        item: dict[str, Any] = {
            "alias": alias,
            "domains": domains,
            "strengths": strengths,
            "supports_json": bool(raw.get("supports_json", False)),
        }
        for key in ("model", "latency_tier", "cost_tier", "safety_profile"):
            if raw.get(key) is not None:
                item[key] = str(raw[key])
        if raw.get("context_window") is not None:
            context_window = int(raw["context_window"])
            if context_window <= 0:
                raise ValueError(f"model_catalog {alias} context_window must be positive")
            item["context_window"] = context_window
        catalog.append(item)
    missing = allowed_models - seen
    catalog.extend({"alias": alias, "domains": []} for alias in sorted(missing))
    return sorted(catalog, key=lambda item: str(item["alias"]))


def _catalog_strings(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list | tuple):
        raise ValueError(f"model_catalog {field} must be a string or array")
    return [str(item) for item in value]


def _supports_label(catalog_item: dict[str, Any] | None, label: str) -> bool:
    if catalog_item is None:
        return True
    domains = catalog_item.get("domains")
    if not domains:
        return True
    return "*" in domains or label in domains
