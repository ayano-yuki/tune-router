from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from typing import Any, Protocol

from .clients import _chat_completions_url, _post_json
from .constants import LABELS, LABEL_TO_MODEL
from .models import Budget, Delegation, PolicyDecision, RouteDecision, RouterSignal
from .selector import GraphSelector


ALLOWED_GRAPHS = {
    "single_specialist",
    "specialist_with_verifier",
    "parallel_experts",
    "clarify_first",
    "safe_refusal_or_handoff",
}

ORCHESTRATOR_SYSTEM_PROMPT = """You are the learned orchestrator for a pool of specialist language models.
Choose the smallest execution graph that can solve the request safely and with high quality.
Use parallel experts only when multiple domains materially contribute. Require verification for high-risk work.
Return one JSON object and no prose with this shape:
{
  "graph_id": "single_specialist|specialist_with_verifier|parallel_experts|clarify_first|safe_refusal_or_handoff",
  "primary_labels": ["Storage|Network|Coding|Security|Database|General"],
  "selected_labels": ["..."],
  "confidence": 0.0,
  "risk_level": "low|normal|high",
  "reason": "short decision reason",
  "delegations": [{"label": "...", "model": "allowed model alias", "objective": "specific subtask"}],
  "synthesis_strategy": "how to reconcile the delegated outputs or null"
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
    ) -> None:
        self.client = client
        self.fallback = fallback or GraphSelector()
        self.allowed_models = allowed_models or set(LABEL_TO_MODEL.values())

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
        )
        try:
            raw_plan = self.client.generate(messages)
            return self._decision_from_plan(parse_plan(raw_plan), deterministic)
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
    ) -> RouteDecision:
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
) -> list[dict[str, str]]:
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
