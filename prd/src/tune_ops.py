from __future__ import annotations

import os
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tune_clients import OpenAIModelClient, OpenAIRouterClient
from tune_constants import LABEL_TO_MODEL
from tune_graphs import GraphDefinition, load_graphs
from tune_learned import LearnedGraphSelector, OpenAIPlanClient, build_plan_messages, parse_plan
from tune_models import Budget, RouterSignal
from tune_selector import GraphSelector


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def overall_status(checks: list[CheckResult]) -> str:
    if any(check.status == "fail" for check in checks):
        return "fail"
    if any(check.status == "warn" for check in checks):
        return "warn"
    return "pass"


def required_model_aliases(graphs: dict[str, GraphDefinition]) -> set[str]:
    aliases = set(LABEL_TO_MODEL.values())
    for graph in graphs.values():
        for node in graph.nodes:
            if node.model:
                aliases.add(node.model)
            aliases.update(node.fallback_models)
    return aliases


def validate_model_config(config: dict[str, Any], graphs: dict[str, GraphDefinition]) -> list[CheckResult]:
    checks: list[CheckResult] = []
    try:
        client = OpenAIModelClient(config)
    except Exception as exc:
        return [CheckResult("model-config", "fail", str(exc))]

    configured = set(client.endpoints)
    required = required_model_aliases(graphs)
    missing = sorted(required - configured)
    extra = sorted(configured - required)
    checks.append(
        CheckResult(
            "model-config.aliases",
            "fail" if missing else "pass",
            "all required model aliases are configured" if not missing else "required model aliases are missing",
            {"required": sorted(required), "configured": sorted(configured), "missing": missing, "extra": extra},
        )
    )

    missing_env = sorted(
        {
            endpoint.api_key_env
            for endpoint in client.endpoints.values()
            if endpoint.api_key_env and not os.environ.get(endpoint.api_key_env)
        }
    )
    checks.append(
        CheckResult(
            "model-config.credentials",
            "fail" if missing_env else "pass",
            "referenced credential environment variables are set" if not missing_env else "credential environment variables are missing",
            {"missing_env": missing_env},
        )
    )
    checks.append(_validate_capability_catalog(client.model_catalog()))
    return checks


def validate_local_orchestrator_adapter(adapter: Path) -> CheckResult:
    if not adapter.exists():
        return CheckResult(
            "learned-orchestrator.adapter",
            "fail",
            "local learned orchestrator adapter directory does not exist",
            {"adapter": str(adapter)},
        )
    if not adapter.is_dir():
        return CheckResult(
            "learned-orchestrator.adapter",
            "fail",
            "local learned orchestrator adapter path is not a directory",
            {"adapter": str(adapter)},
        )
    required = ["adapter_config.json"]
    missing = [name for name in required if not (adapter / name).exists()]
    has_weights = any((adapter / name).exists() for name in ("adapter_model.safetensors", "adapter_model.bin"))
    if not has_weights:
        missing.append("adapter_model.safetensors|adapter_model.bin")
    metadata = {}
    config_path = adapter / "orchestrator_config.json"
    if config_path.exists():
        try:
            metadata["orchestrator_config"] = _read_json_object(config_path)
        except Exception as exc:
            return CheckResult(
                "learned-orchestrator.adapter",
                "fail",
                f"orchestrator_config.json is invalid: {exc}",
                {"adapter": str(adapter)},
            )
    status = "fail" if missing else "pass"
    return CheckResult(
        "learned-orchestrator.adapter",
        status,
        "local learned orchestrator adapter is present" if not missing else "local learned orchestrator adapter is incomplete",
        {"adapter": str(adapter), "missing": missing, **metadata},
    )


def run_preflight(
    *,
    graphs_path: Path,
    router_url: str,
    router_model: str,
    router_timeout: float,
    model_config: dict[str, Any] | None,
    probe_text: str,
    probe_model_endpoints: bool,
    probe_model_chat: bool,
    adapter: Path | None = None,
    orchestrator_url: str | None = None,
    orchestrator_model: str = "tune-orchestrator-ft",
    orchestrator_api_key_env: str | None = None,
    orchestrator_timeout: float = 30.0,
    probe_orchestrator: bool = False,
) -> dict[str, Any]:
    checks: list[CheckResult] = []
    graphs: dict[str, GraphDefinition] = {}
    signal: RouterSignal | None = None
    try:
        graphs = load_graphs(graphs_path)
        checks.append(
            CheckResult(
                "graphs",
                "pass",
                "graph definitions loaded and validated",
                {"count": len(graphs), "graph_ids": sorted(graphs)},
            )
        )
    except Exception as exc:
        checks.append(CheckResult("graphs", "fail", str(exc)))

    try:
        signal = OpenAIRouterClient(router_url, router_model, router_timeout).classify(probe_text)
        decision = GraphSelector().select(probe_text, signal)
        checks.append(
            CheckResult(
                "router",
                "pass",
                "router returned valid scores and selector accepted them",
                {
                    "model": signal.model,
                    "latency_ms": round(signal.latency_ms, 3),
                    "selected_graph": decision.graph_id,
                    "selected_labels": list(decision.selected_labels),
                },
            )
        )
    except Exception as exc:
        checks.append(CheckResult("router", "fail", str(exc), {"router_url": router_url, "router_model": router_model}))

    if model_config is None:
        checks.append(CheckResult("model-config", "warn", "model config was not provided; real model execution cannot be checked"))
    elif graphs:
        checks.extend(validate_model_config(model_config, graphs))
        if probe_model_endpoints or probe_model_chat:
            checks.extend(_probe_models(model_config, graphs, probe_text, probe_model_endpoints, probe_model_chat))

    if adapter is not None:
        checks.append(validate_local_orchestrator_adapter(adapter))

    if orchestrator_url:
        if probe_orchestrator and signal is not None:
            checks.append(
                _probe_orchestrator_endpoint(
                    orchestrator_url=orchestrator_url,
                    orchestrator_model=orchestrator_model,
                    orchestrator_api_key_env=orchestrator_api_key_env,
                    orchestrator_timeout=orchestrator_timeout,
                    probe_text=probe_text,
                    signal=signal,
                    allowed_models=set(model_config.get("models", {})) if model_config else set(LABEL_TO_MODEL.values()),
                    model_catalog=OpenAIModelClient(model_config).model_catalog() if model_config else None,
                )
            )
        else:
            checks.append(
                CheckResult(
                    "learned-orchestrator.endpoint",
                    "warn",
                    "learned orchestrator endpoint was configured but not probed; pass --probe-orchestrator to validate plan generation",
                    {"base_url": orchestrator_url, "model": orchestrator_model},
                )
            )

    return {
        "status": overall_status(checks),
        "checks": [check.to_dict() for check in checks],
    }


def _probe_models(
    config: dict[str, Any],
    graphs: dict[str, GraphDefinition],
    probe_text: str,
    probe_model_endpoints: bool,
    probe_model_chat: bool,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    try:
        client = OpenAIModelClient(config)
    except Exception as exc:
        return [CheckResult("model-probe", "fail", str(exc))]

    for alias in sorted(required_model_aliases(graphs) & set(client.endpoints)):
        endpoint = client.endpoints[alias]
        if probe_model_endpoints:
            try:
                models = client.list_models(alias)
                model_ids = [
                    str(item.get("id"))
                    for item in models.get("data", [])
                    if isinstance(item, dict) and item.get("id") is not None
                ]
                status = "pass" if not model_ids or endpoint.model in model_ids else "warn"
                checks.append(
                    CheckResult(
                        f"model-endpoint.{alias}.models",
                        status,
                        "model endpoint is reachable" if status == "pass" else "endpoint is reachable but configured model was not listed",
                        {"base_url": endpoint.base_url, "model": endpoint.model, "listed_models": model_ids[:20]},
                    )
                )
            except Exception as exc:
                checks.append(
                    CheckResult(
                        f"model-endpoint.{alias}.models",
                        "fail",
                        str(exc),
                        {"base_url": endpoint.base_url, "model": endpoint.model},
                    )
                )
        if probe_model_chat:
            try:
                response = client.complete(
                    alias,
                    [
                        {"role": "system", "content": "Reply with a short health-check acknowledgement."},
                        {"role": "user", "content": probe_text},
                    ],
                    max_tokens=16,
                    timeout_seconds=endpoint.timeout_seconds,
                )
                checks.append(
                    CheckResult(
                        f"model-endpoint.{alias}.chat",
                        "pass",
                        "chat completions endpoint returned content",
                        {
                            "base_url": endpoint.base_url,
                            "model": response.model,
                            "prompt_tokens": response.prompt_tokens,
                            "completion_tokens": response.completion_tokens,
                        },
                    )
                )
            except Exception as exc:
                checks.append(
                    CheckResult(
                        f"model-endpoint.{alias}.chat",
                        "fail",
                        str(exc),
                        {"base_url": endpoint.base_url, "model": endpoint.model},
                    )
                )
    return checks


def _probe_orchestrator_endpoint(
    *,
    orchestrator_url: str,
    orchestrator_model: str,
    orchestrator_api_key_env: str | None,
    orchestrator_timeout: float,
    probe_text: str,
    signal: RouterSignal,
    allowed_models: set[str],
    model_catalog: list[dict[str, Any]] | None,
) -> CheckResult:
    try:
        client = OpenAIPlanClient(
            orchestrator_url,
            orchestrator_model,
            api_key_env=orchestrator_api_key_env,
            timeout_seconds=orchestrator_timeout,
        )
        raw_plan = client.generate(
            _plan_probe_messages(probe_text, signal, allowed_models, model_catalog)
        )
        plan = parse_plan(raw_plan)
        decision = LearnedGraphSelector(
            _StaticPlanClient(raw_plan),
            allowed_models=allowed_models,
            model_catalog=model_catalog,
        ).select(probe_text, signal, budget=Budget())
        if decision.selector_type == "deterministic_fallback":
            raise ValueError(decision.fallback_reason or "learned orchestrator plan was rejected")
        return CheckResult(
            "learned-orchestrator.endpoint",
            "pass",
            "learned orchestrator endpoint returned a valid JSON plan",
            {"base_url": orchestrator_url, "model": orchestrator_model, "graph_id": plan.get("graph_id") or plan.get("plan_type")},
        )
    except Exception as exc:
        return CheckResult(
            "learned-orchestrator.endpoint",
            "fail",
            str(exc),
            {"base_url": orchestrator_url, "model": orchestrator_model},
        )


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


class _StaticPlanClient:
    def __init__(self, plan: str) -> None:
        self.plan = plan

    def generate(self, messages: list[dict[str, str]]) -> str:
        del messages
        return self.plan


def _plan_probe_messages(
    probe_text: str,
    signal: RouterSignal,
    allowed_models: set[str],
    model_catalog: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    return build_plan_messages(
        text=probe_text,
        signal=signal,
        risk_level=GraphSelector().select(probe_text, signal).policy.risk_level,
        budget=Budget(),
        allowed_models=allowed_models,
        model_catalog=model_catalog,
    )


def _validate_capability_catalog(catalog: list[dict[str, Any]]) -> CheckResult:
    errors: list[str] = []
    aliases: set[str] = set()
    for item in catalog:
        alias = str(item.get("alias", ""))
        if not alias:
            errors.append("model alias is empty")
            continue
        if alias in aliases:
            errors.append(f"duplicate model alias: {alias}")
        aliases.add(alias)
        domains = item.get("domains") or []
        unknown_domains = set(domains) - set(LABEL_TO_MODEL) - {"*"}
        if unknown_domains:
            errors.append(f"{alias} has unknown domains: {sorted(unknown_domains)}")
        context_window = item.get("context_window")
        if context_window is not None and int(context_window) <= 0:
            errors.append(f"{alias} context_window must be positive")
    return CheckResult(
        "model-config.capability-catalog",
        "fail" if errors else "pass",
        "model capability catalog is valid" if not errors else "model capability catalog is invalid",
        {
            "models": len(catalog),
            "aliases": sorted(aliases),
            "errors": errors,
        },
    )
