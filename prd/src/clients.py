from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from models import ModelResponse, RouterSignal


def _chat_completions_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/v1/chat/completions"


def _models_url(base_url: str) -> str:
    value = base_url.rstrip("/")
    if value.endswith("/models"):
        return value
    if value.endswith("/v1"):
        return value + "/models"
    return value + "/v1/models"


def _get_json(url: str, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    request = Request(url, method="GET")
    for name, value in headers.items():
        request.add_header(name, value)
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc.reason}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"expected a JSON object from {url}")
    return result


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    for name, value in headers.items():
        request.add_header(name, value)
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"request failed for {url}: {exc.reason}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"expected a JSON object from {url}")
    return result


class RouterClient(Protocol):
    def classify(self, text: str) -> RouterSignal: ...


class OpenAIRouterClient:
    def __init__(self, base_url: str, model: str = "router", timeout_seconds: float = 30.0) -> None:
        self.url = _chat_completions_url(base_url)
        self.model = model
        self.timeout_seconds = timeout_seconds

    def classify(self, text: str) -> RouterSignal:
        started = time.perf_counter()
        response = _post_json(
            self.url,
            {"model": self.model, "messages": [{"role": "user", "content": text}]},
            {},
            self.timeout_seconds,
        )
        router = response.get("router")
        if not isinstance(router, dict) or not isinstance(router.get("scores"), dict):
            raise RuntimeError("router response does not contain router.scores")
        scores = {str(label): float(score) for label, score in router["scores"].items()}
        return RouterSignal(
            scores=scores,
            model=str(response.get("model", self.model)),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


@dataclass(frozen=True)
class ModelEndpoint:
    model: str
    base_url: str
    api_key_env: str | None = None
    timeout_seconds: float = 30.0
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    domains: tuple[str, ...] = ()
    strengths: tuple[str, ...] = ()
    context_window: int | None = None
    latency_tier: str | None = None
    cost_tier: str | None = None
    safety_profile: str | None = None
    supports_json: bool = False

    @classmethod
    def from_dict(cls, alias: str, value: dict[str, Any], defaults: dict[str, Any]) -> "ModelEndpoint":
        merged = {**defaults, **value}
        if not merged.get("base_url"):
            raise ValueError(f"model {alias} is missing base_url")
        return cls(
            model=str(merged.get("model", alias)),
            base_url=str(merged["base_url"]),
            api_key_env=merged.get("api_key_env"),
            timeout_seconds=float(merged.get("timeout_seconds", 30.0)),
            input_cost_per_million=float(merged.get("input_cost_per_million", 0.0)),
            output_cost_per_million=float(merged.get("output_cost_per_million", 0.0)),
            domains=_string_tuple(merged.get("domains", ())),
            strengths=_string_tuple(merged.get("strengths", ())),
            context_window=_optional_positive_int(merged.get("context_window")),
            latency_tier=_optional_string(merged.get("latency_tier")),
            cost_tier=_optional_string(merged.get("cost_tier")),
            safety_profile=_optional_string(merged.get("safety_profile")),
            supports_json=bool(merged.get("supports_json", False)),
        )

    def capability_dict(self, alias: str) -> dict[str, Any]:
        value: dict[str, Any] = {
            "alias": alias,
            "model": self.model,
            "domains": list(self.domains),
            "strengths": list(self.strengths),
            "supports_json": self.supports_json,
        }
        if self.context_window is not None:
            value["context_window"] = self.context_window
        if self.latency_tier:
            value["latency_tier"] = self.latency_tier
        if self.cost_tier:
            value["cost_tier"] = self.cost_tier
        if self.safety_profile:
            value["safety_profile"] = self.safety_profile
        return value


class ModelClient(Protocol):
    def complete(
        self,
        model_alias: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        timeout_seconds: float | None = None,
    ) -> ModelResponse: ...


class OpenAIModelClient:
    def __init__(self, config: dict[str, Any]) -> None:
        defaults = config.get("defaults", {})
        models = config.get("models")
        if not isinstance(models, dict) or not models:
            raise ValueError("model config must contain a non-empty models object")
        self.endpoints = {
            alias: ModelEndpoint.from_dict(alias, value or {}, defaults)
            for alias, value in models.items()
        }

    def model_catalog(self) -> list[dict[str, Any]]:
        return [
            endpoint.capability_dict(alias)
            for alias, endpoint in sorted(self.endpoints.items())
        ]

    def complete(
        self,
        model_alias: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        timeout_seconds: float | None = None,
    ) -> ModelResponse:
        if model_alias not in self.endpoints:
            raise KeyError(f"model alias is not configured: {model_alias}")
        endpoint = self.endpoints[model_alias]
        headers: dict[str, str] = {}
        if endpoint.api_key_env:
            api_key = os.environ.get(endpoint.api_key_env)
            if not api_key:
                raise RuntimeError(f"environment variable is not set: {endpoint.api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"
        response = _post_json(
            _chat_completions_url(endpoint.base_url),
            {"model": endpoint.model, "messages": messages, "max_tokens": max_tokens},
            headers,
            timeout_seconds or endpoint.timeout_seconds,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("model response does not contain choices[0].message.content") from exc
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        cost = (
            prompt_tokens * endpoint.input_cost_per_million
            + completion_tokens * endpoint.output_cost_per_million
        ) / 1_000_000
        return ModelResponse(
            content=str(content),
            model=str(response.get("model", endpoint.model)),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            raw=response,
        )

    def list_models(self, model_alias: str, timeout_seconds: float | None = None) -> dict[str, Any]:
        if model_alias not in self.endpoints:
            raise KeyError(f"model alias is not configured: {model_alias}")
        endpoint = self.endpoints[model_alias]
        headers: dict[str, str] = {}
        if endpoint.api_key_env:
            api_key = os.environ.get(endpoint.api_key_env)
            if not api_key:
                raise RuntimeError(f"environment variable is not set: {endpoint.api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"
        return _get_json(_models_url(endpoint.base_url), headers, timeout_seconds or endpoint.timeout_seconds)


class MockModelClient:
    """Deterministic client for graph and trace verification without model calls."""

    def complete(
        self,
        model_alias: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        timeout_seconds: float | None = None,
    ) -> ModelResponse:
        del max_tokens, timeout_seconds
        system = messages[0]["content"].lower() if messages else ""
        if "verifier" in system:
            content = json.dumps({"passed": True, "issues": []})
        elif "synthesizer" in system:
            content = "[mock synthesis] " + messages[-1]["content"][-500:]
        elif "repair" in system:
            content = "[mock repaired answer] " + messages[-1]["content"][-400:]
        else:
            content = f"[mock {model_alias}] " + messages[-1]["content"][:400]
        return ModelResponse(
            content=content,
            model=f"mock/{model_alias}",
            prompt_tokens=sum(len(item["content"].split()) for item in messages),
            completion_tokens=len(content.split()),
            cost_usd=0.0,
        )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    raise ValueError(f"expected a string or array of strings: {value!r}")


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("context_window must be positive")
    return parsed


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
