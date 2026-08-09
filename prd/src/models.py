from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RouterSignal:
    scores: dict[str, float]
    model: str = "unknown-router"
    latency_ms: float = 0.0


@dataclass(frozen=True)
class PolicyDecision:
    risk_level: str
    action: str
    reason: str | None = None


@dataclass(frozen=True)
class Delegation:
    label: str
    model: str
    objective: str


@dataclass(frozen=True)
class RouteDecision:
    graph_id: str
    primary_labels: tuple[str, ...]
    secondary_labels: tuple[str, ...]
    selected_labels: tuple[str, ...]
    confidence: float
    margin: float
    reason: str
    policy: PolicyDecision
    delegations: tuple[Delegation, ...] = ()
    synthesis_strategy: str | None = None
    selector_type: str = "deterministic"
    fallback_reason: str | None = None
    generated_graph: dict[str, Any] | None = None
    selection_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["primary_labels"] = list(self.primary_labels)
        value["secondary_labels"] = list(self.secondary_labels)
        value["selected_labels"] = list(self.selected_labels)
        value["delegations"] = [asdict(item) for item in self.delegations]
        return value


@dataclass(frozen=True)
class Budget:
    max_cost_usd: float = 1.0
    max_latency_ms: int = 60_000
    max_steps: int = 12


@dataclass(frozen=True)
class ModelResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeTrace:
    id: str
    role: str
    model: str | None
    status: str
    started_at: str
    completed_at: str
    latency_ms: float
    attempts: int = 1
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    output: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionResult:
    trace: dict[str, Any]
    final_answer: str
