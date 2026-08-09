from __future__ import annotations

import math
import re
from dataclasses import dataclass

from constants import LABELS, LABEL_TO_MODEL
from models import Delegation, PolicyDecision, RouteDecision, RouterSignal


_HIGH_RISK_PATTERNS = (
    r"credential\s*(?:dump|steal|exfiltrat)",
    r"認証情報.*(?:窃取|抜き取|流出)",
    r"(?:ransomware|ランサムウェア).*(?:作|実装|deploy|配布)",
    r"(?:bypass|回避).*(?:edr|waf|mfa|認証)",
    r"(?:disable|停止|無効化).*(?:audit|logging|監査|ログ)",
    r"(?:破壊|削除).*(?:証拠|ログ)",
)


@dataclass(frozen=True)
class SelectionPolicy:
    minimum_confidence: float = 0.45
    single_specialist_margin: float = 0.25
    secondary_score: float = 0.15
    max_selected_labels: int = 3


class GraphSelector:
    def __init__(self, policy: SelectionPolicy | None = None) -> None:
        self.policy = policy or SelectionPolicy()
        for name in ("minimum_confidence", "single_specialist_margin", "secondary_score"):
            value = getattr(self.policy, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.policy.max_selected_labels < 1:
            raise ValueError("max_selected_labels must be at least 1")

    def select(
        self,
        text: str,
        signal: RouterSignal,
        risk_level: str = "auto",
    ) -> RouteDecision:
        scores = self._validated_scores(signal.scores)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        top_label, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_score - second_score
        policy = self._policy_decision(text, scores, risk_level)

        primary = (top_label,)
        secondary = tuple(
            label
            for label, score in ranked[1:]
            if score >= self.policy.secondary_score
        )[: self.policy.max_selected_labels - 1]

        if policy.action == "restrict":
            graph_id = "safe_refusal_or_handoff"
            reason = policy.reason or "policy restricted the request"
            selected = primary
        elif top_score < self.policy.minimum_confidence:
            graph_id = "clarify_first"
            reason = f"top score {top_score:.3f} is below {self.policy.minimum_confidence:.3f}"
            selected = primary
        elif policy.risk_level == "high":
            graph_id = "specialist_with_verifier"
            reason = "high-risk request requires verification"
            selected = primary
        elif margin >= self.policy.single_specialist_margin:
            graph_id = "single_specialist"
            reason = f"top-1 margin {margin:.3f} meets {self.policy.single_specialist_margin:.3f}"
            selected = primary
        else:
            graph_id = "parallel_experts"
            reason = f"top-1 margin {margin:.3f} is below {self.policy.single_specialist_margin:.3f}"
            selected = primary + secondary
            if len(selected) == 1:
                selected = tuple(label for label, _ in ranked[:2])

        return RouteDecision(
            graph_id=graph_id,
            primary_labels=primary,
            secondary_labels=secondary,
            selected_labels=selected,
            confidence=top_score,
            margin=margin,
            reason=reason,
            policy=policy,
            delegations=tuple(
                Delegation(
                    label=label,
                    model=LABEL_TO_MODEL[label],
                    objective=f"Analyze and answer the {label} aspects of the request.",
                )
                for label in selected
            ),
            synthesis_strategy=(
                "Prioritize findings, reconcile conflicts, and identify missing evidence."
                if graph_id == "parallel_experts"
                else None
            ),
        )

    @staticmethod
    def _validated_scores(raw_scores: dict[str, float]) -> dict[str, float]:
        if not isinstance(raw_scores, dict):
            raise ValueError("router scores must be an object")
        unknown = set(raw_scores) - set(LABELS)
        if unknown:
            raise ValueError(f"unknown router labels: {sorted(unknown)}")
        scores: dict[str, float] = {}
        for label in LABELS:
            value = float(raw_scores.get(label, 0.0))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid score for {label}: {value}")
            scores[label] = value
        total = sum(scores.values())
        if total <= 0:
            raise ValueError("router scores must contain at least one positive value")
        return {label: value / total for label, value in scores.items()}

    @staticmethod
    def _policy_decision(
        text: str,
        scores: dict[str, float],
        requested_risk: str,
    ) -> PolicyDecision:
        if requested_risk not in {"auto", "low", "normal", "high"}:
            raise ValueError("risk_level must be auto, low, normal, or high")

        pattern = next((pattern for pattern in _HIGH_RISK_PATTERNS if re.search(pattern, text, re.I)), None)
        if pattern:
            return PolicyDecision(
                risk_level="high",
                action="restrict",
                reason="security policy matched a destructive or credential-exfiltration request",
            )

        risk = requested_risk
        if risk == "auto":
            risk = "high" if scores.get("Security", 0.0) >= 0.45 else "normal"
        return PolicyDecision(risk_level=risk, action="allow")
