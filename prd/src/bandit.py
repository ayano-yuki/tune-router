from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from models import RouteDecision, RouterSignal
from selector import GraphSelector
from shadow import ShadowConfig, build_shadow_decisions


BANDIT_STATE_FORMAT = "tune-orchestrator-bandit-v1"


@dataclass(frozen=True)
class BanditBuildConfig:
    cost_weight: float = 0.25
    latency_weight: float = 0.001


@dataclass(frozen=True)
class BanditPolicyConfig:
    min_observations: int = 3
    exploration_weight: float = 0.1
    explore_unobserved: bool = False
    include_high_risk: bool = False
    traffic_percent: float = 100.0
    rollout_salt: str = "default"
    rollout_digest: str | None = None
    release_digest: str | None = None
    release_id: str | None = None
    release_current_digest: str | None = None


@dataclass(frozen=True)
class BanditReplayConfig:
    min_observations: int = 3
    exploration_weight: float = 0.1
    explore_unobserved: bool = False
    include_high_risk: bool = False
    leave_one_out: bool = True


@dataclass(frozen=True)
class BanditPromotionConfig:
    min_evaluated_requests: int = 30
    min_mean_reward_delta: float = 0.0
    max_loss_rate: float = 0.05
    max_switch_rate: float = 0.50
    max_skip_rate: float = 0.80


@dataclass(frozen=True)
class BanditMonitorConfig:
    min_bandit_traces: int = 10
    max_bandit_failure_rate: float = 0.05
    max_relative_failure_rate: float = 0.10
    max_bandit_p95_latency_ms: float = 60_000.0
    max_bandit_mean_cost_usd: float = 1.0


@dataclass(frozen=True)
class BanditRolloutPlanConfig:
    current_traffic_percent: float = 0.0
    step_percent: float = 10.0
    max_traffic_percent: float = 100.0
    rollback_traffic_percent: float = 0.0
    rollout_salt: str = "default"
    min_monitor_bandit_traces: int = 10
    max_age_hours: float = 24.0


def build_bandit_state(traces: list[dict[str, Any]], config: BanditBuildConfig | None = None) -> dict[str, Any]:
    cfg = config or BanditBuildConfig()
    contexts: dict[str, dict[str, Any]] = {}
    skipped = 0
    for trace in _iter_bandit_observation_traces(traces):
        reward = reward_from_trace(trace, cfg)
        if reward is None:
            skipped += 1
            continue
        context = context_key_from_trace(trace)
        arm = arm_key_from_trace(trace)
        if not context or not arm:
            skipped += 1
            continue
        usage = trace.get("usage", {})
        context_state = contexts.setdefault(context, {"arms": {}, "observations": 0})
        arm_state = context_state["arms"].setdefault(
            arm,
            {
                "pulls": 0,
                "reward_sum": 0.0,
                "mean_reward": 0.0,
                "cost_sum": 0.0,
                "latency_sum_ms": 0.0,
                "successes": 0,
                "last_trace_id": None,
                "example_decision": _decision_summary_from_trace(trace),
            },
        )
        arm_state["pulls"] += 1
        arm_state["reward_sum"] += reward
        arm_state["mean_reward"] = arm_state["reward_sum"] / arm_state["pulls"]
        arm_state["cost_sum"] += float(usage.get("cost_usd", 0.0))
        arm_state["latency_sum_ms"] += float(usage.get("latency_ms", 0.0))
        arm_state["successes"] += int(_trace_success(trace))
        arm_state["last_trace_id"] = trace.get("trace_id")
        context_state["observations"] += 1
    return {
        "format": BANDIT_STATE_FORMAT,
        "config": {
            "cost_weight": cfg.cost_weight,
            "latency_weight": cfg.latency_weight,
        },
        "contexts": contexts,
        "summary": {
            "contexts": len(contexts),
            "observations": sum(int(value["observations"]) for value in contexts.values()),
            "skipped": skipped,
        },
    }


def _iter_bandit_observation_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for trace in traces:
        observations.append(trace)
        for shadow in trace.get("shadow_executions", []):
            if not isinstance(shadow, dict) or shadow.get("status") != "completed":
                continue
            shadow_trace = shadow.get("trace")
            if isinstance(shadow_trace, dict):
                observations.append(shadow_trace)
    return observations


def write_bandit_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def load_bandit_state(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("format") != BANDIT_STATE_FORMAT:
        raise ValueError(f"invalid bandit state: {path}")
    return state


def replay_bandit_policy(
    traces: list[dict[str, Any]],
    state: dict[str, Any],
    config: BanditReplayConfig | None = None,
    reward_config: BanditBuildConfig | None = None,
) -> dict[str, Any]:
    cfg = config or BanditReplayConfig()
    reward_cfg = reward_config or _build_config_from_state(state)
    details: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    deltas: list[float] = []
    served_rewards: list[float] = []
    selected_rewards: list[float] = []
    switches = 0
    wins = 0
    losses = 0
    ties = 0
    policy_applied = 0

    for trace in traces:
        request = _replay_request(trace, reward_cfg)
        if request is None:
            _increment(skipped, "unreviewed_or_incomplete")
            continue
        context_state = state.get("contexts", {}).get(request["context_key"])
        if not isinstance(context_state, dict):
            _increment(skipped, "unobserved_context")
            details.append(_skipped_replay_detail(request, "unobserved_context"))
            continue
        if request["risk_level"] == "high" and not cfg.include_high_risk:
            _increment(skipped, "high_risk")
            details.append(_skipped_replay_detail(request, "high_risk"))
            continue
        replay_context = _remove_request_observations(context_state, request["candidates"]) if cfg.leave_one_out else context_state
        if int(replay_context.get("observations", 0)) < cfg.min_observations:
            _increment(skipped, "insufficient_observations")
            details.append(_skipped_replay_detail(request, "insufficient_observations"))
            continue

        scored = _score_replay_candidates(request["candidates"], replay_context, cfg)
        if not scored:
            _increment(skipped, "no_scored_candidates")
            details.append(_skipped_replay_detail(request, "no_scored_candidates"))
            continue

        best = max(scored, key=lambda item: (item["score"], item["observed"], item["source"]))
        served = request["served"]
        selected = best["candidate"]
        delta = float(selected["reward"]) - float(served["reward"])
        details.append(
            {
                "trace_id": request["trace_id"],
                "context_key": request["context_key"],
                "risk_level": request["risk_level"],
                "served_arm_key": served["arm_key"],
                "served_source": served["source"],
                "served_reward": served["reward"],
                "selected_arm_key": selected["arm_key"],
                "selected_source": selected["source"],
                "selected_reward": selected["reward"],
                "selected_score": best["score"],
                "selected_observations": best["observed"],
                "reward_delta": delta,
                "candidate_count": len(request["candidates"]),
                "leave_one_out": cfg.leave_one_out,
            }
        )
        policy_applied += 1
        served_rewards.append(float(served["reward"]))
        selected_rewards.append(float(selected["reward"]))
        deltas.append(delta)
        if selected["arm_key"] != served["arm_key"]:
            switches += 1
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1

    evaluated = len(deltas)
    return {
        "format": "tune-orchestrator-bandit-replay-v1",
        "config": {
            "min_observations": cfg.min_observations,
            "exploration_weight": cfg.exploration_weight,
            "explore_unobserved": cfg.explore_unobserved,
            "include_high_risk": cfg.include_high_risk,
            "leave_one_out": cfg.leave_one_out,
            "reward": {
                "cost_weight": reward_cfg.cost_weight,
                "latency_weight": reward_cfg.latency_weight,
            },
        },
        "summary": {
            "requests": len(traces),
            "evaluated_requests": evaluated,
            "policy_applied": policy_applied,
            "switches": switches,
            "switch_rate": switches / evaluated if evaluated else 0.0,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": wins / evaluated if evaluated else 0.0,
            "loss_rate": losses / evaluated if evaluated else 0.0,
            "mean_served_reward": _mean(served_rewards),
            "mean_selected_reward": _mean(selected_rewards),
            "mean_reward_delta": _mean(deltas),
            "skipped": dict(sorted(skipped.items())),
        },
        "details": details,
    }


def write_bandit_replay_report(path: Path, replay: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = replay["summary"]
    lines = [
        "# Bandit Replay Report",
        "",
        f"- Requests: {summary['requests']}",
        f"- Evaluated requests: {summary['evaluated_requests']}",
        f"- Switch rate: {summary['switch_rate']:.1%}",
        f"- Win / loss / tie: {summary['wins']} / {summary['losses']} / {summary['ties']}",
        f"- Mean served reward: {summary['mean_served_reward']:.4f}",
        f"- Mean selected reward: {summary['mean_selected_reward']:.4f}",
        f"- Mean reward delta: {summary['mean_reward_delta']:.4f}",
        "",
        "## Skipped",
        "",
    ]
    skipped = summary.get("skipped", {})
    if skipped:
        lines.extend(f"- `{reason}`: {count}" for reason, count in skipped.items())
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "| Trace | Selected | Delta | Candidates |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in replay.get("details", []):
        if "reward_delta" not in row:
            continue
        lines.append(
            f"| `{row['trace_id']}` | `{row['selected_source']}` | "
            f"{float(row['reward_delta']):.4f} | {int(row['candidate_count'])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def evaluate_bandit_promotion(
    replay: dict[str, Any],
    config: BanditPromotionConfig | None = None,
) -> dict[str, Any]:
    cfg = config or BanditPromotionConfig()
    summary = replay.get("summary", {})
    requests = int(summary.get("requests", 0))
    evaluated = int(summary.get("evaluated_requests", 0))
    skipped = summary.get("skipped", {})
    skipped_total = sum(int(value) for value in skipped.values()) if isinstance(skipped, dict) else 0
    skip_rate = skipped_total / requests if requests else 1.0
    checks = {
        "min_evaluated_requests": _promotion_check(evaluated >= cfg.min_evaluated_requests, evaluated, cfg.min_evaluated_requests),
        "min_mean_reward_delta": _promotion_check(
            float(summary.get("mean_reward_delta", 0.0)) >= cfg.min_mean_reward_delta,
            float(summary.get("mean_reward_delta", 0.0)),
            cfg.min_mean_reward_delta,
        ),
        "max_loss_rate": _promotion_check(
            float(summary.get("loss_rate", 0.0)) <= cfg.max_loss_rate,
            float(summary.get("loss_rate", 0.0)),
            cfg.max_loss_rate,
        ),
        "max_switch_rate": _promotion_check(
            float(summary.get("switch_rate", 0.0)) <= cfg.max_switch_rate,
            float(summary.get("switch_rate", 0.0)),
            cfg.max_switch_rate,
        ),
        "max_skip_rate": _promotion_check(skip_rate <= cfg.max_skip_rate, skip_rate, cfg.max_skip_rate),
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "format": "tune-orchestrator-bandit-promotion-v1",
        "status": "pass" if passed else "fail",
        "config": {
            "min_evaluated_requests": cfg.min_evaluated_requests,
            "min_mean_reward_delta": cfg.min_mean_reward_delta,
            "max_loss_rate": cfg.max_loss_rate,
            "max_switch_rate": cfg.max_switch_rate,
            "max_skip_rate": cfg.max_skip_rate,
        },
        "summary": {
            "requests": requests,
            "evaluated_requests": evaluated,
            "skip_rate": skip_rate,
            "mean_reward_delta": float(summary.get("mean_reward_delta", 0.0)),
            "loss_rate": float(summary.get("loss_rate", 0.0)),
            "switch_rate": float(summary.get("switch_rate", 0.0)),
        },
        "checks": checks,
    }


def write_bandit_promotion_report(path: Path, promotion: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = promotion["summary"]
    lines = [
        "# Bandit Promotion Gate",
        "",
        f"- Status: {promotion['status']}",
        f"- Evaluated requests: {summary['evaluated_requests']}",
        f"- Mean reward delta: {summary['mean_reward_delta']:.4f}",
        f"- Loss rate: {summary['loss_rate']:.1%}",
        f"- Switch rate: {summary['switch_rate']:.1%}",
        f"- Skip rate: {summary['skip_rate']:.1%}",
        "",
        "## Checks",
        "",
        "| Check | Passed | Actual | Threshold |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, check in promotion["checks"].items():
        lines.append(
            f"| `{name}` | {str(check['passed']).lower()} | "
            f"{float(check['actual']):.4f} | {float(check['threshold']):.4f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def monitor_bandit_rollout(
    traces: list[dict[str, Any]],
    config: BanditMonitorConfig | None = None,
) -> dict[str, Any]:
    cfg = config or BanditMonitorConfig()
    bandit_traces = [trace for trace in traces if _is_bandit_selected_trace(trace)]
    baseline_traces = [trace for trace in traces if not _is_bandit_selected_trace(trace)]
    eligible_traces = [trace for trace in traces if _bandit_candidate_metadata(trace) is not None]
    sampled_traces = [
        trace
        for trace in eligible_traces
        if _bandit_candidate_metadata(trace).get("sampled") or _is_bandit_selected_trace(trace)
    ]
    bandit_stats = _rollout_trace_stats(bandit_traces)
    baseline_stats = _rollout_trace_stats(baseline_traces)
    relative_failure_rate = (
        max(0.0, bandit_stats["failure_rate"] - baseline_stats["failure_rate"])
        if baseline_stats["traces"]
        else 0.0
    )
    checks = {
        "min_bandit_traces": _promotion_check(
            bandit_stats["traces"] >= cfg.min_bandit_traces,
            bandit_stats["traces"],
            cfg.min_bandit_traces,
        ),
        "max_bandit_failure_rate": _promotion_check(
            bandit_stats["failure_rate"] <= cfg.max_bandit_failure_rate,
            bandit_stats["failure_rate"],
            cfg.max_bandit_failure_rate,
        ),
        "max_relative_failure_rate": _promotion_check(
            relative_failure_rate <= cfg.max_relative_failure_rate,
            relative_failure_rate,
            cfg.max_relative_failure_rate,
        ),
        "max_bandit_p95_latency_ms": _promotion_check(
            bandit_stats["latency_p95_ms"] <= cfg.max_bandit_p95_latency_ms,
            bandit_stats["latency_p95_ms"],
            cfg.max_bandit_p95_latency_ms,
        ),
        "max_bandit_mean_cost_usd": _promotion_check(
            bandit_stats["mean_cost_usd"] <= cfg.max_bandit_mean_cost_usd,
            bandit_stats["mean_cost_usd"],
            cfg.max_bandit_mean_cost_usd,
        ),
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "format": "tune-orchestrator-bandit-monitor-v1",
        "status": "pass" if passed else "fail",
        "config": {
            "min_bandit_traces": cfg.min_bandit_traces,
            "max_bandit_failure_rate": cfg.max_bandit_failure_rate,
            "max_relative_failure_rate": cfg.max_relative_failure_rate,
            "max_bandit_p95_latency_ms": cfg.max_bandit_p95_latency_ms,
            "max_bandit_mean_cost_usd": cfg.max_bandit_mean_cost_usd,
        },
        "summary": {
            "traces": len(traces),
            "bandit": bandit_stats,
            "baseline": baseline_stats,
            "relative_failure_rate": relative_failure_rate,
            "canary_eligible": len(eligible_traces),
            "canary_sampled": len(sampled_traces),
            "canary_sample_rate": len(sampled_traces) / len(eligible_traces) if eligible_traces else 0.0,
            "stop_reasons": _stop_reason_counts(traces),
        },
        "checks": checks,
    }


def write_bandit_monitor_report(path: Path, monitor: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = monitor["summary"]
    bandit = summary["bandit"]
    baseline = summary["baseline"]
    lines = [
        "# Bandit Rollout Monitor",
        "",
        f"- Status: {monitor['status']}",
        f"- Traces: {summary['traces']}",
        f"- Bandit traces: {bandit['traces']}",
        f"- Bandit failure rate: {bandit['failure_rate']:.1%}",
        f"- Baseline failure rate: {baseline['failure_rate']:.1%}",
        f"- Relative failure rate: {summary['relative_failure_rate']:.1%}",
        f"- Bandit P95 latency: {bandit['latency_p95_ms']:.0f} ms",
        f"- Bandit mean cost: ${bandit['mean_cost_usd']:.6f}",
        f"- Canary eligible / sampled: {summary['canary_eligible']} / {summary['canary_sampled']}",
        "",
        "## Checks",
        "",
        "| Check | Passed | Actual | Threshold |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, check in monitor["checks"].items():
        lines.append(
            f"| `{name}` | {str(check['passed']).lower()} | "
            f"{float(check['actual']):.4f} | {float(check['threshold']):.4f} |"
        )
    lines.extend(["", "## Stop Reasons", ""])
    stop_reasons = summary.get("stop_reasons", {})
    if stop_reasons:
        lines.extend(f"- `{reason}`: {count}" for reason, count in stop_reasons.items())
    else:
        lines.append("- none")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def plan_bandit_rollout(
    *,
    promotion: dict[str, Any] | None,
    monitor: dict[str, Any] | None,
    state: dict[str, Any] | None = None,
    config: BanditRolloutPlanConfig | None = None,
) -> dict[str, Any]:
    cfg = config or BanditRolloutPlanConfig()
    current = max(0.0, min(100.0, cfg.current_traffic_percent))
    step = max(0.0, cfg.step_percent)
    maximum = max(0.0, min(100.0, cfg.max_traffic_percent))
    rollback = max(0.0, min(100.0, cfg.rollback_traffic_percent))
    reasons: list[str] = []

    promotion_status = str((promotion or {}).get("status", "missing"))
    monitor_status = str((monitor or {}).get("status", "missing"))
    monitor_bandit_traces = int((monitor or {}).get("summary", {}).get("bandit", {}).get("traces", 0))
    created_at = datetime.now(UTC)
    expires_at = created_at + timedelta(hours=max(0.0, cfg.max_age_hours))

    if promotion_status != "pass":
        action = "rollback" if current > rollback else "hold"
        target = rollback if action == "rollback" else current
        reasons.append(f"promotion status is {promotion_status}")
    elif monitor_status == "fail":
        action = "rollback"
        target = rollback
        reasons.append("monitor health gate failed")
    elif current > 0 and monitor is None:
        action = "hold"
        target = current
        reasons.append("monitor result is required after rollout starts")
    elif current > 0 and monitor_bandit_traces < cfg.min_monitor_bandit_traces:
        action = "hold"
        target = current
        reasons.append(f"monitor has only {monitor_bandit_traces} bandit traces")
    elif current >= maximum:
        action = "hold"
        target = maximum
        reasons.append("traffic is already at max")
    else:
        action = "advance"
        target = min(maximum, current + step)
        reasons.append("promotion and monitor gates passed")

    runtime = {
        "enabled": target > 0,
        "traffic_percent": target,
        "rollout_salt": cfg.rollout_salt,
    }
    return {
        "format": "tune-orchestrator-bandit-rollout-v1",
        "created_at": _format_utc(created_at),
        "expires_at": _format_utc(expires_at),
        "action": action,
        "current_traffic_percent": current,
        "target_traffic_percent": target,
        "max_traffic_percent": maximum,
        "runtime": runtime,
        "artifacts": {
            "bandit_state_digest": artifact_digest(state) if state is not None else None,
            "promotion_digest": artifact_digest(promotion) if promotion is not None else None,
            "monitor_digest": artifact_digest(monitor) if monitor is not None else None,
        },
        "inputs": {
            "promotion_status": promotion_status,
            "monitor_status": monitor_status,
            "monitor_bandit_traces": monitor_bandit_traces,
            "max_age_hours": cfg.max_age_hours,
        },
        "reasons": reasons,
    }


def write_bandit_rollout_plan(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def validate_bandit_rollout_binding(
    *,
    state: dict[str, Any],
    rollout: dict[str, Any],
    require_state_binding: bool = True,
    require_not_expired: bool = True,
) -> dict[str, Any]:
    artifacts = rollout.get("artifacts", {})
    expected = artifacts.get("bandit_state_digest") if isinstance(artifacts, dict) else None
    actual = artifact_digest(state)
    expires_at = _parse_utc(str(rollout.get("expires_at", "")))
    expired = expires_at is None or datetime.now(UTC) >= expires_at
    checks = {
        "rollout_format": _promotion_check(
            rollout.get("format") == "tune-orchestrator-bandit-rollout-v1",
            1 if rollout.get("format") == "tune-orchestrator-bandit-rollout-v1" else 0,
            1,
        ),
        "state_binding_present": _promotion_check(
            bool(expected) or not require_state_binding,
            1 if expected else 0,
            1 if require_state_binding else 0,
        ),
        "state_digest_matches": _promotion_check(
            (not expected and not require_state_binding) or expected == actual,
            1 if expected == actual else 0,
            1,
        ),
        "not_expired": _promotion_check(
            not require_not_expired or not expired,
            0 if expired else 1,
            1,
        ),
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "format": "tune-orchestrator-bandit-rollout-binding-v1",
        "status": "pass" if passed else "fail",
        "expected_state_digest": expected,
        "actual_state_digest": actual,
        "expires_at": rollout.get("expires_at"),
        "checks": checks,
    }


def validate_bandit_rollout_artifacts(
    *,
    rollout: dict[str, Any],
    state: dict[str, Any],
    promotion: dict[str, Any] | None = None,
    monitor: dict[str, Any] | None = None,
    require_state_binding: bool = True,
    require_promotion_binding: bool = True,
    require_monitor_binding: bool = False,
    require_not_expired: bool = True,
) -> dict[str, Any]:
    binding = validate_bandit_rollout_binding(
        state=state,
        rollout=rollout,
        require_state_binding=require_state_binding,
        require_not_expired=require_not_expired,
    )
    artifacts = rollout.get("artifacts", {})
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    expected_promotion = artifacts.get("promotion_digest")
    expected_monitor = artifacts.get("monitor_digest")
    actual_promotion = artifact_digest(promotion) if promotion is not None else None
    actual_monitor = artifact_digest(monitor) if monitor is not None else None
    checks = {
        **binding["checks"],
        "promotion_binding_present": _promotion_check(
            bool(expected_promotion) or not require_promotion_binding,
            1 if expected_promotion else 0,
            1 if require_promotion_binding else 0,
        ),
        "promotion_digest_matches": _promotion_check(
            _artifact_match(expected_promotion, actual_promotion, require_promotion_binding),
            1 if expected_promotion and actual_promotion == expected_promotion else 0,
            1,
        ),
        "monitor_binding_present": _promotion_check(
            bool(expected_monitor) or not require_monitor_binding,
            1 if expected_monitor else 0,
            1 if require_monitor_binding else 0,
        ),
        "monitor_digest_matches": _promotion_check(
            _artifact_match(expected_monitor, actual_monitor, require_monitor_binding),
            1 if expected_monitor and actual_monitor == expected_monitor else 0,
            1,
        ),
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "format": "tune-orchestrator-bandit-rollout-artifacts-v1",
        "status": "pass" if passed else "fail",
        "expected": {
            "bandit_state_digest": binding["expected_state_digest"],
            "promotion_digest": expected_promotion,
            "monitor_digest": expected_monitor,
        },
        "actual": {
            "bandit_state_digest": binding["actual_state_digest"],
            "promotion_digest": actual_promotion,
            "monitor_digest": actual_monitor,
        },
        "expires_at": rollout.get("expires_at"),
        "checks": checks,
    }


def write_bandit_artifact_verification_report(path: Path, verification: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Bandit Rollout Artifact Verification",
        "",
        f"- Status: {verification['status']}",
        f"- Expires at: `{verification.get('expires_at')}`",
        "",
        "## Digests",
        "",
        "| Artifact | Expected | Actual |",
        "| --- | --- | --- |",
    ]
    expected = verification.get("expected", {})
    actual = verification.get("actual", {})
    for name in ("bandit_state_digest", "promotion_digest", "monitor_digest"):
        lines.append(f"| `{name}` | `{expected.get(name)}` | `{actual.get(name)}` |")
    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| Check | Passed | Actual | Threshold |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, check in verification["checks"].items():
        lines.append(
            f"| `{name}` | {str(check['passed']).lower()} | "
            f"{float(check['actual']):.4f} | {float(check['threshold']):.4f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def build_bandit_release_manifest(
    *,
    state: dict[str, Any],
    rollout: dict[str, Any],
    promotion: dict[str, Any] | None = None,
    monitor: dict[str, Any] | None = None,
    verification: dict[str, Any] | None = None,
    require_monitor: bool = False,
) -> dict[str, Any]:
    verification = verification or validate_bandit_rollout_artifacts(
        rollout=rollout,
        state=state,
        promotion=promotion,
        monitor=monitor,
        require_monitor_binding=require_monitor,
    )
    runtime = rollout.get("runtime", {})
    runtime = runtime if isinstance(runtime, dict) else {}
    artifacts = {
        "bandit_state_digest": artifact_digest(state),
        "promotion_digest": artifact_digest(promotion) if promotion is not None else None,
        "monitor_digest": artifact_digest(monitor) if monitor is not None else None,
        "rollout_digest": artifact_digest(rollout),
        "verification_digest": artifact_digest(verification),
    }
    status = "pass"
    reasons: list[str] = []
    if verification.get("status") != "pass":
        status = "fail"
        reasons.append("rollout artifact verification failed")
    if promotion is not None and promotion.get("status") != "pass":
        status = "fail"
        reasons.append("promotion gate did not pass")
    if require_monitor and (monitor is None or monitor.get("status") != "pass"):
        status = "fail"
        reasons.append("monitor gate is required and did not pass")
    if rollout.get("action") == "rollback":
        status = "fail"
        reasons.append("rollout action is rollback")
    if not runtime:
        status = "fail"
        reasons.append("rollout runtime config is missing")
    release_id = hashlib.sha256(json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
    return {
        "format": "tune-orchestrator-bandit-release-v1",
        "release_id": f"bandit-release-{release_id}",
        "created_at": _utc_now(),
        "status": status,
        "runtime": {
            "enabled": bool(runtime.get("enabled", False)),
            "traffic_percent": float(runtime.get("traffic_percent", 0.0)),
            "rollout_salt": str(runtime.get("rollout_salt", "default")),
        },
        "expires_at": rollout.get("expires_at"),
        "rollout_action": rollout.get("action"),
        "artifacts": artifacts,
        "gates": {
            "promotion_status": (promotion or {}).get("status"),
            "monitor_status": (monitor or {}).get("status"),
            "verification_status": verification.get("status"),
            "require_monitor": require_monitor,
        },
        "reasons": reasons or ["release artifact chain passed"],
    }


def validate_bandit_release_manifest(
    *,
    manifest: dict[str, Any],
    state: dict[str, Any],
    rollout: dict[str, Any] | None = None,
    verification: dict[str, Any] | None = None,
    require_not_expired: bool = True,
) -> dict[str, Any]:
    artifacts = manifest.get("artifacts", {})
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    expected_state = artifacts.get("bandit_state_digest")
    expected_rollout = artifacts.get("rollout_digest")
    expected_verification = artifacts.get("verification_digest")
    actual_state = artifact_digest(state)
    actual_rollout = artifact_digest(rollout) if rollout is not None else None
    actual_verification = artifact_digest(verification) if verification is not None else None
    expires_at = _parse_utc(str(manifest.get("expires_at", "")))
    expired = expires_at is None or datetime.now(UTC) >= expires_at
    checks = {
        "release_format": _promotion_check(
            manifest.get("format") == "tune-orchestrator-bandit-release-v1",
            1 if manifest.get("format") == "tune-orchestrator-bandit-release-v1" else 0,
            1,
        ),
        "release_status_pass": _promotion_check(
            manifest.get("status") == "pass",
            1 if manifest.get("status") == "pass" else 0,
            1,
        ),
        "state_digest_matches": _promotion_check(
            bool(expected_state) and expected_state == actual_state,
            1 if expected_state == actual_state else 0,
            1,
        ),
        "rollout_digest_matches": _promotion_check(
            rollout is None or (bool(expected_rollout) and expected_rollout == actual_rollout),
            1 if expected_rollout and expected_rollout == actual_rollout else 0,
            1,
        ),
        "verification_digest_matches": _promotion_check(
            verification is None or (bool(expected_verification) and expected_verification == actual_verification),
            1 if expected_verification and expected_verification == actual_verification else 0,
            1,
        ),
        "not_expired": _promotion_check(
            not require_not_expired or not expired,
            0 if expired else 1,
            1,
        ),
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "format": "tune-orchestrator-bandit-release-verification-v1",
        "status": "pass" if passed else "fail",
        "release_id": manifest.get("release_id"),
        "expected": {
            "bandit_state_digest": expected_state,
            "rollout_digest": expected_rollout,
            "verification_digest": expected_verification,
        },
        "actual": {
            "bandit_state_digest": actual_state,
            "rollout_digest": actual_rollout,
            "verification_digest": actual_verification,
        },
        "expires_at": manifest.get("expires_at"),
        "checks": checks,
    }


def write_bandit_release_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def write_bandit_release_report(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Bandit Release Manifest",
        "",
        f"- Release: `{manifest['release_id']}`",
        f"- Status: {manifest['status']}",
        f"- Runtime enabled: {str(manifest['runtime']['enabled']).lower()}",
        f"- Traffic: {float(manifest['runtime']['traffic_percent']):.1f}%",
        f"- Rollout salt: `{manifest['runtime']['rollout_salt']}`",
        f"- Expires at: `{manifest.get('expires_at')}`",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in manifest.get("artifacts", {}).items())
    lines.extend(["", "## Reasons", ""])
    lines.extend(f"- {reason}" for reason in manifest.get("reasons", []))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def build_bandit_current_release(
    *,
    manifest: dict[str, Any],
    manifest_path: str,
    channel: str = "production",
) -> dict[str, Any]:
    return {
        "format": "tune-orchestrator-bandit-current-v1",
        "channel": channel,
        "activated_at": _utc_now(),
        "release_id": manifest.get("release_id"),
        "manifest_path": manifest_path,
        "manifest_digest": artifact_digest(manifest),
        "runtime": manifest.get("runtime", {}),
        "expires_at": manifest.get("expires_at"),
    }


def validate_bandit_current_release(
    *,
    current: dict[str, Any],
    manifest: dict[str, Any],
    require_not_expired: bool = True,
) -> dict[str, Any]:
    expected_manifest = current.get("manifest_digest")
    actual_manifest = artifact_digest(manifest)
    expires_at = _parse_utc(str(current.get("expires_at", "")))
    expired = expires_at is None or datetime.now(UTC) >= expires_at
    checks = {
        "current_format": _promotion_check(
            current.get("format") == "tune-orchestrator-bandit-current-v1",
            1 if current.get("format") == "tune-orchestrator-bandit-current-v1" else 0,
            1,
        ),
        "release_id_matches": _promotion_check(
            current.get("release_id") == manifest.get("release_id"),
            1 if current.get("release_id") == manifest.get("release_id") else 0,
            1,
        ),
        "manifest_digest_matches": _promotion_check(
            bool(expected_manifest) and expected_manifest == actual_manifest,
            1 if expected_manifest == actual_manifest else 0,
            1,
        ),
        "not_expired": _promotion_check(
            not require_not_expired or not expired,
            0 if expired else 1,
            1,
        ),
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "format": "tune-orchestrator-bandit-current-verification-v1",
        "status": "pass" if passed else "fail",
        "channel": current.get("channel"),
        "release_id": current.get("release_id"),
        "expected_manifest_digest": expected_manifest,
        "actual_manifest_digest": actual_manifest,
        "manifest_path": current.get("manifest_path"),
        "expires_at": current.get("expires_at"),
        "checks": checks,
    }


def validate_bandit_current_artifacts(
    *,
    current: dict[str, Any],
    manifest: dict[str, Any],
    state: dict[str, Any],
    registry: dict[str, Any] | None = None,
    require_registry_entry: bool = False,
    require_not_expired: bool = True,
) -> dict[str, Any]:
    current_verification = validate_bandit_current_release(
        current=current,
        manifest=manifest,
        require_not_expired=require_not_expired,
    )
    release_verification = validate_bandit_release_manifest(
        manifest=manifest,
        state=state,
        require_not_expired=require_not_expired,
    )
    current_digest = artifact_digest(current)
    expected_state = (manifest.get("artifacts", {}) if isinstance(manifest.get("artifacts"), dict) else {}).get("bandit_state_digest")
    actual_state = artifact_digest(state)
    registry_entry = _find_release_registry_entry(current=current, registry=registry)
    registry_present = registry_entry is not None
    registry_digest_matches = registry_entry is not None and registry_entry.get("current_digest") == current_digest
    checks = {
        "current_pointer_valid": _promotion_check(
            current_verification["status"] == "pass",
            1 if current_verification["status"] == "pass" else 0,
            1,
        ),
        "release_manifest_valid": _promotion_check(
            release_verification["status"] == "pass",
            1 if release_verification["status"] == "pass" else 0,
            1,
        ),
        "state_digest_matches": _promotion_check(
            bool(expected_state) and expected_state == actual_state,
            1 if expected_state == actual_state else 0,
            1,
        ),
        "registry_entry_present": _promotion_check(
            not require_registry_entry or registry_present,
            1 if registry_present else 0,
            1,
        ),
        "registry_current_digest_matches": _promotion_check(
            registry is None or not registry_present or registry_digest_matches,
            1 if registry_digest_matches else 0,
            1,
        ),
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "format": "tune-orchestrator-bandit-current-artifact-verification-v1",
        "status": "pass" if passed else "fail",
        "channel": current.get("channel"),
        "release_id": current.get("release_id"),
        "current_digest": current_digest,
        "manifest_path": current.get("manifest_path"),
        "expected": {
            "manifest_digest": current.get("manifest_digest"),
            "state_digest": expected_state,
            "registry_current_digest": registry_entry.get("current_digest") if registry_entry else None,
        },
        "actual": {
            "manifest_digest": artifact_digest(manifest),
            "state_digest": actual_state,
            "registry_entry_present": registry_present,
        },
        "current_verification": current_verification,
        "release_verification": release_verification,
        "checks": checks,
    }


def build_bandit_runtime_bundle(
    *,
    current: dict[str, Any],
    manifest: dict[str, Any],
    state: dict[str, Any],
    current_verification: dict[str, Any],
    registry: dict[str, Any] | None = None,
    graphs_digest: str | None = None,
    model_config_digest: str | None = None,
    paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    artifacts = {
        "current_digest": artifact_digest(current),
        "manifest_digest": artifact_digest(manifest),
        "state_digest": artifact_digest(state),
        "current_verification_digest": artifact_digest(current_verification),
        "registry_digest": artifact_digest(registry) if registry is not None else None,
        "graphs_digest": graphs_digest,
        "model_config_digest": model_config_digest,
    }
    status = "pass" if current_verification.get("status") == "pass" else "fail"
    if manifest.get("status") != "pass":
        status = "fail"
    bundle_seed = {
        "release_id": current.get("release_id"),
        "channel": current.get("channel"),
        "artifacts": artifacts,
    }
    bundle_id = hashlib.sha256(json.dumps(bundle_seed, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]
    return {
        "format": "tune-orchestrator-bandit-runtime-bundle-v1",
        "bundle_id": f"bandit-runtime-{bundle_id}",
        "created_at": _utc_now(),
        "status": status,
        "channel": current.get("channel"),
        "release_id": current.get("release_id"),
        "runtime": current.get("runtime", {}),
        "expires_at": current.get("expires_at"),
        "artifacts": artifacts,
        "paths": paths or {},
        "gates": {
            "current_verification_status": current_verification.get("status"),
            "manifest_status": manifest.get("status"),
        },
        "reasons": ["runtime bundle artifact chain passed"] if status == "pass" else ["runtime bundle artifact chain failed"],
    }


def validate_bandit_runtime_bundle(
    *,
    bundle: dict[str, Any],
    current: dict[str, Any],
    manifest: dict[str, Any],
    state: dict[str, Any],
    current_verification: dict[str, Any],
    registry: dict[str, Any] | None = None,
    graphs_digest: str | None = None,
    model_config_digest: str | None = None,
    require_not_expired: bool = True,
) -> dict[str, Any]:
    artifacts = bundle.get("artifacts", {})
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    expires_at = _parse_utc(str(bundle.get("expires_at", "")))
    expired = expires_at is None or datetime.now(UTC) >= expires_at
    checks = {
        "bundle_format": _promotion_check(
            bundle.get("format") == "tune-orchestrator-bandit-runtime-bundle-v1",
            1 if bundle.get("format") == "tune-orchestrator-bandit-runtime-bundle-v1" else 0,
            1,
        ),
        "bundle_status_pass": _promotion_check(
            bundle.get("status") == "pass",
            1 if bundle.get("status") == "pass" else 0,
            1,
        ),
        "release_id_matches": _promotion_check(
            bundle.get("release_id") == current.get("release_id") == manifest.get("release_id"),
            1 if bundle.get("release_id") == current.get("release_id") == manifest.get("release_id") else 0,
            1,
        ),
        "current_digest_matches": _digest_check(artifacts.get("current_digest"), artifact_digest(current)),
        "manifest_digest_matches": _digest_check(artifacts.get("manifest_digest"), artifact_digest(manifest)),
        "state_digest_matches": _digest_check(artifacts.get("state_digest"), artifact_digest(state)),
        "current_verification_digest_matches": _digest_check(
            artifacts.get("current_verification_digest"),
            artifact_digest(current_verification),
        ),
        "registry_digest_matches": _optional_digest_check(artifacts.get("registry_digest"), artifact_digest(registry) if registry is not None else None),
        "graphs_digest_matches": _optional_digest_check(artifacts.get("graphs_digest"), graphs_digest),
        "model_config_digest_matches": _optional_digest_check(artifacts.get("model_config_digest"), model_config_digest),
        "not_expired": _promotion_check(
            not require_not_expired or not expired,
            0 if expired else 1,
            1,
        ),
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "format": "tune-orchestrator-bandit-runtime-bundle-verification-v1",
        "status": "pass" if passed else "fail",
        "bundle_id": bundle.get("bundle_id"),
        "release_id": bundle.get("release_id"),
        "channel": bundle.get("channel"),
        "expected": artifacts,
        "actual": {
            "current_digest": artifact_digest(current),
            "manifest_digest": artifact_digest(manifest),
            "state_digest": artifact_digest(state),
            "current_verification_digest": artifact_digest(current_verification),
            "registry_digest": artifact_digest(registry) if registry is not None else None,
            "graphs_digest": graphs_digest,
            "model_config_digest": model_config_digest,
        },
        "expires_at": bundle.get("expires_at"),
        "checks": checks,
    }


def write_bandit_current_release(path: Path, current: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def append_bandit_release_registry(
    *,
    registry: dict[str, Any] | None,
    current: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    existing = registry if isinstance(registry, dict) else {}
    entries = existing.get("entries", [])
    entries = list(entries) if isinstance(entries, list) else []
    entry = _release_registry_entry(current=current, manifest=manifest)
    deduped = [
        item
        for item in entries
        if not (
            isinstance(item, dict)
            and item.get("channel") == entry["channel"]
            and item.get("release_id") == entry["release_id"]
            and item.get("manifest_digest") == entry["manifest_digest"]
        )
    ]
    deduped.append(entry)
    return {
        "format": "tune-orchestrator-bandit-release-registry-v1",
        "updated_at": _utc_now(),
        "entries": deduped,
        "summary": {
            "entries": len(deduped),
            "channels": sorted({str(item.get("channel")) for item in deduped if isinstance(item, dict)}),
            "latest_release_id": entry["release_id"],
            "latest_channel": entry["channel"],
        },
    }


def select_bandit_rollback_release(
    *,
    registry: dict[str, Any],
    current_release_id: str | None = None,
    channel: str = "production",
    require_not_expired: bool = True,
) -> dict[str, Any]:
    entries = registry.get("entries", [])
    entries = [item for item in entries if isinstance(item, dict) and item.get("channel") == channel]
    ordered = sorted(enumerate(entries), key=lambda pair: (_parse_utc(str(pair[1].get("activated_at", ""))) or datetime.min.replace(tzinfo=UTC), pair[0]))
    latest = ordered[-1][1] if ordered else None
    active_release_id = current_release_id or (latest or {}).get("release_id")
    reasons: list[str] = []
    candidates: list[tuple[datetime, int, dict[str, Any]]] = []
    now = datetime.now(UTC)
    for index, entry in ordered:
        release_id = entry.get("release_id")
        if active_release_id and release_id == active_release_id:
            continue
        if entry.get("status") != "pass":
            reasons.append(f"skip {release_id}: status is {entry.get('status')}")
            continue
        if not entry.get("runtime_enabled"):
            reasons.append(f"skip {release_id}: runtime is disabled")
            continue
        if not entry.get("manifest_path") or not entry.get("manifest_digest"):
            reasons.append(f"skip {release_id}: manifest reference is incomplete")
            continue
        expires_at = _parse_utc(str(entry.get("expires_at", "")))
        if require_not_expired and (expires_at is None or now >= expires_at):
            reasons.append(f"skip {release_id}: release is expired")
            continue
        activated_at = _parse_utc(str(entry.get("activated_at", ""))) or datetime.min.replace(tzinfo=UTC)
        candidates.append((activated_at, index, entry))
    candidate = sorted(candidates, key=lambda item: (item[0], item[1]))[-1][2] if candidates else None
    return {
        "format": "tune-orchestrator-bandit-rollback-candidate-v1",
        "status": "pass" if candidate is not None else "fail",
        "channel": channel,
        "current_release_id": active_release_id,
        "selected_release_id": candidate.get("release_id") if candidate else None,
        "candidate": candidate,
        "reasons": ["rollback candidate selected"] if candidate else (reasons or ["no release history for channel"]),
    }


def write_bandit_release_registry(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def write_bandit_rollback_report(path: Path, rollback: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = rollback.get("candidate") or {}
    lines = [
        "# Bandit Rollback Candidate",
        "",
        f"- Status: {rollback['status']}",
        f"- Channel: `{rollback.get('channel')}`",
        f"- Current release: `{rollback.get('current_release_id')}`",
        f"- Selected release: `{rollback.get('selected_release_id')}`",
        "",
    ]
    if candidate:
        lines.extend(
            [
                "## Candidate",
                "",
                f"- Activated at: `{candidate.get('activated_at')}`",
                f"- Manifest: `{candidate.get('manifest_path')}`",
                f"- Manifest digest: `{candidate.get('manifest_digest')}`",
                f"- Traffic: {float(candidate.get('traffic_percent', 0.0)):.1f}%",
                f"- Expires at: `{candidate.get('expires_at')}`",
                "",
            ]
        )
    lines.extend(["## Reasons", ""])
    lines.extend(f"- {reason}" for reason in rollback.get("reasons", []))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def validate_bandit_rollback_candidate(
    *,
    rollback: dict[str, Any],
    manifest: dict[str, Any],
    state: dict[str, Any] | None = None,
    require_not_expired: bool = True,
) -> dict[str, Any]:
    candidate = rollback.get("candidate", {})
    candidate = candidate if isinstance(candidate, dict) else {}
    expected_manifest = candidate.get("manifest_digest")
    actual_manifest = artifact_digest(manifest)
    manifest_artifacts = manifest.get("artifacts", {})
    manifest_artifacts = manifest_artifacts if isinstance(manifest_artifacts, dict) else {}
    expected_state = candidate.get("state_digest") or manifest_artifacts.get("bandit_state_digest")
    actual_state = artifact_digest(state) if state is not None else None
    expires_at = _parse_utc(str(candidate.get("expires_at") or manifest.get("expires_at") or ""))
    expired = expires_at is None or datetime.now(UTC) >= expires_at
    checks = {
        "rollback_format": _promotion_check(
            rollback.get("format") == "tune-orchestrator-bandit-rollback-candidate-v1",
            1 if rollback.get("format") == "tune-orchestrator-bandit-rollback-candidate-v1" else 0,
            1,
        ),
        "rollback_status_pass": _promotion_check(
            rollback.get("status") == "pass",
            1 if rollback.get("status") == "pass" else 0,
            1,
        ),
        "release_id_matches": _promotion_check(
            bool(candidate.get("release_id")) and candidate.get("release_id") == manifest.get("release_id"),
            1 if candidate.get("release_id") == manifest.get("release_id") else 0,
            1,
        ),
        "manifest_digest_matches": _promotion_check(
            bool(expected_manifest) and expected_manifest == actual_manifest,
            1 if expected_manifest == actual_manifest else 0,
            1,
        ),
        "manifest_status_pass": _promotion_check(
            manifest.get("status") == "pass",
            1 if manifest.get("status") == "pass" else 0,
            1,
        ),
        "state_digest_matches": _promotion_check(
            state is None or (bool(expected_state) and expected_state == actual_state),
            1 if expected_state == actual_state else 0,
            1,
        ),
        "runtime_enabled": _promotion_check(
            bool(candidate.get("runtime_enabled")),
            1 if candidate.get("runtime_enabled") else 0,
            1,
        ),
        "not_expired": _promotion_check(
            not require_not_expired or not expired,
            0 if expired else 1,
            1,
        ),
    }
    passed = all(item["passed"] for item in checks.values())
    return {
        "format": "tune-orchestrator-bandit-rollback-verification-v1",
        "status": "pass" if passed else "fail",
        "channel": rollback.get("channel"),
        "current_release_id": rollback.get("current_release_id"),
        "selected_release_id": rollback.get("selected_release_id"),
        "expected": {
            "manifest_digest": expected_manifest,
            "state_digest": expected_state,
        },
        "actual": {
            "manifest_digest": actual_manifest,
            "state_digest": actual_state,
        },
        "manifest_path": candidate.get("manifest_path"),
        "expires_at": candidate.get("expires_at") or manifest.get("expires_at"),
        "checks": checks,
    }


def build_bandit_rollback_current_release(
    *,
    rollback: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: str,
    channel: str | None = None,
) -> dict[str, Any]:
    current = build_bandit_current_release(
        manifest=manifest,
        manifest_path=manifest_path,
        channel=channel or str(rollback.get("channel") or "production"),
    )
    current["rollback"] = {
        "rolled_back_at": current["activated_at"],
        "from_release_id": rollback.get("current_release_id"),
        "to_release_id": manifest.get("release_id"),
        "rollback_digest": artifact_digest(rollback),
    }
    return current


def write_bandit_rollout_report(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Bandit Rollout Plan",
        "",
        f"- Action: {plan['action']}",
        f"- Current traffic: {float(plan['current_traffic_percent']):.1f}%",
        f"- Target traffic: {float(plan['target_traffic_percent']):.1f}%",
        f"- Max traffic: {float(plan['max_traffic_percent']):.1f}%",
        f"- Runtime enabled: {str(plan['runtime']['enabled']).lower()}",
        f"- Rollout salt: `{plan['runtime']['rollout_salt']}`",
        f"- Created at: `{plan.get('created_at')}`",
        f"- Expires at: `{plan.get('expires_at')}`",
        "",
        "## Artifacts",
        "",
        f"- Bandit state digest: `{plan.get('artifacts', {}).get('bandit_state_digest')}`",
        f"- Promotion digest: `{plan.get('artifacts', {}).get('promotion_digest')}`",
        f"- Monitor digest: `{plan.get('artifacts', {}).get('monitor_digest')}`",
        "",
        "## Inputs",
        "",
        f"- Promotion status: `{plan['inputs']['promotion_status']}`",
        f"- Monitor status: `{plan['inputs']['monitor_status']}`",
        f"- Monitor bandit traces: {plan['inputs']['monitor_bandit_traces']}",
        "",
        "## Reasons",
        "",
    ]
    lines.extend(f"- {reason}" for reason in plan.get("reasons", []))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def apply_bandit_policy(
    *,
    text: str,
    signal: RouterSignal,
    served_decision: RouteDecision,
    risk_level: str,
    selector: GraphSelector,
    state: dict[str, Any],
    config: BanditPolicyConfig | None = None,
) -> RouteDecision:
    cfg = config or BanditPolicyConfig()
    if served_decision.policy.action != "allow":
        return served_decision
    if served_decision.policy.risk_level == "high" and not cfg.include_high_risk:
        return served_decision

    candidates = [("served", served_decision)] + build_shadow_decisions(
        text=text,
        signal=signal,
        served_decision=served_decision,
        risk_level=risk_level,
        selector=selector,
        config=ShadowConfig(
            mode="alternatives",
            max_count=3,
            low_risk_only=not cfg.include_high_risk,
        ),
    )
    context = context_key_from_signal(signal, served_decision.policy.risk_level)
    context_state = state.get("contexts", {}).get(context)
    if not isinstance(context_state, dict) or int(context_state.get("observations", 0)) < cfg.min_observations:
        return served_decision

    scored = _score_candidates(candidates, context_state, cfg)
    if not scored:
        return served_decision
    best = max(scored, key=lambda item: (item["score"], item["observed"], item["reason"]))
    selected = best["decision"]
    if selected is served_decision or _decision_key(selected) == _decision_key(served_decision):
        return served_decision
    traffic_percent = max(0.0, min(100.0, cfg.traffic_percent))
    canary_bucket = _canary_bucket(cfg.rollout_salt, text, context, best["arm_key"])
    canary = {
        "context_key": context,
        "candidate_arm_key": best["arm_key"],
        "candidate_graph_id": selected.graph_id,
        "candidate_selector_type": selected.selector_type,
        "candidate_score": best["score"],
        "candidate_observations": best["observed"],
        "candidate_reason": best["reason"],
        "traffic_percent": traffic_percent,
        "bucket": canary_bucket,
        "sampled": canary_bucket < traffic_percent,
        "rollout_config_digest": cfg.rollout_digest,
        "release_manifest_digest": cfg.release_digest,
        "release_id": cfg.release_id,
        "release_current_digest": cfg.release_current_digest,
    }
    if not canary["sampled"]:
        return replace(
            served_decision,
            selection_metadata={
                **served_decision.selection_metadata,
                "bandit_canary": canary,
            },
        )
    return replace(
        selected,
        selector_type="bandit_policy",
        selection_metadata={
            **selected.selection_metadata,
            "bandit": {
                "context_key": context,
                "arm_key": best["arm_key"],
                "score": best["score"],
                "observed": best["observed"],
                "served_selector_type": served_decision.selector_type,
                "served_graph_id": served_decision.graph_id,
                "candidate_reason": best["reason"],
                "traffic_percent": traffic_percent,
                "canary_bucket": canary_bucket,
                "canary_sampled": True,
                "rollout_config_digest": cfg.rollout_digest,
                "release_manifest_digest": cfg.release_digest,
                "release_id": cfg.release_id,
                "release_current_digest": cfg.release_current_digest,
            },
        },
    )


def reward_from_trace(trace: dict[str, Any], config: BanditBuildConfig | None = None) -> float | None:
    cfg = config or BanditBuildConfig()
    base = _review_score(trace)
    if base is None:
        return None
    usage = trace.get("usage", {})
    cost = float(usage.get("cost_usd", 0.0))
    latency_seconds = float(usage.get("latency_ms", 0.0)) / 1000
    reward = base - cfg.cost_weight * cost - cfg.latency_weight * latency_seconds
    if not _trace_success(trace):
        reward -= 0.25
    return max(-1.0, min(1.0, reward))


def context_key_from_trace(trace: dict[str, Any]) -> str | None:
    router = trace.get("router", {})
    scores = router.get("scores")
    if not isinstance(scores, dict):
        return None
    risk = str(trace.get("policy", {}).get("risk_level", "normal"))
    return context_key_from_signal(RouterSignal(scores={str(key): float(value) for key, value in scores.items()}), risk)


def context_key_from_signal(signal: RouterSignal, risk_level: str) -> str:
    ranked = sorted(signal.scores.items(), key=lambda item: (-float(item[1]), item[0]))
    top = ranked[0][0] if ranked else "General"
    secondary = ",".join(label for label, score in ranked[1:3] if float(score) >= 0.15) or "none"
    return f"risk={risk_level}|top={top}|secondary={secondary}"


def arm_key_from_trace(trace: dict[str, Any]) -> str | None:
    graph = trace.get("graph", {})
    graph_id = graph.get("id")
    if not graph_id:
        return None
    labels = tuple(trace.get("router", {}).get("primary_labels", ())) + tuple(trace.get("router", {}).get("secondary_labels", ()))
    delegations = tuple(
        (item.get("label"), item.get("model"))
        for item in graph.get("delegations", ())
        if isinstance(item, dict)
    )
    generated = graph.get("generated_graph")
    generated_digest = _stable_digest(generated) if generated else "none"
    return _arm_key(str(graph_id), labels, delegations, generated_digest)


def arm_key_from_decision(decision: RouteDecision) -> str:
    generated_digest = _stable_digest(decision.generated_graph) if decision.generated_graph else "none"
    delegations = tuple((item.label, item.model) for item in decision.delegations)
    return _arm_key(decision.graph_id, decision.selected_labels, delegations, generated_digest)


def _score_candidates(
    candidates: list[tuple[str, RouteDecision]],
    context_state: dict[str, Any],
    cfg: BanditPolicyConfig,
) -> list[dict[str, Any]]:
    arms = context_state.get("arms", {})
    total = max(1, int(context_state.get("observations", 0)))
    scored = []
    for reason, decision in candidates:
        arm_key = arm_key_from_decision(decision)
        stats = arms.get(arm_key)
        if not isinstance(stats, dict):
            if not cfg.explore_unobserved:
                continue
            score = -0.05 + cfg.exploration_weight
            observed = 0
        else:
            pulls = max(1, int(stats.get("pulls", 0)))
            score = float(stats.get("mean_reward", 0.0)) + cfg.exploration_weight * math.sqrt(math.log(total + 1) / pulls)
            observed = pulls
        scored.append({"reason": reason, "decision": decision, "arm_key": arm_key, "score": score, "observed": observed})
    return scored


def _arm_key(
    graph_id: str,
    labels: tuple[Any, ...],
    delegations: tuple[tuple[Any, Any], ...],
    generated_digest: str,
) -> str:
    label_value = ",".join(str(item) for item in labels) or "none"
    delegation_value = ",".join(f"{label}:{model}" for label, model in delegations) or "none"
    return f"graph={graph_id}|labels={label_value}|delegations={delegation_value}|generated={generated_digest}"


def _decision_key(decision: RouteDecision) -> tuple[Any, ...]:
    return (decision.graph_id, tuple(decision.selected_labels), tuple((item.label, item.model) for item in decision.delegations), _stable_digest(decision.generated_graph))


def _stable_digest(value: Any) -> str:
    if value is None:
        return "none"
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _review_score(trace: dict[str, Any]) -> float | None:
    evaluation = trace.get("evaluation", {})
    rating = evaluation.get("user_rating")
    if isinstance(rating, (int, float)):
        return max(0.0, min(1.0, float(rating) / 5.0))
    label = str(evaluation.get("review_label") or "").lower()
    if label == "preferred":
        return 1.0
    if label in {"approved", "success"}:
        return 0.85
    if label in {"rejected", "failed", "failure"}:
        return 0.0
    return None


def _trace_success(trace: dict[str, Any]) -> bool:
    return trace.get("graph", {}).get("stop_reason") in {"completed", "verifier_passed", "repaired_and_verified"}


def _decision_summary_from_trace(trace: dict[str, Any]) -> dict[str, Any]:
    graph = trace.get("graph", {})
    return {
        "graph_id": graph.get("id"),
        "selector_type": graph.get("selector_type"),
        "delegations": graph.get("delegations", []),
        "generated_graph": graph.get("generated_graph"),
    }


def _replay_request(trace: dict[str, Any], cfg: BanditBuildConfig) -> dict[str, Any] | None:
    context = context_key_from_trace(trace)
    served_arm = arm_key_from_trace(trace)
    served_reward = reward_from_trace(trace, cfg)
    if not context or not served_arm or served_reward is None:
        return None
    served = _replay_candidate("served", trace, served_arm, served_reward)
    candidates = [served]
    seen = {served_arm}
    for shadow in trace.get("shadow_executions", []):
        if not isinstance(shadow, dict) or shadow.get("status") != "completed":
            continue
        shadow_trace = shadow.get("trace")
        if not isinstance(shadow_trace, dict):
            continue
        arm = arm_key_from_trace(shadow_trace)
        reward = reward_from_trace(shadow_trace, cfg)
        if not arm or reward is None or arm in seen:
            continue
        seen.add(arm)
        candidates.append(_replay_candidate(f"shadow:{shadow.get('reason', 'unknown')}", shadow_trace, arm, reward))
    return {
        "trace_id": str(trace.get("trace_id", "")),
        "context_key": context,
        "risk_level": str(trace.get("policy", {}).get("risk_level", "normal")),
        "served": served,
        "candidates": candidates,
    }


def _replay_candidate(source: str, trace: dict[str, Any], arm_key: str, reward: float) -> dict[str, Any]:
    graph = trace.get("graph", {})
    return {
        "source": source,
        "trace_id": str(trace.get("trace_id", "")),
        "arm_key": arm_key,
        "reward": reward,
        "graph_id": graph.get("id"),
        "selector_type": graph.get("selector_type"),
    }


def _score_replay_candidates(
    candidates: list[dict[str, Any]],
    context_state: dict[str, Any],
    cfg: BanditReplayConfig,
) -> list[dict[str, Any]]:
    arms = context_state.get("arms", {})
    total = max(1, int(context_state.get("observations", 0)))
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        stats = arms.get(candidate["arm_key"])
        if not isinstance(stats, dict):
            if not cfg.explore_unobserved:
                continue
            score = -0.05 + cfg.exploration_weight
            observed = 0
        else:
            pulls = max(1, int(stats.get("pulls", 0)))
            score = float(stats.get("mean_reward", 0.0)) + cfg.exploration_weight * math.sqrt(math.log(total + 1) / pulls)
            observed = pulls
        scored.append(
            {
                "candidate": candidate,
                "source": candidate["source"],
                "arm_key": candidate["arm_key"],
                "score": score,
                "observed": observed,
            }
        )
    return scored


def _remove_request_observations(context_state: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    arms: dict[str, Any] = {
        arm_key: dict(stats)
        for arm_key, stats in context_state.get("arms", {}).items()
        if isinstance(stats, dict)
    }
    observations = int(context_state.get("observations", 0))
    for candidate in candidates:
        stats = arms.get(candidate["arm_key"])
        if not isinstance(stats, dict):
            continue
        pulls = int(stats.get("pulls", 0))
        if pulls <= 0:
            continue
        pulls -= 1
        observations = max(0, observations - 1)
        reward_sum = float(stats.get("reward_sum", 0.0)) - float(candidate["reward"])
        if pulls <= 0:
            arms.pop(candidate["arm_key"], None)
            continue
        stats["pulls"] = pulls
        stats["reward_sum"] = reward_sum
        stats["mean_reward"] = reward_sum / pulls
    return {"observations": observations, "arms": arms}


def _skipped_replay_detail(request: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "trace_id": request["trace_id"],
        "context_key": request["context_key"],
        "risk_level": request["risk_level"],
        "skipped_reason": reason,
        "candidate_count": len(request["candidates"]),
    }


def _build_config_from_state(state: dict[str, Any]) -> BanditBuildConfig:
    config = state.get("config", {})
    if not isinstance(config, dict):
        return BanditBuildConfig()
    defaults = BanditBuildConfig()
    return BanditBuildConfig(
        cost_weight=float(config.get("cost_weight", defaults.cost_weight)),
        latency_weight=float(config.get("latency_weight", defaults.latency_weight)),
    )


def _canary_bucket(salt: str, text: str, context_key: str, arm_key: str) -> float:
    digest = hashlib.sha256(f"{salt}\n{text}\n{context_key}\n{arm_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 * 100


def _release_registry_entry(*, current: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    runtime = current.get("runtime") or manifest.get("runtime") or {}
    runtime = runtime if isinstance(runtime, dict) else {}
    artifacts = manifest.get("artifacts", {})
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    return {
        "activated_at": current.get("activated_at") or _utc_now(),
        "channel": str(current.get("channel") or "production"),
        "release_id": str(current.get("release_id") or manifest.get("release_id") or ""),
        "manifest_path": str(current.get("manifest_path") or ""),
        "manifest_digest": str(current.get("manifest_digest") or artifact_digest(manifest)),
        "current_digest": artifact_digest(current),
        "state_digest": artifacts.get("bandit_state_digest"),
        "rollout_digest": artifacts.get("rollout_digest"),
        "verification_digest": artifacts.get("verification_digest"),
        "runtime_enabled": bool(runtime.get("enabled", False)),
        "traffic_percent": float(runtime.get("traffic_percent", 0.0)),
        "rollout_salt": str(runtime.get("rollout_salt", "default")),
        "expires_at": current.get("expires_at") or manifest.get("expires_at"),
        "status": manifest.get("status"),
        "rollout_action": manifest.get("rollout_action"),
    }


def _find_release_registry_entry(*, current: dict[str, Any], registry: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(registry, dict):
        return None
    entries = registry.get("entries", [])
    if not isinstance(entries, list):
        return None
    release_id = current.get("release_id")
    channel = current.get("channel")
    manifest_digest = current.get("manifest_digest")
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("release_id") == release_id
        and entry.get("channel") == channel
        and entry.get("manifest_digest") == manifest_digest
    ]
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda entry: _parse_utc(str(entry.get("activated_at", ""))) or datetime.min.replace(tzinfo=UTC),
    )[-1]


def _digest_check(expected: Any, actual: Any) -> dict[str, Any]:
    return _promotion_check(bool(expected) and expected == actual, 1 if expected == actual else 0, 1)


def _optional_digest_check(expected: Any, actual: Any) -> dict[str, Any]:
    if expected is None:
        return _promotion_check(True, 1, 1)
    return _promotion_check(expected == actual, 1 if expected == actual else 0, 1)


def artifact_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return _format_utc(datetime.now(UTC))


def _format_utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _is_bandit_selected_trace(trace: dict[str, Any]) -> bool:
    graph = trace.get("graph", {})
    metadata = graph.get("selection_metadata", {})
    return graph.get("selector_type") == "bandit_policy" or isinstance(metadata.get("bandit"), dict)


def _bandit_candidate_metadata(trace: dict[str, Any]) -> dict[str, Any] | None:
    metadata = trace.get("graph", {}).get("selection_metadata", {})
    bandit = metadata.get("bandit")
    if isinstance(bandit, dict):
        return {**bandit, "sampled": True}
    canary = metadata.get("bandit_canary")
    return canary if isinstance(canary, dict) else None


def _rollout_trace_stats(traces: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(trace.get("usage", {}).get("latency_ms", 0.0)) for trace in traces]
    costs = [float(trace.get("usage", {}).get("cost_usd", 0.0)) for trace in traces]
    failures = [trace for trace in traces if not _trace_success(trace)]
    return {
        "traces": len(traces),
        "failure_rate": len(failures) / len(traces) if traces else 0.0,
        "latency_p50_ms": _percentile(latencies, 0.50),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "mean_cost_usd": _mean(costs),
    }


def _stop_reason_counts(traces: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trace in traces:
        reason = str(trace.get("graph", {}).get("stop_reason", "missing"))
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


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


def _promotion_check(passed: bool, actual: float | int, threshold: float | int) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "actual": actual,
        "threshold": threshold,
    }


def _artifact_match(expected: str | None, actual: str | None, required: bool) -> bool:
    if expected and actual:
        return expected == actual
    if expected and actual is None:
        return False
    if required:
        return False
    return True


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1
