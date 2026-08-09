from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from bandit import (
    BanditBuildConfig,
    BanditMonitorConfig,
    BanditPolicyConfig,
    BanditPromotionConfig,
    BanditReplayConfig,
    BanditRolloutPlanConfig,
    apply_bandit_policy,
    append_bandit_release_registry,
    artifact_digest,
    build_bandit_current_release,
    build_bandit_release_manifest,
    build_bandit_rollback_current_release,
    build_bandit_runtime_bundle,
    build_bandit_state,
    evaluate_bandit_promotion,
    load_bandit_state,
    monitor_bandit_rollout,
    plan_bandit_rollout,
    replay_bandit_policy,
    select_bandit_rollback_release,
    validate_bandit_current_artifacts,
    validate_bandit_runtime_bundle,
    validate_bandit_rollout_artifacts,
    validate_bandit_rollout_binding,
    validate_bandit_release_manifest,
    validate_bandit_current_release,
    validate_bandit_rollback_candidate,
    write_bandit_artifact_verification_report,
    write_bandit_current_release,
    write_bandit_release_registry,
    write_bandit_release_manifest,
    write_bandit_release_report,
    write_bandit_monitor_report,
    write_bandit_promotion_report,
    write_bandit_replay_report,
    write_bandit_rollback_report,
    write_bandit_rollout_plan,
    write_bandit_rollout_report,
    write_bandit_state,
)
from clients import MockModelClient, OpenAIModelClient, OpenAIRouterClient
from composition import graph_from_bounded_plan
from evaluation import (
    evaluate_offline,
    load_records,
    summarize_traces,
    write_evaluation_outputs,
    write_trace_report,
)
from executor import GraphExecutor
from ft_data import FTDataConfig, build_ft_datasets, write_ft_datasets
from graphs import load_graphs
from learned import LearnedGraphSelector, OpenAIPlanClient
from models import Budget, RouterSignal
from ops import run_preflight
from router_learning import (
    RouterContinualConfig,
    RouterDataConfig,
    RouterMergeConfig,
    build_router_continual_dataset,
    build_router_pretrain_dataset,
    evaluate_router_prototype,
    load_router_records,
    merge_router_training_datasets,
    predict_router_prototype,
    train_router_lora,
    train_router_prototype,
    write_json,
    write_router_dataset,
)
from selector import GraphSelector, SelectionPolicy
from shadow import ShadowConfig, build_shadow_decisions, execute_shadow_decisions
from training import (
    DEFAULT_ORCHESTRATOR_MODEL,
    LocalAdapterPlanClient,
    evaluate_adapter,
    read_jsonl,
    train_orchestrator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_GRAPHS = PROJECT_ROOT / "graphs"
PACKAGED_GRAPHS = Path(__file__).resolve().parent / "graph_definitions"
DEFAULT_GRAPHS = SOURCE_GRAPHS if SOURCE_GRAPHS.exists() else PACKAGED_GRAPHS
DEFAULT_ARTIFACTS = PROJECT_ROOT / "artifacts"
DEFAULT_RUNTIME_ARTIFACTS = DEFAULT_ARTIFACTS / "runtime"
DEFAULT_FT_DATA = DEFAULT_ARTIFACTS / "ft-data"
DEFAULT_FT_ADAPTER = DEFAULT_ARTIFACTS / "orchestrator-lora"
DEFAULT_TRACE = DEFAULT_RUNTIME_ARTIFACTS / "traces.jsonl"


def cmd_select(args: argparse.Namespace) -> None:
    signal = _router_signal(args)
    decision = _decision(args, signal, Budget())
    print(json.dumps({"router": _signal_dict(signal), "decision": decision.to_dict()}, ensure_ascii=False, indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    signal = _router_signal(args)
    model_config = _load_document(Path(args.model_config)) if args.model_config else None
    allowed_models = set(model_config.get("models", {})) if model_config else None
    model_catalog = _model_catalog(model_config) if model_config else None
    budget = _budget_from_args(args)
    decision = _decision(args, signal, budget, allowed_models, model_catalog)
    decision = _apply_bandit(args, signal, decision)
    _execute_selected_graph(args, signal, decision, budget, model_config)


def cmd_run_ft(args: argparse.Namespace) -> None:
    signal = _router_signal(args)
    model_config = _load_document(Path(args.model_config)) if args.model_config else None
    allowed_models = set(model_config.get("models", {})) if model_config else None
    model_catalog = _model_catalog(model_config) if model_config else None
    budget = _budget_from_args(args)
    client = LocalAdapterPlanClient(
        Path(args.adapter),
        base_model=args.base_model,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    decision = LearnedGraphSelector(
        client,
        fallback=_selector(args),
        allowed_models=allowed_models,
        model_catalog=model_catalog,
    ).select(args.text, signal, args.risk, budget)
    decision = _apply_bandit(args, signal, decision)
    _execute_selected_graph(args, signal, decision, budget, model_config)


def _budget_from_args(args: argparse.Namespace) -> Budget:
    return Budget(
        max_cost_usd=args.max_cost,
        max_latency_ms=args.max_latency_ms,
        max_steps=args.max_steps,
    )


def _execute_selected_graph(
    args: argparse.Namespace,
    signal: RouterSignal,
    decision,
    budget: Budget,
    model_config: dict[str, Any] | None,
) -> None:
    graphs = load_graphs(Path(args.graphs))
    if decision.generated_graph:
        graph = graph_from_bounded_plan(decision.generated_graph)
    else:
        if decision.graph_id not in graphs:
            raise SystemExit(f"graph definition is missing: {decision.graph_id}")
        graph = graphs[decision.graph_id]

    if args.mock:
        model_client = MockModelClient()
    elif model_config:
        model_client = OpenAIModelClient(model_config)
    else:
        raise SystemExit("run requires --model-config or --mock")

    result = GraphExecutor(model_client, max_workers=args.max_workers).execute(
        text=args.text,
        signal=signal,
        decision=decision,
        graph=graph,
        budget=budget,
    )
    shadow_config = _shadow_config_from_args(args)
    shadows = build_shadow_decisions(
        text=args.text,
        signal=signal,
        served_decision=decision,
        risk_level=args.risk,
        selector=_selector(args),
        config=shadow_config,
    )
    if shadows:
        result.trace["shadow_executions"] = execute_shadow_decisions(
            text=args.text,
            signal=signal,
            model_client=model_client,
            graphs=graphs,
            shadows=shadows,
            config=shadow_config,
            max_workers=args.max_workers,
        )
    if args.trace:
        _append_jsonl(Path(args.trace), result.trace)
    output: dict[str, Any] = {
        "trace_id": result.trace["trace_id"],
        "graph": decision.graph_id,
        "final_answer": result.final_answer,
        "usage": result.trace["usage"],
        "stop_reason": result.trace["graph"]["stop_reason"],
    }
    if args.debug:
        output["trace"] = result.trace
    print(json.dumps(output, ensure_ascii=False, indent=2))


def _shadow_config_from_args(args: argparse.Namespace) -> ShadowConfig:
    return ShadowConfig(
        mode=args.shadow_mode,
        max_count=args.shadow_max_count,
        max_cost_usd=args.shadow_max_cost,
        max_latency_ms=args.shadow_max_latency_ms,
        low_risk_only=not args.shadow_include_high_risk,
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    candidates = load_records(Path(args.candidate_results))
    predictions = load_records(Path(args.predictions)) if args.predictions else None
    summaries, details, pareto = evaluate_offline(candidates, predictions)
    out_dir = Path(args.out)
    write_evaluation_outputs(out_dir, summaries, details, pareto)
    print(json.dumps({"out": str(out_dir), "routers": summaries}, ensure_ascii=False, indent=2))


def cmd_build_bandit_state(args: argparse.Namespace) -> None:
    traces = load_records(Path(args.traces))
    state = build_bandit_state(
        traces,
        BanditBuildConfig(
            cost_weight=args.cost_weight,
            latency_weight=args.latency_weight,
        ),
    )
    write_bandit_state(Path(args.out), state)
    print(json.dumps({"out": args.out, "summary": state["summary"]}, ensure_ascii=False, indent=2))


def cmd_replay_bandit(args: argparse.Namespace) -> None:
    traces = load_records(Path(args.traces))
    state = load_bandit_state(Path(args.bandit_state))
    replay = replay_bandit_policy(
        traces,
        state,
        BanditReplayConfig(
            min_observations=args.min_observations,
            exploration_weight=args.exploration_weight,
            explore_unobserved=args.explore_unobserved,
            include_high_risk=args.include_high_risk,
            leave_one_out=not args.no_leave_one_out,
        ),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(replay, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    if args.report:
        write_bandit_replay_report(Path(args.report), replay)
    print(json.dumps({"out": args.out, "report": args.report, "summary": replay["summary"]}, ensure_ascii=False, indent=2))


def cmd_gate_bandit(args: argparse.Namespace) -> None:
    replay = _load_document(Path(args.replay))
    promotion = evaluate_bandit_promotion(
        replay,
        BanditPromotionConfig(
            min_evaluated_requests=args.min_evaluated_requests,
            min_mean_reward_delta=args.min_mean_reward_delta,
            max_loss_rate=args.max_loss_rate,
            max_switch_rate=args.max_switch_rate,
            max_skip_rate=args.max_skip_rate,
        ),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(promotion, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    if args.report:
        write_bandit_promotion_report(Path(args.report), promotion)
    print(json.dumps({"out": args.out, "report": args.report, **promotion}, ensure_ascii=False, indent=2))
    if promotion["status"] != "pass" and not args.no_fail:
        raise SystemExit(1)


def cmd_monitor_bandit(args: argparse.Namespace) -> None:
    traces = load_records(Path(args.traces))
    monitor = monitor_bandit_rollout(
        traces,
        BanditMonitorConfig(
            min_bandit_traces=args.min_bandit_traces,
            max_bandit_failure_rate=args.max_bandit_failure_rate,
            max_relative_failure_rate=args.max_relative_failure_rate,
            max_bandit_p95_latency_ms=args.max_bandit_p95_latency_ms,
            max_bandit_mean_cost_usd=args.max_bandit_mean_cost_usd,
        ),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(monitor, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    if args.report:
        write_bandit_monitor_report(Path(args.report), monitor)
    print(json.dumps({"out": args.out, "report": args.report, **monitor}, ensure_ascii=False, indent=2))
    if monitor["status"] != "pass" and not args.no_fail:
        raise SystemExit(1)


def cmd_plan_bandit_rollout(args: argparse.Namespace) -> None:
    promotion = _load_document(Path(args.promotion)) if args.promotion else None
    monitor = _load_document(Path(args.monitor)) if args.monitor else None
    state = load_bandit_state(Path(args.bandit_state)) if args.bandit_state else None
    plan = plan_bandit_rollout(
        promotion=promotion,
        monitor=monitor,
        state=state,
        config=BanditRolloutPlanConfig(
            current_traffic_percent=args.current_traffic_percent,
            step_percent=args.step_percent,
            max_traffic_percent=args.max_traffic_percent,
            rollback_traffic_percent=args.rollback_traffic_percent,
            rollout_salt=args.rollout_salt,
            min_monitor_bandit_traces=args.min_monitor_bandit_traces,
            max_age_hours=args.max_age_hours,
        ),
    )
    write_bandit_rollout_plan(Path(args.out), plan)
    if args.report:
        write_bandit_rollout_report(Path(args.report), plan)
    print(json.dumps({"out": args.out, "report": args.report, **plan}, ensure_ascii=False, indent=2))
    if plan["action"] == "rollback" and not args.no_fail:
        raise SystemExit(1)


def cmd_verify_bandit_rollout(args: argparse.Namespace) -> None:
    rollout = _load_document(Path(args.rollout))
    state = load_bandit_state(Path(args.bandit_state))
    promotion = _load_document(Path(args.promotion)) if args.promotion else None
    monitor = _load_document(Path(args.monitor)) if args.monitor else None
    verification = validate_bandit_rollout_artifacts(
        rollout=rollout,
        state=state,
        promotion=promotion,
        monitor=monitor,
        require_state_binding=not args.allow_unbound_state,
        require_promotion_binding=not args.allow_missing_promotion,
        require_monitor_binding=args.require_monitor,
        require_not_expired=not args.allow_expired,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    if args.report:
        write_bandit_artifact_verification_report(Path(args.report), verification)
    print(json.dumps({"out": args.out, "report": args.report, **verification}, ensure_ascii=False, indent=2))
    if verification["status"] != "pass" and not args.no_fail:
        raise SystemExit(1)


def cmd_build_bandit_release(args: argparse.Namespace) -> None:
    rollout = _load_document(Path(args.rollout))
    state = load_bandit_state(Path(args.bandit_state))
    promotion = _load_document(Path(args.promotion)) if args.promotion else None
    monitor = _load_document(Path(args.monitor)) if args.monitor else None
    verification = _load_document(Path(args.verification)) if args.verification else None
    manifest = build_bandit_release_manifest(
        state=state,
        rollout=rollout,
        promotion=promotion,
        monitor=monitor,
        verification=verification,
        require_monitor=args.require_monitor,
    )
    write_bandit_release_manifest(Path(args.out), manifest)
    if args.report:
        write_bandit_release_report(Path(args.report), manifest)
    print(json.dumps({"out": args.out, "report": args.report, **manifest}, ensure_ascii=False, indent=2))
    if manifest["status"] != "pass" and not args.no_fail:
        raise SystemExit(1)


def cmd_activate_bandit_release(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    manifest = _load_document(manifest_path)
    state = load_bandit_state(Path(args.bandit_state))
    verification = validate_bandit_release_manifest(
        manifest=manifest,
        state=state,
        require_not_expired=not args.allow_expired,
    )
    if verification["status"] != "pass":
        print(json.dumps({"verification": verification}, ensure_ascii=False, indent=2))
        if not args.no_fail:
            raise SystemExit(1)
    current_path = Path(args.out)
    manifest_ref = _relative_or_absolute_manifest_path(manifest_path, current_path)
    current = build_bandit_current_release(
        manifest=manifest,
        manifest_path=manifest_ref,
        channel=args.channel,
    )
    write_bandit_current_release(current_path, current)
    print(json.dumps({"out": args.out, "verification": verification, **current}, ensure_ascii=False, indent=2))


def cmd_record_bandit_release(args: argparse.Namespace) -> None:
    current = _load_document(Path(args.current))
    manifest = _load_document(Path(args.manifest))
    verification = validate_bandit_current_release(
        current=current,
        manifest=manifest,
        require_not_expired=not args.allow_expired,
    )
    if verification["status"] != "pass":
        print(json.dumps({"verification": verification}, ensure_ascii=False, indent=2))
        if not args.no_fail:
            raise SystemExit(1)
    registry_path = Path(args.registry)
    registry = _load_document(registry_path) if registry_path.exists() else None
    updated = append_bandit_release_registry(registry=registry, current=current, manifest=manifest)
    write_bandit_release_registry(registry_path, updated)
    print(json.dumps({"registry": args.registry, "verification": verification, **updated}, ensure_ascii=False, indent=2))


def cmd_select_bandit_rollback(args: argparse.Namespace) -> None:
    registry = _load_document(Path(args.registry))
    rollback = select_bandit_rollback_release(
        registry=registry,
        current_release_id=args.current_release_id,
        channel=args.channel,
        require_not_expired=not args.allow_expired,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rollback, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    if args.report:
        write_bandit_rollback_report(Path(args.report), rollback)
    print(json.dumps({"out": args.out, "report": args.report, **rollback}, ensure_ascii=False, indent=2))
    if rollback["status"] != "pass" and not args.no_fail:
        raise SystemExit(1)


def cmd_apply_bandit_rollback(args: argparse.Namespace) -> None:
    rollback_path = Path(args.rollback)
    rollback = _load_document(rollback_path)
    manifest_path = Path(args.manifest) if args.manifest else _resolve_rollback_manifest_path(rollback, rollback_path)
    manifest = _load_document(manifest_path)
    state = load_bandit_state(Path(args.bandit_state)) if args.bandit_state else None
    verification = validate_bandit_rollback_candidate(
        rollback=rollback,
        manifest=manifest,
        state=state,
        require_not_expired=not args.allow_expired,
    )
    if verification["status"] != "pass":
        print(json.dumps({"verification": verification}, ensure_ascii=False, indent=2))
        if not args.no_fail:
            raise SystemExit(1)
    current_path = Path(args.out)
    manifest_ref = _relative_or_absolute_manifest_path(manifest_path, current_path)
    current = build_bandit_rollback_current_release(
        rollback=rollback,
        manifest=manifest,
        manifest_path=manifest_ref,
        channel=args.channel,
    )
    write_bandit_current_release(current_path, current)
    if args.registry:
        registry_path = Path(args.registry)
        registry = _load_document(registry_path) if registry_path.exists() else None
        registry = append_bandit_release_registry(registry=registry, current=current, manifest=manifest)
        write_bandit_release_registry(registry_path, registry)
    print(
        json.dumps(
            {"out": args.out, "registry": args.registry, "verification": verification, **current},
            ensure_ascii=False,
            indent=2,
        )
    )


def cmd_verify_bandit_current(args: argparse.Namespace) -> None:
    current_path = Path(args.current)
    current = _load_document(current_path)
    manifest_path = Path(args.manifest) if args.manifest else _resolve_current_manifest_path(current, current_path)
    manifest = _load_document(manifest_path)
    state = load_bandit_state(Path(args.bandit_state))
    registry = _load_document(Path(args.registry)) if args.registry else None
    verification = validate_bandit_current_artifacts(
        current=current,
        manifest=manifest,
        state=state,
        registry=registry,
        require_registry_entry=args.require_registry,
        require_not_expired=not args.allow_expired,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    print(json.dumps({"out": args.out, **verification}, ensure_ascii=False, indent=2))
    if verification["status"] != "pass" and not args.no_fail:
        raise SystemExit(1)


def cmd_build_bandit_runtime_bundle(args: argparse.Namespace) -> None:
    current_path = Path(args.current)
    current = _load_document(current_path)
    manifest_path = Path(args.manifest) if args.manifest else _resolve_current_manifest_path(current, current_path)
    manifest = _load_document(manifest_path)
    state = load_bandit_state(Path(args.bandit_state))
    verification = _load_document(Path(args.current_verification))
    registry = _load_document(Path(args.registry)) if args.registry else None
    graphs_digest = _path_digest(Path(args.graphs)) if args.graphs else None
    model_config_digest = _path_digest(Path(args.model_config)) if args.model_config else None
    out_path = Path(args.out)
    bundle = build_bandit_runtime_bundle(
        current=current,
        manifest=manifest,
        state=state,
        current_verification=verification,
        registry=registry,
        graphs_digest=graphs_digest,
        model_config_digest=model_config_digest,
        paths={
            "current": str(current_path),
            "manifest": str(manifest_path),
            "bandit_state": args.bandit_state,
            "current_verification": args.current_verification,
            "registry": args.registry,
            "graphs": args.graphs,
            "model_config": args.model_config,
        },
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    print(json.dumps({"out": args.out, **bundle}, ensure_ascii=False, indent=2))
    if bundle["status"] != "pass" and not args.no_fail:
        raise SystemExit(1)


def cmd_verify_bandit_runtime_bundle(args: argparse.Namespace) -> None:
    bundle = _load_document(Path(args.bundle))
    current_path = Path(args.current)
    current = _load_document(current_path)
    manifest_path = Path(args.manifest) if args.manifest else _resolve_current_manifest_path(current, current_path)
    manifest = _load_document(manifest_path)
    state = load_bandit_state(Path(args.bandit_state))
    current_verification = _load_document(Path(args.current_verification))
    registry = _load_document(Path(args.registry)) if args.registry else None
    verification = validate_bandit_runtime_bundle(
        bundle=bundle,
        current=current,
        manifest=manifest,
        state=state,
        current_verification=current_verification,
        registry=registry,
        graphs_digest=_path_digest(Path(args.graphs)) if args.graphs else None,
        model_config_digest=_path_digest(Path(args.model_config)) if args.model_config else None,
        require_not_expired=not args.allow_expired,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    print(json.dumps({"out": args.out, **verification}, ensure_ascii=False, indent=2))
    if verification["status"] != "pass" and not args.no_fail:
        raise SystemExit(1)


def cmd_validate_graphs(args: argparse.Namespace) -> None:
    graphs = load_graphs(Path(args.graphs))
    print(json.dumps({"valid": True, "graphs": sorted(graphs)}, indent=2))


def cmd_doctor(args: argparse.Namespace) -> None:
    model_config = _load_document(Path(args.model_config)) if args.model_config else None
    result = run_preflight(
        graphs_path=Path(args.graphs),
        router_url=args.router_url,
        router_model=args.router_model,
        router_timeout=args.router_timeout,
        model_config=model_config,
        probe_text=args.text,
        probe_model_endpoints=not args.skip_model_endpoints,
        probe_model_chat=args.probe_model_chat,
        adapter=Path(args.adapter) if args.adapter else None,
        orchestrator_url=args.orchestrator_url,
        orchestrator_model=args.orchestrator_model,
        orchestrator_api_key_env=args.orchestrator_api_key_env,
        orchestrator_timeout=args.orchestrator_timeout,
        probe_orchestrator=args.probe_orchestrator,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "fail" and not args.no_fail:
        raise SystemExit(1)


def cmd_trace_report(args: argparse.Namespace) -> None:
    summary = summarize_traces(load_records(Path(args.traces)))
    write_trace_report(Path(args.out), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cmd_prepare_ft_data(args: argparse.Namespace) -> None:
    candidates = load_records(Path(args.candidate_results))
    traces = load_records(Path(args.traces)) if args.traces else None
    datasets = build_ft_datasets(
        candidates,
        traces,
        FTDataConfig(
            quality_tolerance=args.quality_tolerance,
            cost_weight=args.cost_weight,
            latency_weight=args.latency_weight,
            dev_ratio=args.dev_ratio,
            seed=args.seed,
        ),
    )
    write_ft_datasets(Path(args.out), datasets)
    print(json.dumps(datasets["summary"], ensure_ascii=False, indent=2))


def cmd_prepare_router_pretrain(args: argparse.Namespace) -> None:
    records = load_router_records(Path(args.source))
    dataset = build_router_pretrain_dataset(
        records,
        RouterDataConfig(
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio,
            max_per_label=args.max_per_label,
            min_text_chars=args.min_text_chars,
            seed=args.seed,
        ),
    )
    write_router_dataset(Path(args.out), dataset)
    print(json.dumps({"out": args.out, **dataset["metadata"]}, ensure_ascii=False, indent=2))


def cmd_prepare_router_continual(args: argparse.Namespace) -> None:
    traces = load_records(Path(args.traces))
    dataset = build_router_continual_dataset(
        traces,
        RouterContinualConfig(
            min_rating=args.min_rating,
            include_failed_corrections=args.include_failed_corrections,
            max_per_label=args.max_per_label,
            seed=args.seed,
        ),
    )
    write_router_dataset(Path(args.out), dataset)
    print(json.dumps({"out": args.out, **dataset["metadata"]}, ensure_ascii=False, indent=2))


def cmd_merge_router_data(args: argparse.Namespace) -> None:
    base = _load_document(Path(args.base))
    continual = _load_document(Path(args.continual))
    dataset = merge_router_training_datasets(
        base=base,
        continual=continual,
        config=RouterMergeConfig(
            continual_ratio=args.continual_ratio,
            max_per_label=args.max_per_label,
            seed=args.seed,
        ),
    )
    write_router_dataset(Path(args.out), dataset)
    print(json.dumps({"out": args.out, **dataset["metadata"]}, ensure_ascii=False, indent=2))


def cmd_train_router_prototype(args: argparse.Namespace) -> None:
    records = load_router_records(Path(args.train))
    model = train_router_prototype(records, min_token_count=args.min_token_count)
    write_json(Path(args.out), model)
    print(json.dumps({"out": args.out, "examples": model["created_from"]["examples"]}, ensure_ascii=False, indent=2))


def cmd_evaluate_router_prototype(args: argparse.Namespace) -> None:
    model = _load_document(Path(args.model))
    records = load_router_records(Path(args.data))
    metrics = evaluate_router_prototype(model, records)
    write_json(Path(args.out), metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def cmd_predict_router_prototype(args: argparse.Namespace) -> None:
    model = _load_document(Path(args.model))
    prediction = predict_router_prototype(model, args.text)
    print(json.dumps(prediction, ensure_ascii=False, indent=2))


def cmd_train_router(args: argparse.Namespace) -> None:
    train_router_lora(args)
    print(json.dumps({"adapter": str(args.output), "base_model": args.base_model}, indent=2))


def cmd_train_ft(args: argparse.Namespace) -> None:
    train_orchestrator(args)
    print(json.dumps({"adapter": str(args.output), "base_model": args.base_model}, indent=2))


def cmd_evaluate_ft(args: argparse.Namespace) -> None:
    client = LocalAdapterPlanClient(
        Path(args.adapter),
        base_model=args.base_model,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    metrics, predictions = evaluate_adapter(client, read_jsonl(Path(args.data)))
    output = Path(args.predictions)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": metrics, "records": predictions}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def cmd_select_ft(args: argparse.Namespace) -> None:
    signal = _router_signal(args)
    client = LocalAdapterPlanClient(
        Path(args.adapter),
        base_model=args.base_model,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    decision = LearnedGraphSelector(client, fallback=_selector(args)).select(args.text, signal, args.risk)
    print(json.dumps({"router": _signal_dict(signal), "decision": decision.to_dict()}, ensure_ascii=False, indent=2))


def _selector(args: argparse.Namespace) -> GraphSelector:
    return GraphSelector(
        SelectionPolicy(
            minimum_confidence=args.minimum_confidence,
            single_specialist_margin=args.single_margin,
            secondary_score=args.secondary_score,
            max_selected_labels=args.max_selected_labels,
        )
    )


def _apply_bandit(args: argparse.Namespace, signal: RouterSignal, decision):
    if not args.bandit_state:
        return decision
    state = load_bandit_state(Path(args.bandit_state))
    rollout_config = _bandit_rollout_config(args, state)
    if not rollout_config["enabled"]:
        return decision
    selected = apply_bandit_policy(
        text=args.text,
        signal=signal,
        served_decision=decision,
        risk_level=args.risk,
        selector=_selector(args),
        state=state,
        config=BanditPolicyConfig(
            min_observations=args.bandit_min_observations,
            exploration_weight=args.bandit_exploration_weight,
            explore_unobserved=args.bandit_explore_unobserved,
            include_high_risk=args.bandit_include_high_risk,
            traffic_percent=rollout_config["traffic_percent"],
            rollout_salt=rollout_config["rollout_salt"],
            rollout_digest=rollout_config.get("rollout_digest"),
            release_digest=rollout_config.get("release_digest"),
            release_id=rollout_config.get("release_id"),
            release_current_digest=rollout_config.get("release_current_digest"),
        ),
    )
    if selected is decision:
        return decision
    return replace(
        selected,
        selection_metadata={
            **selected.selection_metadata,
            "pre_bandit_decision": decision.to_dict(),
        },
    )


def _bandit_rollout_config(args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any]:
    config = {
        "enabled": True,
        "traffic_percent": args.bandit_traffic_percent,
        "rollout_salt": args.bandit_rollout_salt,
        "rollout_digest": None,
        "release_digest": None,
        "release_id": None,
        "release_current_digest": None,
    }
    if args.bandit_release_current:
        current_path = Path(args.bandit_release_current)
        current = _load_document(current_path)
        manifest_path = _resolve_current_manifest_path(current, current_path)
        manifest = _load_document(manifest_path)
        current_validation = validate_bandit_current_release(
            current=current,
            manifest=manifest,
            require_not_expired=not args.bandit_allow_expired_release,
        )
        if current_validation["status"] != "pass":
            raise SystemExit(f"bandit current release pointer is not valid: {current_validation}")
        release_validation = validate_bandit_release_manifest(
            manifest=manifest,
            state=state,
            require_not_expired=not args.bandit_allow_expired_release,
        )
        if release_validation["status"] != "pass":
            raise SystemExit(f"bandit release manifest is not valid for the provided state: {release_validation}")
        runtime = manifest.get("runtime", {})
        if not isinstance(runtime, dict):
            raise ValueError("bandit release manifest must contain a runtime object")
        return {
            **config,
            "enabled": bool(runtime.get("enabled", False)),
            "traffic_percent": float(runtime.get("traffic_percent", 0.0)),
            "rollout_salt": str(runtime.get("rollout_salt", config["rollout_salt"])),
            "release_digest": artifact_digest(manifest),
            "release_id": manifest.get("release_id"),
            "release_current_digest": artifact_digest(current),
        }
    if args.bandit_release_manifest:
        manifest = _load_document(Path(args.bandit_release_manifest))
        release_validation = validate_bandit_release_manifest(
            manifest=manifest,
            state=state,
            require_not_expired=not args.bandit_allow_expired_release,
        )
        if release_validation["status"] != "pass":
            raise SystemExit(f"bandit release manifest is not valid for the provided state: {release_validation}")
        runtime = manifest.get("runtime", {})
        if not isinstance(runtime, dict):
            raise ValueError("bandit release manifest must contain a runtime object")
        return {
            **config,
            "enabled": bool(runtime.get("enabled", False)),
            "traffic_percent": float(runtime.get("traffic_percent", 0.0)),
            "rollout_salt": str(runtime.get("rollout_salt", config["rollout_salt"])),
            "release_digest": artifact_digest(manifest),
            "release_id": manifest.get("release_id"),
        }
    if not args.bandit_rollout_config:
        return config
    value = _load_document(Path(args.bandit_rollout_config))
    binding = validate_bandit_rollout_binding(
        state=state,
        rollout=value,
        require_state_binding=not args.bandit_allow_unbound_rollout,
        require_not_expired=not args.bandit_allow_expired_rollout,
    )
    if binding["status"] != "pass":
        raise SystemExit(f"bandit rollout config is not bound to the provided state: {binding}")
    runtime = value.get("runtime", value)
    if not isinstance(runtime, dict):
        raise ValueError("bandit rollout config must contain an object or a runtime object")
    if runtime.get("enabled") is False:
        return {**config, "enabled": False, "traffic_percent": 0.0, "rollout_digest": artifact_digest(value)}
    return {
        **config,
        "enabled": True,
        "traffic_percent": float(runtime.get("traffic_percent", config["traffic_percent"])),
        "rollout_salt": str(runtime.get("rollout_salt", config["rollout_salt"])),
        "rollout_digest": artifact_digest(value),
    }


def _resolve_current_manifest_path(current: dict[str, Any], current_path: Path) -> Path:
    manifest_path = Path(str(current.get("manifest_path", "")))
    if not manifest_path:
        raise ValueError("bandit current release pointer must contain manifest_path")
    return manifest_path if manifest_path.is_absolute() else current_path.parent / manifest_path


def _resolve_rollback_manifest_path(rollback: dict[str, Any], rollback_path: Path) -> Path:
    candidate = rollback.get("candidate", {})
    candidate = candidate if isinstance(candidate, dict) else {}
    manifest_path = Path(str(candidate.get("manifest_path", "")))
    if not manifest_path:
        raise ValueError("bandit rollback candidate must contain candidate.manifest_path or --manifest")
    return manifest_path if manifest_path.is_absolute() else rollback_path.parent / manifest_path


def _relative_or_absolute_manifest_path(manifest_path: Path, current_path: Path) -> str:
    try:
        return str(manifest_path.resolve().relative_to(current_path.resolve().parent))
    except ValueError:
        return str(manifest_path)


def _decision(
    args: argparse.Namespace,
    signal: RouterSignal,
    budget: Budget,
    allowed_models: set[str] | None = None,
    model_catalog: list[dict[str, Any]] | None = None,
):
    deterministic = _selector(args)
    orchestrator_url = getattr(args, "orchestrator_url", None)
    if not orchestrator_url:
        return deterministic.select(args.text, signal, args.risk)
    client = OpenAIPlanClient(
        orchestrator_url,
        args.orchestrator_model,
        api_key_env=args.orchestrator_api_key_env,
        timeout_seconds=args.orchestrator_timeout,
    )
    return LearnedGraphSelector(
        client,
        fallback=deterministic,
        allowed_models=allowed_models,
        model_catalog=model_catalog,
    ).select(args.text, signal, args.risk, budget)


def _router_signal(args: argparse.Namespace) -> RouterSignal:
    if args.scores:
        scores = json.loads(args.scores)
        if not isinstance(scores, dict):
            raise SystemExit("--scores must be a JSON object")
        return RouterSignal(scores={str(key): float(value) for key, value in scores.items()}, model="static-scores")
    if args.scores_file:
        value = _load_document(Path(args.scores_file))
        scores = value.get("scores", value)
        if not isinstance(scores, dict):
            raise SystemExit("scores file must contain an object or a scores object")
        return RouterSignal(scores={str(key): float(item) for key, item in scores.items()}, model="static-scores")
    return OpenAIRouterClient(args.router_url, args.router_model, args.router_timeout).classify(args.text)


def _signal_dict(signal: RouterSignal) -> dict[str, Any]:
    return {"model": signal.model, "latency_ms": signal.latency_ms, "scores": signal.scores}


def _load_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    value = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def _path_digest(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise ValueError(f"path does not exist: {path}")
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        data = item.read_bytes()
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _model_catalog(model_config: dict[str, Any]) -> list[dict[str, Any]]:
    return OpenAIModelClient(model_config).model_catalog()


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _add_router_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--scores", help="router scores as a JSON object")
    source.add_argument("--scores-file", help="JSON/YAML file containing router scores")
    parser.add_argument("--router-url", default="http://127.0.0.1:18001/v1")
    parser.add_argument("--router-model", default="router")
    parser.add_argument("--router-timeout", type=float, default=30.0)
    parser.add_argument("--risk", choices=("auto", "low", "normal", "high"), default="auto")
    parser.add_argument("--minimum-confidence", type=float, default=0.45)
    parser.add_argument("--single-margin", type=float, default=0.25)
    parser.add_argument("--secondary-score", type=float, default=0.15)
    parser.add_argument("--max-selected-labels", type=int, default=3)


def _add_learned_endpoint_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--orchestrator-url", help="OpenAI-compatible endpoint serving the FT orchestrator")
    parser.add_argument("--orchestrator-model", default="tune-orchestrator-ft")
    parser.add_argument("--orchestrator-api-key-env")
    parser.add_argument("--orchestrator-timeout", type=float, default=30.0)


def _add_local_adapter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=1024)


def _add_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graphs", default=str(DEFAULT_GRAPHS))
    parser.add_argument("--model-config")
    parser.add_argument("--mock", action="store_true", help="use deterministic local model responses")
    parser.add_argument("--trace", default=str(DEFAULT_TRACE))
    parser.add_argument("--max-cost", type=float, default=1.0)
    parser.add_argument("--max-latency-ms", type=int, default=60_000)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--shadow-mode",
        choices=("off", "deterministic-baseline", "alternatives"),
        default="off",
        help="run non-served comparison graphs and store them under trace.shadow_executions",
    )
    parser.add_argument("--shadow-max-count", type=int, default=1)
    parser.add_argument("--shadow-max-cost", type=float, default=0.0)
    parser.add_argument("--shadow-max-latency-ms", type=int, default=30_000)
    parser.add_argument(
        "--shadow-include-high-risk",
        action="store_true",
        help="allow shadow execution for high-risk requests; policy and verifier constraints still apply",
    )
    parser.add_argument("--bandit-state", help="JSON state produced by build-bandit-state")
    parser.add_argument("--bandit-min-observations", type=int, default=3)
    parser.add_argument("--bandit-exploration-weight", type=float, default=0.1)
    parser.add_argument("--bandit-explore-unobserved", action="store_true")
    parser.add_argument("--bandit-traffic-percent", type=float, default=100.0)
    parser.add_argument("--bandit-rollout-salt", default="default")
    parser.add_argument("--bandit-rollout-config", help="JSON/YAML rollout plan produced by plan-bandit-rollout")
    parser.add_argument("--bandit-release-manifest", help="JSON/YAML release manifest produced by build-bandit-release")
    parser.add_argument("--bandit-release-current", help="JSON/YAML current release pointer produced by activate-bandit-release")
    parser.add_argument("--bandit-allow-unbound-rollout", action="store_true")
    parser.add_argument("--bandit-allow-expired-rollout", action="store_true")
    parser.add_argument("--bandit-allow-expired-release", action="store_true")
    parser.add_argument(
        "--bandit-include-high-risk",
        action="store_true",
        help="allow bandit policy selection for high-risk requests; candidate safety gates still apply",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TuneRouter deterministic and learned graph orchestrator")
    subparsers = parser.add_subparsers(required=True)

    select = subparsers.add_parser("select", help="select an execution graph")
    select.add_argument("text")
    _add_router_args(select)
    _add_learned_endpoint_args(select)
    select.set_defaults(func=cmd_select)

    run = subparsers.add_parser("run", help="select and execute a graph")
    run.add_argument("text")
    _add_router_args(run)
    _add_learned_endpoint_args(run)
    _add_execution_args(run)
    run.set_defaults(func=cmd_run)

    evaluate = subparsers.add_parser("evaluate", help="offline replay over query x candidate results")
    evaluate.add_argument("--candidate-results", required=True)
    evaluate.add_argument("--predictions")
    evaluate.add_argument("--out", default=str(DEFAULT_RUNTIME_ARTIFACTS / "evaluation"))
    evaluate.set_defaults(func=cmd_evaluate)

    bandit = subparsers.add_parser("build-bandit-state", help="build bounded contextual bandit state from reviewed traces")
    bandit.add_argument("--traces", default=str(DEFAULT_TRACE))
    bandit.add_argument("--out", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-state.json"))
    bandit.add_argument("--cost-weight", type=float, default=0.25)
    bandit.add_argument("--latency-weight", type=float, default=0.001)
    bandit.set_defaults(func=cmd_build_bandit_state)

    replay_bandit = subparsers.add_parser("replay-bandit", help="offline replay a bandit state against reviewed served/shadow traces")
    replay_bandit.add_argument("--traces", default=str(DEFAULT_TRACE))
    replay_bandit.add_argument("--bandit-state", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-state.json"))
    replay_bandit.add_argument("--out", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-replay.json"))
    replay_bandit.add_argument("--report", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-replay.md"))
    replay_bandit.add_argument("--min-observations", type=int, default=3)
    replay_bandit.add_argument("--exploration-weight", type=float, default=0.1)
    replay_bandit.add_argument("--explore-unobserved", action="store_true")
    replay_bandit.add_argument("--include-high-risk", action="store_true")
    replay_bandit.add_argument("--no-leave-one-out", action="store_true")
    replay_bandit.set_defaults(func=cmd_replay_bandit)

    gate_bandit = subparsers.add_parser("gate-bandit", help="evaluate bandit replay metrics against promotion thresholds")
    gate_bandit.add_argument("--replay", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-replay.json"))
    gate_bandit.add_argument("--out", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-promotion.json"))
    gate_bandit.add_argument("--report", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-promotion.md"))
    gate_bandit.add_argument("--min-evaluated-requests", type=int, default=30)
    gate_bandit.add_argument("--min-mean-reward-delta", type=float, default=0.0)
    gate_bandit.add_argument("--max-loss-rate", type=float, default=0.05)
    gate_bandit.add_argument("--max-switch-rate", type=float, default=0.50)
    gate_bandit.add_argument("--max-skip-rate", type=float, default=0.80)
    gate_bandit.add_argument("--no-fail", action="store_true")
    gate_bandit.set_defaults(func=cmd_gate_bandit)

    monitor_bandit = subparsers.add_parser("monitor-bandit", help="evaluate live canary rollout health from orchestration traces")
    monitor_bandit.add_argument("--traces", default=str(DEFAULT_TRACE))
    monitor_bandit.add_argument("--out", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-monitor.json"))
    monitor_bandit.add_argument("--report", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-monitor.md"))
    monitor_bandit.add_argument("--min-bandit-traces", type=int, default=10)
    monitor_bandit.add_argument("--max-bandit-failure-rate", type=float, default=0.05)
    monitor_bandit.add_argument("--max-relative-failure-rate", type=float, default=0.10)
    monitor_bandit.add_argument("--max-bandit-p95-latency-ms", type=float, default=60_000.0)
    monitor_bandit.add_argument("--max-bandit-mean-cost-usd", type=float, default=1.0)
    monitor_bandit.add_argument("--no-fail", action="store_true")
    monitor_bandit.set_defaults(func=cmd_monitor_bandit)

    rollout_bandit = subparsers.add_parser("plan-bandit-rollout", help="plan the next bandit rollout traffic step from promotion and monitor gates")
    rollout_bandit.add_argument("--promotion", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-promotion.json"))
    rollout_bandit.add_argument("--monitor")
    rollout_bandit.add_argument("--bandit-state", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-state.json"))
    rollout_bandit.add_argument("--out", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-rollout.json"))
    rollout_bandit.add_argument("--report", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-rollout.md"))
    rollout_bandit.add_argument("--current-traffic-percent", type=float, default=0.0)
    rollout_bandit.add_argument("--step-percent", type=float, default=10.0)
    rollout_bandit.add_argument("--max-traffic-percent", type=float, default=100.0)
    rollout_bandit.add_argument("--rollback-traffic-percent", type=float, default=0.0)
    rollout_bandit.add_argument("--rollout-salt", default="default")
    rollout_bandit.add_argument("--min-monitor-bandit-traces", type=int, default=10)
    rollout_bandit.add_argument("--max-age-hours", type=float, default=24.0)
    rollout_bandit.add_argument("--no-fail", action="store_true")
    rollout_bandit.set_defaults(func=cmd_plan_bandit_rollout)

    verify_rollout = subparsers.add_parser("verify-bandit-rollout", help="verify rollout artifact digests before runtime deployment")
    verify_rollout.add_argument("--rollout", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-rollout.json"))
    verify_rollout.add_argument("--bandit-state", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-state.json"))
    verify_rollout.add_argument("--promotion", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-promotion.json"))
    verify_rollout.add_argument("--monitor")
    verify_rollout.add_argument("--out", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-rollout-verification.json"))
    verify_rollout.add_argument("--report", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-rollout-verification.md"))
    verify_rollout.add_argument("--require-monitor", action="store_true")
    verify_rollout.add_argument("--allow-unbound-state", action="store_true")
    verify_rollout.add_argument("--allow-missing-promotion", action="store_true")
    verify_rollout.add_argument("--allow-expired", action="store_true")
    verify_rollout.add_argument("--no-fail", action="store_true")
    verify_rollout.set_defaults(func=cmd_verify_bandit_rollout)

    release_bandit = subparsers.add_parser("build-bandit-release", help="bundle verified bandit rollout artifacts into a runtime release manifest")
    release_bandit.add_argument("--rollout", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-rollout.json"))
    release_bandit.add_argument("--bandit-state", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-state.json"))
    release_bandit.add_argument("--promotion", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-promotion.json"))
    release_bandit.add_argument("--monitor")
    release_bandit.add_argument("--verification")
    release_bandit.add_argument("--out", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-release.json"))
    release_bandit.add_argument("--report", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-release.md"))
    release_bandit.add_argument("--require-monitor", action="store_true")
    release_bandit.add_argument("--no-fail", action="store_true")
    release_bandit.set_defaults(func=cmd_build_bandit_release)

    activate_bandit = subparsers.add_parser("activate-bandit-release", help="validate and write the current bandit release pointer")
    activate_bandit.add_argument("--manifest", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-release.json"))
    activate_bandit.add_argument("--bandit-state", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-state.json"))
    activate_bandit.add_argument("--out", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-current.json"))
    activate_bandit.add_argument("--channel", default="production")
    activate_bandit.add_argument("--allow-expired", action="store_true")
    activate_bandit.add_argument("--no-fail", action="store_true")
    activate_bandit.set_defaults(func=cmd_activate_bandit_release)

    record_release = subparsers.add_parser("record-bandit-release", help="append an activated bandit release to the release registry")
    record_release.add_argument("--current", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-current.json"))
    record_release.add_argument("--manifest", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-release.json"))
    record_release.add_argument("--registry", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-release-registry.json"))
    record_release.add_argument("--allow-expired", action="store_true")
    record_release.add_argument("--no-fail", action="store_true")
    record_release.set_defaults(func=cmd_record_bandit_release)

    rollback_release = subparsers.add_parser("select-bandit-rollback", help="select the latest healthy prior bandit release for rollback")
    rollback_release.add_argument("--registry", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-release-registry.json"))
    rollback_release.add_argument("--current-release-id")
    rollback_release.add_argument("--channel", default="production")
    rollback_release.add_argument("--out", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-rollback.json"))
    rollback_release.add_argument("--report", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-rollback.md"))
    rollback_release.add_argument("--allow-expired", action="store_true")
    rollback_release.add_argument("--no-fail", action="store_true")
    rollback_release.set_defaults(func=cmd_select_bandit_rollback)

    apply_rollback = subparsers.add_parser("apply-bandit-rollback", help="validate a rollback candidate and write the current release pointer")
    apply_rollback.add_argument("--rollback", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-rollback.json"))
    apply_rollback.add_argument("--manifest", help="candidate manifest path; defaults to candidate.manifest_path from rollback JSON")
    apply_rollback.add_argument("--bandit-state", help="optional state file used to verify the candidate state digest")
    apply_rollback.add_argument("--out", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-current.json"))
    apply_rollback.add_argument("--registry", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-release-registry.json"))
    apply_rollback.add_argument("--channel")
    apply_rollback.add_argument("--allow-expired", action="store_true")
    apply_rollback.add_argument("--no-fail", action="store_true")
    apply_rollback.set_defaults(func=cmd_apply_bandit_rollback)

    verify_current = subparsers.add_parser("verify-bandit-current", help="verify the active bandit current pointer, manifest, state, and optional registry")
    verify_current.add_argument("--current", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-current.json"))
    verify_current.add_argument("--manifest", help="release manifest path; defaults to manifest_path from current pointer")
    verify_current.add_argument("--bandit-state", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-state.json"))
    verify_current.add_argument("--registry", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-release-registry.json"))
    verify_current.add_argument("--out", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-current-verification.json"))
    verify_current.add_argument("--require-registry", action="store_true")
    verify_current.add_argument("--allow-expired", action="store_true")
    verify_current.add_argument("--no-fail", action="store_true")
    verify_current.set_defaults(func=cmd_verify_bandit_current)

    build_bundle = subparsers.add_parser("build-bandit-runtime-bundle", help="freeze current bandit runtime artifacts into an auditable bundle manifest")
    build_bundle.add_argument("--current", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-current.json"))
    build_bundle.add_argument("--manifest", help="release manifest path; defaults to manifest_path from current pointer")
    build_bundle.add_argument("--bandit-state", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-state.json"))
    build_bundle.add_argument("--current-verification", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-current-verification.json"))
    build_bundle.add_argument("--registry", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-release-registry.json"))
    build_bundle.add_argument("--graphs", default=str(DEFAULT_GRAPHS))
    build_bundle.add_argument("--model-config")
    build_bundle.add_argument("--out", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-runtime-bundle.json"))
    build_bundle.add_argument("--no-fail", action="store_true")
    build_bundle.set_defaults(func=cmd_build_bandit_runtime_bundle)

    verify_bundle = subparsers.add_parser("verify-bandit-runtime-bundle", help="verify that runtime artifacts still match a bandit runtime bundle")
    verify_bundle.add_argument("--bundle", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-runtime-bundle.json"))
    verify_bundle.add_argument("--current", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-current.json"))
    verify_bundle.add_argument("--manifest", help="release manifest path; defaults to manifest_path from current pointer")
    verify_bundle.add_argument("--bandit-state", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-state.json"))
    verify_bundle.add_argument("--current-verification", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-current-verification.json"))
    verify_bundle.add_argument("--registry", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-release-registry.json"))
    verify_bundle.add_argument("--graphs", default=str(DEFAULT_GRAPHS))
    verify_bundle.add_argument("--model-config")
    verify_bundle.add_argument("--out", default=str(DEFAULT_RUNTIME_ARTIFACTS / "bandit-runtime-bundle-verification.json"))
    verify_bundle.add_argument("--allow-expired", action="store_true")
    verify_bundle.add_argument("--no-fail", action="store_true")
    verify_bundle.set_defaults(func=cmd_verify_bandit_runtime_bundle)

    validate = subparsers.add_parser("validate-graphs", help="validate graph definitions")
    validate.add_argument("--graphs", default=str(DEFAULT_GRAPHS))
    validate.set_defaults(func=cmd_validate_graphs)

    doctor = subparsers.add_parser("doctor", help="run production readiness checks")
    doctor.add_argument("--router-url", default="http://127.0.0.1:18001/v1")
    doctor.add_argument("--router-model", default="router")
    doctor.add_argument("--router-timeout", type=float, default=30.0)
    doctor.add_argument("--graphs", default=str(DEFAULT_GRAPHS))
    doctor.add_argument("--model-config")
    doctor.add_argument("--adapter", help="local PRD learned orchestrator LoRA adapter directory to validate")
    doctor.add_argument("--orchestrator-url", help="OpenAI-compatible endpoint serving the PRD learned orchestrator")
    doctor.add_argument("--orchestrator-model", default="tune-orchestrator-ft")
    doctor.add_argument("--orchestrator-api-key-env")
    doctor.add_argument("--orchestrator-timeout", type=float, default=30.0)
    doctor.add_argument(
        "--text",
        default="Kubernetes上のPostgreSQLが遅い。PVCはNFSです",
        help="probe request used for router and optional model checks",
    )
    doctor.add_argument(
        "--skip-model-endpoints",
        action="store_true",
        help="skip /v1/models checks for configured model endpoints",
    )
    doctor.add_argument(
        "--probe-model-chat",
        action="store_true",
        help="send a small chat completion to each configured production model alias",
    )
    doctor.add_argument(
        "--probe-orchestrator",
        action="store_true",
        help="send a small plan-generation request to the learned orchestrator endpoint",
    )
    doctor.add_argument("--no-fail", action="store_true", help="always exit 0 and report status in JSON")
    doctor.set_defaults(func=cmd_doctor)

    trace_report = subparsers.add_parser("trace-report", help="summarize orchestration traces")
    trace_report.add_argument("--traces", default=str(DEFAULT_TRACE))
    trace_report.add_argument("--out", default=str(DEFAULT_RUNTIME_ARTIFACTS / "trace-report.md"))
    trace_report.set_defaults(func=cmd_trace_report)

    prepare_ft = subparsers.add_parser("prepare-ft-data", help="build SFT and preference data from outcomes")
    prepare_ft.add_argument("--candidate-results", required=True)
    prepare_ft.add_argument("--traces")
    prepare_ft.add_argument("--out", default=str(DEFAULT_FT_DATA))
    prepare_ft.add_argument("--quality-tolerance", type=float, default=0.02)
    prepare_ft.add_argument("--cost-weight", type=float, default=0.25)
    prepare_ft.add_argument("--latency-weight", type=float, default=0.001)
    prepare_ft.add_argument("--dev-ratio", type=float, default=0.1)
    prepare_ft.add_argument("--seed", type=int, default=42)
    prepare_ft.set_defaults(func=cmd_prepare_ft_data)

    router_pretrain = subparsers.add_parser("prepare-router-pretrain", help="normalize labeled router records for router pretraining")
    router_pretrain.add_argument("--source", default=str(PROJECT_ROOT / "artifacts" / "train.json"))
    router_pretrain.add_argument("--out", default=str(DEFAULT_ARTIFACTS / "router-pretrain"))
    router_pretrain.add_argument("--dev-ratio", type=float, default=0.15)
    router_pretrain.add_argument("--test-ratio", type=float, default=0.15)
    router_pretrain.add_argument("--max-per-label", type=int)
    router_pretrain.add_argument("--min-text-chars", type=int, default=8)
    router_pretrain.add_argument("--seed", type=int, default=42)
    router_pretrain.set_defaults(func=cmd_prepare_router_pretrain)

    router_continual = subparsers.add_parser("prepare-router-continual", help="extract reviewed runtime traces as router continual-learning records")
    router_continual.add_argument("--traces", default=str(DEFAULT_TRACE))
    router_continual.add_argument("--out", default=str(DEFAULT_ARTIFACTS / "router-continual"))
    router_continual.add_argument("--min-rating", type=int, default=4)
    router_continual.add_argument("--include-failed-corrections", action=argparse.BooleanOptionalAction, default=True)
    router_continual.add_argument("--max-per-label", type=int)
    router_continual.add_argument("--seed", type=int, default=42)
    router_continual.set_defaults(func=cmd_prepare_router_continual)

    router_merge = subparsers.add_parser("merge-router-data", help="merge base router data with continual-learning records")
    router_merge.add_argument("--base", default=str(DEFAULT_ARTIFACTS / "router-pretrain" / "dataset.json"))
    router_merge.add_argument("--continual", default=str(DEFAULT_ARTIFACTS / "router-continual" / "dataset.json"))
    router_merge.add_argument("--out", default=str(DEFAULT_ARTIFACTS / "router-merged"))
    router_merge.add_argument("--continual-ratio", type=float, default=0.35)
    router_merge.add_argument("--max-per-label", type=int)
    router_merge.add_argument("--seed", type=int, default=42)
    router_merge.set_defaults(func=cmd_merge_router_data)

    router_proto_train = subparsers.add_parser("train-router-prototype", help="train a dependency-free lexical router prototype")
    router_proto_train.add_argument("--train", default=str(DEFAULT_ARTIFACTS / "router-merged" / "train.json"))
    router_proto_train.add_argument("--out", default=str(DEFAULT_ARTIFACTS / "router-prototype.json"))
    router_proto_train.add_argument("--min-token-count", type=int, default=1)
    router_proto_train.set_defaults(func=cmd_train_router_prototype)

    router_proto_eval = subparsers.add_parser("evaluate-router-prototype", help="evaluate a lexical router prototype")
    router_proto_eval.add_argument("--model", default=str(DEFAULT_ARTIFACTS / "router-prototype.json"))
    router_proto_eval.add_argument("--data", default=str(DEFAULT_ARTIFACTS / "router-merged" / "dev.json"))
    router_proto_eval.add_argument("--out", default=str(DEFAULT_ARTIFACTS / "router-prototype-evaluation.json"))
    router_proto_eval.set_defaults(func=cmd_evaluate_router_prototype)

    router_proto_predict = subparsers.add_parser("predict-router-prototype", help="predict router scores with a lexical prototype")
    router_proto_predict.add_argument("text")
    router_proto_predict.add_argument("--model", default=str(DEFAULT_ARTIFACTS / "router-prototype.json"))
    router_proto_predict.set_defaults(func=cmd_predict_router_prototype)

    train_router = subparsers.add_parser("train-router", help="LoRA pretrain or continue-train the model router as sequence classification")
    train_router.add_argument("--train", default=str(DEFAULT_ARTIFACTS / "router-merged" / "train.json"))
    train_router.add_argument("--dev", default=str(DEFAULT_ARTIFACTS / "router-merged" / "dev.json"))
    train_router.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS / "router-lora")
    train_router.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B")
    train_router.add_argument("--epochs", type=float, default=2.0)
    train_router.add_argument("--learning-rate", type=float, default=2e-5)
    train_router.add_argument("--weight-decay", type=float, default=0.01)
    train_router.add_argument("--batch-size", type=int, default=4)
    train_router.add_argument("--eval-batch-size", type=int, default=4)
    train_router.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train_router.add_argument("--logging-steps", type=int, default=10)
    train_router.add_argument("--max-length", type=int, default=256)
    train_router.add_argument("--lora-r", type=int, default=8)
    train_router.add_argument("--lora-alpha", type=int, default=16)
    train_router.add_argument("--lora-dropout", type=float, default=0.05)
    train_router.add_argument("--fp16", action="store_true")
    train_router.add_argument("--bf16", action="store_true")
    train_router.add_argument("--seed", type=int, default=42)
    train_router.set_defaults(func=cmd_train_router)

    train_ft = subparsers.add_parser("train-ft", help="LoRA SFT a causal LM to generate orchestration plans")
    train_ft.add_argument("--train", default=str(DEFAULT_FT_DATA / "train.jsonl"))
    train_ft.add_argument("--dev", default=str(DEFAULT_FT_DATA / "dev.jsonl"))
    train_ft.add_argument("--output", type=Path, default=DEFAULT_FT_ADAPTER)
    train_ft.add_argument("--base-model", default=DEFAULT_ORCHESTRATOR_MODEL)
    train_ft.add_argument("--epochs", type=float, default=2.0)
    train_ft.add_argument("--learning-rate", type=float, default=2e-4)
    train_ft.add_argument("--weight-decay", type=float, default=0.01)
    train_ft.add_argument("--batch-size", type=int, default=1)
    train_ft.add_argument("--eval-batch-size", type=int, default=1)
    train_ft.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train_ft.add_argument("--logging-steps", type=int, default=10)
    train_ft.add_argument("--max-length", type=int, default=2048)
    train_ft.add_argument("--lora-r", type=int, default=16)
    train_ft.add_argument("--lora-alpha", type=int, default=32)
    train_ft.add_argument("--lora-dropout", type=float, default=0.05)
    train_ft.add_argument("--fp16", action="store_true")
    train_ft.add_argument("--bf16", action="store_true")
    train_ft.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    train_ft.add_argument("--seed", type=int, default=42)
    train_ft.set_defaults(func=cmd_train_ft)

    evaluate_ft = subparsers.add_parser("evaluate-ft", help="evaluate structured plan generation")
    _add_local_adapter_args(evaluate_ft)
    evaluate_ft.add_argument("--data", default=str(DEFAULT_FT_DATA / "dev.jsonl"))
    evaluate_ft.add_argument("--predictions", default=str(DEFAULT_ARTIFACTS / "ft-evaluation.json"))
    evaluate_ft.set_defaults(func=cmd_evaluate_ft)

    select_ft = subparsers.add_parser("select-ft", help="select a graph with a local LoRA orchestrator")
    select_ft.add_argument("text")
    _add_router_args(select_ft)
    _add_local_adapter_args(select_ft)
    select_ft.set_defaults(func=cmd_select_ft)

    run_ft = subparsers.add_parser("run-ft", help="select with a local LoRA orchestrator and execute a graph")
    run_ft.add_argument("text")
    _add_router_args(run_ft)
    _add_local_adapter_args(run_ft)
    _add_execution_args(run_ft)
    run_ft.set_defaults(func=cmd_run_ft)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
