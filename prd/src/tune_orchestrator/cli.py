from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from .clients import MockModelClient, OpenAIModelClient, OpenAIRouterClient
from .evaluation import (
    evaluate_offline,
    load_records,
    summarize_traces,
    write_evaluation_outputs,
    write_trace_report,
)
from .executor import GraphExecutor
from .ft_data import FTDataConfig, build_ft_datasets, write_ft_datasets
from .graphs import load_graphs
from .learned import LearnedGraphSelector, OpenAIPlanClient
from .models import Budget, RouterSignal
from .selector import GraphSelector, SelectionPolicy
from .training import (
    DEFAULT_ORCHESTRATOR_MODEL,
    LocalAdapterPlanClient,
    evaluate_adapter,
    read_jsonl,
    train_orchestrator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    budget = Budget(
        max_cost_usd=args.max_cost,
        max_latency_ms=args.max_latency_ms,
        max_steps=args.max_steps,
    )
    decision = _decision(args, signal, budget, allowed_models)
    graphs = load_graphs(Path(args.graphs))
    if decision.graph_id not in graphs:
        raise SystemExit(f"graph definition is missing: {decision.graph_id}")

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
        graph=graphs[decision.graph_id],
        budget=budget,
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


def cmd_evaluate(args: argparse.Namespace) -> None:
    candidates = load_records(Path(args.candidate_results))
    predictions = load_records(Path(args.predictions)) if args.predictions else None
    summaries, details, pareto = evaluate_offline(candidates, predictions)
    out_dir = Path(args.out)
    write_evaluation_outputs(out_dir, summaries, details, pareto)
    print(json.dumps({"out": str(out_dir), "routers": summaries}, ensure_ascii=False, indent=2))


def cmd_validate_graphs(args: argparse.Namespace) -> None:
    graphs = load_graphs(Path(args.graphs))
    print(json.dumps({"valid": True, "graphs": sorted(graphs)}, indent=2))


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


def _decision(
    args: argparse.Namespace,
    signal: RouterSignal,
    budget: Budget,
    allowed_models: set[str] | None = None,
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
    run.add_argument("--graphs", default=str(DEFAULT_GRAPHS))
    run.add_argument("--model-config")
    run.add_argument("--mock", action="store_true", help="use deterministic local model responses")
    run.add_argument("--trace", default=str(DEFAULT_TRACE))
    run.add_argument("--max-cost", type=float, default=1.0)
    run.add_argument("--max-latency-ms", type=int, default=60_000)
    run.add_argument("--max-steps", type=int, default=12)
    run.add_argument("--max-workers", type=int, default=4)
    run.add_argument("--debug", action="store_true")
    run.set_defaults(func=cmd_run)

    evaluate = subparsers.add_parser("evaluate", help="offline replay over query x candidate results")
    evaluate.add_argument("--candidate-results", required=True)
    evaluate.add_argument("--predictions")
    evaluate.add_argument("--out", default=str(DEFAULT_RUNTIME_ARTIFACTS / "evaluation"))
    evaluate.set_defaults(func=cmd_evaluate)

    validate = subparsers.add_parser("validate-graphs", help="validate graph definitions")
    validate.add_argument("--graphs", default=str(DEFAULT_GRAPHS))
    validate.set_defaults(func=cmd_validate_graphs)

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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
