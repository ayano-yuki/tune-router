from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tunescope.checkpoints import prune_checkpoints
from tunescope.config import ConfigError, get_experiment, load_all, load_matrix, repo_root, validate_workspace
from tunescope.config_edit import pin_dataset_revisions, set_base_model
from tunescope.dashboard import generate_dashboard
from tunescope.dataset_setup import experiment_dataset_requests, prepare_dataset
from tunescope.evaluation import apply_elyza_judge_scores, evaluate
from tunescope.export import export_metrics
from tunescope.judge import judge_elyza
from tunescope.reporting import generate_report
from tunescope.registry import load_artifact_registry
from tunescope.training import train_dpo, train_sft


def _root(value: str | None) -> Path:
    return Path(value).resolve() if value else repo_root()


def _rank(experiment: dict[str, Any]) -> str:
    overrides = experiment.get("overrides")
    if isinstance(overrides, dict):
        lora = overrides.get("lora")
        if isinstance(lora, dict) and lora.get("r") is not None:
            return str(lora["r"])
    return "-"


def _print_table(rows: list[list[str]]) -> None:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    for row_index, row in enumerate(rows):
        print("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
        if row_index == 0:
            print("  ".join("-" * width for width in widths))


def cmd_validate(args: argparse.Namespace) -> int:
    root = _root(args.root)
    messages = validate_workspace(root)
    exit_code = 0
    for item in messages:
        print(f"{item.level.upper()}: {item.message}")
        if item.level == "error":
            exit_code = 1
    return exit_code


def cmd_list_experiments(args: argparse.Namespace) -> int:
    root = _root(args.root)
    configs = load_all(root)
    matrix = load_matrix(root)
    priority = set(matrix.get("priority", []))
    rows = [["ID", "Priority", "Phase", "Method", "Dataset", "Count", "Rank", "Reuse"]]
    for experiment_id in matrix.get("experiments", []):
        experiment = configs["experiments"][experiment_id]
        rows.append(
            [
                experiment_id,
                "yes" if experiment_id in priority else "",
                str(experiment.get("phase", "")),
                str(experiment.get("method", "")),
                str(experiment.get("dataset") or "none"),
                str(experiment.get("sample_count", "")),
                _rank(experiment),
                str(experiment.get("reuses_result_from") or ""),
            ]
        )
    _print_table(rows)
    return 0


def render_run_card(experiment_id: str, root: Path | None = None) -> str:
    root = root or repo_root()
    experiment = get_experiment(experiment_id, root)
    lines = [
        f"# Run Card: {experiment['id']} - {experiment['name']}",
        "",
        f"- phase: {experiment.get('phase')}",
        f"- method: {experiment.get('method')}",
        f"- base_model: {experiment.get('base_model')}",
        f"- dataset: {experiment.get('dataset') or 'none'}",
        f"- sample_count: {experiment.get('sample_count')}",
        f"- seed: {experiment.get('seed', '-')}",
        f"- train_config: {experiment.get('train_config') or 'none'}",
        f"- evaluation_config: {experiment.get('evaluation_config')}",
        f"- lora_rank: {_rank(experiment)}",
    ]
    if experiment.get("reuses_result_from"):
        lines.append(f"- reuses_result_from: {experiment['reuses_result_from']}")
    if experiment.get("starts_from_experiment"):
        lines.append(f"- starts_from_experiment: {experiment['starts_from_experiment']}")

    lines.extend(
        [
            "",
            "## Checklist",
            "",
            "- [ ] Dataset revision is pinned.",
            "- [ ] Base model is pinned.",
            "- [ ] Training command and git commit are recorded.",
            "- [ ] Max VRAM and wall-clock time are recorded.",
            "- [ ] Evaluation outputs are saved under the result directory.",
            "- [ ] Notes include qualitative behavior changes.",
        ]
    )
    return "\n".join(lines)


def cmd_run_card(args: argparse.Namespace) -> int:
    root = _root(args.root)
    print(render_run_card(args.experiment_id, root))
    return 0


def _parse_sample_count(value: str) -> int | str:
    if value == "all":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sample count must be a positive integer or 'all'.") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("sample count must be a positive integer or 'all'.")
    return parsed


def cmd_prepare_dataset(args: argparse.Namespace) -> int:
    root = _root(args.root)
    result = prepare_dataset(
        root=root,
        dataset_id=args.dataset_id,
        sample_count=args.sample_count,
        seed=args.seed,
        allow_floating_revision=args.allow_floating_revision,
        force=args.force,
    )
    action = "skipped" if result.skipped else "prepared"
    print(f"{action}: {result.dataset_id}")
    print(f"records: {result.record_count}")
    print(f"jsonl: {result.output_path}")
    print(f"manifest: {result.manifest_path}")
    return 0


def cmd_setup_datasets(args: argparse.Namespace) -> int:
    root = _root(args.root)
    matrix = load_matrix(root)
    if args.experiment_id:
        experiment_ids = args.experiment_id
    elif args.priority_only:
        experiment_ids = list(matrix.get("priority", []))
    else:
        experiment_ids = list(matrix.get("experiments", []))

    requests = experiment_dataset_requests(root, experiment_ids)
    if not requests:
        print("No datasets required for the selected experiments.")
        return 0

    for dataset_id, sample_count, seed in requests:
        result = prepare_dataset(
            root=root,
            dataset_id=dataset_id,
            sample_count=sample_count,
            seed=seed,
            allow_floating_revision=args.allow_floating_revision,
            force=args.force,
        )
        action = "skipped" if result.skipped else "prepared"
        print(f"{action}: {dataset_id} ({sample_count}, seed={seed}) -> {result.output_path}")
    return 0


def cmd_train_sft(args: argparse.Namespace) -> int:
    root = _root(args.root)
    output_dir = train_sft(
        root=root,
        experiment_id=args.experiment_id,
        output_dir_arg=args.output_dir,
        dry_run=args.dry_run,
        max_steps=args.max_steps,
        allow_placeholder_model=args.allow_placeholder_model,
        resume_from_checkpoint=args.resume_from_checkpoint,
        auto_resume=args.auto_resume,
    )
    print(f"train-sft complete: {args.experiment_id}")
    print(f"output_dir: {output_dir}")
    return 0


def cmd_train_dpo(args: argparse.Namespace) -> int:
    root = _root(args.root)
    output_dir = train_dpo(
        root=root,
        experiment_id=args.experiment_id,
        output_dir_arg=args.output_dir,
        dry_run=args.dry_run,
        max_steps=args.max_steps,
        allow_placeholder_model=args.allow_placeholder_model,
        resume_from_checkpoint=args.resume_from_checkpoint,
        auto_resume=args.auto_resume,
    )
    print(f"train-dpo complete: {args.experiment_id}")
    print(f"output_dir: {output_dir}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    root = _root(args.root)
    output_dir = evaluate(
        root=root,
        experiment_id=args.experiment_id,
        output_dir_arg=args.output_dir,
        dry_run=args.dry_run,
        max_new_tokens=args.max_new_tokens,
        allow_placeholder_model=args.allow_placeholder_model,
        reuse_result_from=args.reuse_result_from,
        task_ids=args.task_id,
        limit=args.limit,
        allow_floating_revision=args.allow_floating_revision,
    )
    print(f"evaluate complete: {args.experiment_id}")
    print(f"output_dir: {output_dir}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    root = _root(args.root)
    output = generate_report(
        root=root,
        output=args.output,
        matrix_path=args.matrix,
        experiment_ids=args.experiment_id,
        results_dir=args.results_dir,
    )
    print(f"report written: {output}")
    print(f"json written: {output.with_suffix('.json')}")
    return 0


def cmd_score_elyza(args: argparse.Namespace) -> int:
    root = _root(args.root)
    output_dir = apply_elyza_judge_scores(
        root=root,
        experiment_id=args.experiment_id,
        scores_path_arg=args.scores,
        output_dir_arg=args.output_dir,
    )
    print(f"elyza scores applied: {args.experiment_id}")
    print(f"output_dir: {output_dir}")
    return 0


def cmd_pin_dataset_revisions(args: argparse.Namespace) -> int:
    root = _root(args.root)
    changes = pin_dataset_revisions(root, dataset_ids=args.dataset_id, force=args.force)
    for change in changes:
        print(f"{change['status']}: {change['id']} -> {change['revision']}")
    return 0


def cmd_set_base_model(args: argparse.Namespace) -> int:
    root = _root(args.root)
    changes = set_base_model(
        root,
        model=args.model,
        experiment_ids=args.experiment_id,
        methods=args.method,
        validate_model=args.validate_model,
        pin_revision=args.pin_revision,
    )
    for change in changes:
        print(f"{change['status']}: {change['id']} -> {change['base_model']}")
    return 0


def cmd_prune_checkpoints(args: argparse.Namespace) -> int:
    root = _root(args.root)
    result = prune_checkpoints(
        root=root,
        experiment_id=args.experiment_id,
        output_dir_arg=args.output_dir,
        keep_last=args.keep_last,
    )
    print(f"pruned: {result['experiment_id']}")
    print(f"kept: {len(result['kept'])}")
    print(f"removed: {len(result['removed'])}")
    for path in result["removed"]:
        print(f"removed: {path}")
    return 0


def cmd_list_artifacts(args: argparse.Namespace) -> int:
    root = _root(args.root)
    registry = load_artifact_registry(root)
    artifacts = registry.get("artifacts", [])
    if not artifacts:
        print("No artifacts registered.")
        return 0
    rows = [["ID", "Method", "Kind", "Status", "Exists", "Size MB", "Path"]]
    for item in artifacts:
        size_bytes = item.get("size_bytes") or 0
        size_mb = float(size_bytes) / (1024**2)
        rows.append(
            [
                str(item.get("experiment_id")),
                str(item.get("method")),
                str(item.get("kind")),
                str(item.get("status")),
                str(item.get("exists")),
                f"{size_mb:.3g}",
                str(item.get("path")),
            ]
        )
    _print_table(rows)
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    root = _root(args.root)
    output = generate_dashboard(root, report_json=args.report_json, output=args.output)
    print(f"dashboard written: {output}")
    return 0


def cmd_export_metrics(args: argparse.Namespace) -> int:
    root = _root(args.root)
    output = export_metrics(root, report_json=args.report_json, output=args.output, fmt=args.format)
    print(f"metrics exported: {output}")
    return 0


def cmd_judge_elyza(args: argparse.Namespace) -> int:
    root = _root(args.root)
    output = judge_elyza(
        root=root,
        experiment_id=args.experiment_id,
        output_dir_arg=args.output_dir,
        scores_output=args.scores_output,
        provider=args.provider,
        judge_command=args.judge_command,
    )
    print(f"elyza judge scores written: {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tunescope")
    parser.add_argument("--root", help="Repository root. Defaults to the current TuneScope checkout.")
    subparsers = parser.add_subparsers(required=True)

    validate = subparsers.add_parser("validate", help="Validate config references and repository layout.")
    validate.set_defaults(func=cmd_validate)

    list_experiments = subparsers.add_parser("list-experiments", help="Print the initial experiment matrix.")
    list_experiments.set_defaults(func=cmd_list_experiments)

    run_card = subparsers.add_parser("run-card", help="Render a Markdown run card for an experiment.")
    run_card.add_argument("experiment_id")
    run_card.set_defaults(func=cmd_run_card)

    prepare_dataset_parser = subparsers.add_parser(
        "prepare-dataset",
        help="Download, sample, normalize, and manifest one configured dataset.",
    )
    prepare_dataset_parser.add_argument("dataset_id")
    prepare_dataset_parser.add_argument("--sample-count", type=_parse_sample_count, default="all")
    prepare_dataset_parser.add_argument("--seed", type=int, default=42)
    prepare_dataset_parser.add_argument("--allow-floating-revision", action="store_true")
    prepare_dataset_parser.add_argument("--force", action="store_true")
    prepare_dataset_parser.set_defaults(func=cmd_prepare_dataset)

    setup_datasets = subparsers.add_parser(
        "setup-datasets",
        help="Prepare datasets required by selected experiments.",
    )
    setup_datasets.add_argument("--experiment-id", action="append", help="Experiment ID to prepare. Repeatable.")
    setup_datasets.add_argument("--priority-only", action="store_true", help="Prepare only matrix priority experiments.")
    setup_datasets.add_argument("--allow-floating-revision", action="store_true")
    setup_datasets.add_argument("--force", action="store_true")
    setup_datasets.set_defaults(func=cmd_setup_datasets)

    train_sft_parser = subparsers.add_parser("train-sft", help="Run LoRA, QLoRA, or Full SFT for an experiment.")
    train_sft_parser.add_argument("--experiment-id", required=True)
    train_sft_parser.add_argument("--output-dir")
    train_sft_parser.add_argument("--dry-run", action="store_true")
    train_sft_parser.add_argument("--max-steps", type=int)
    train_sft_parser.add_argument("--allow-placeholder-model", action="store_true")
    train_sft_parser.add_argument("--resume-from-checkpoint")
    train_sft_parser.add_argument("--auto-resume", action="store_true")
    train_sft_parser.set_defaults(func=cmd_train_sft)

    train_dpo_parser = subparsers.add_parser("train-dpo", help="Run DPO for an experiment.")
    train_dpo_parser.add_argument("--experiment-id", required=True)
    train_dpo_parser.add_argument("--output-dir")
    train_dpo_parser.add_argument("--dry-run", action="store_true")
    train_dpo_parser.add_argument("--max-steps", type=int)
    train_dpo_parser.add_argument("--allow-placeholder-model", action="store_true")
    train_dpo_parser.add_argument("--resume-from-checkpoint")
    train_dpo_parser.add_argument("--auto-resume", action="store_true")
    train_dpo_parser.set_defaults(func=cmd_train_dpo)

    evaluate_parser = subparsers.add_parser("evaluate", help="Generate sample outputs and evaluation metrics.")
    evaluate_parser.add_argument("--experiment-id", required=True)
    evaluate_parser.add_argument("--output-dir")
    evaluate_parser.add_argument("--dry-run", action="store_true")
    evaluate_parser.add_argument("--max-new-tokens", type=int)
    evaluate_parser.add_argument("--allow-placeholder-model", action="store_true")
    evaluate_parser.add_argument("--reuse-result-from")
    evaluate_parser.add_argument("--task-id", action="append", help="Evaluation task ID to run. Repeatable.")
    evaluate_parser.add_argument("--limit", type=int, help="Maximum records per evaluation task.")
    evaluate_parser.add_argument("--allow-floating-revision", action="store_true")
    evaluate_parser.set_defaults(func=cmd_evaluate)

    report_parser = subparsers.add_parser("report", help="Aggregate experiment metrics into Markdown and JSON.")
    report_parser.add_argument("--matrix")
    report_parser.add_argument("--experiment-id", action="append")
    report_parser.add_argument("--results-dir")
    report_parser.add_argument("--output", required=True)
    report_parser.set_defaults(func=cmd_report)

    score_elyza = subparsers.add_parser("score-elyza", help="Apply human or LLM judge scores to ELYZA predictions.")
    score_elyza.add_argument("--experiment-id", required=True)
    score_elyza.add_argument("--scores", required=True, help="CSV or JSONL with id, score, and optional comment.")
    score_elyza.add_argument("--output-dir")
    score_elyza.set_defaults(func=cmd_score_elyza)

    pin_revisions = subparsers.add_parser(
        "pin-dataset-revisions",
        help="Resolve current Hugging Face dataset commits and write them to dataset configs.",
    )
    pin_revisions.add_argument("--dataset-id", action="append", help="Dataset config ID to pin. Repeatable.")
    pin_revisions.add_argument("--force", action="store_true", help="Overwrite already pinned revisions.")
    pin_revisions.set_defaults(func=cmd_pin_dataset_revisions)

    base_model = subparsers.add_parser("set-base-model", help="Set base_model across experiment configs.")
    base_model.add_argument("--model", required=True)
    base_model.add_argument("--experiment-id", action="append", help="Experiment ID to update. Repeatable.")
    base_model.add_argument("--method", action="append", help="Only update experiments with this method. Repeatable.")
    base_model.add_argument("--validate-model", action="store_true", help="Check that the model exists on Hugging Face.")
    base_model.add_argument("--pin-revision", action="store_true", help="Also write base_model_revision.")
    base_model.set_defaults(func=cmd_set_base_model)

    prune = subparsers.add_parser("prune-checkpoints", help="Delete old checkpoint-* directories for an experiment.")
    prune.add_argument("--experiment-id", required=True)
    prune.add_argument("--output-dir")
    prune.add_argument("--keep-last", type=int, default=1)
    prune.set_defaults(func=cmd_prune_checkpoints)

    list_artifacts = subparsers.add_parser("list-artifacts", help="Print registered model and adapter artifacts.")
    list_artifacts.set_defaults(func=cmd_list_artifacts)

    dashboard = subparsers.add_parser("dashboard", help="Generate a static HTML dashboard from report JSON.")
    dashboard.add_argument("--report-json", required=True)
    dashboard.add_argument("--output", required=True)
    dashboard.set_defaults(func=cmd_dashboard)

    export = subparsers.add_parser("export-metrics", help="Export report metrics as CSV or JSONL.")
    export.add_argument("--report-json", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--format", choices=["csv", "jsonl"], default="csv")
    export.set_defaults(func=cmd_export_metrics)

    judge = subparsers.add_parser("judge-elyza", help="Automatically judge ELYZA predictions.")
    judge.add_argument("--experiment-id", required=True)
    judge.add_argument("--output-dir")
    judge.add_argument("--provider", choices=["heuristic", "command"], default="heuristic")
    judge.add_argument("--judge-command", help="Command template. Use {input}, {prompt}, {response}, or {id}.")
    judge.add_argument("--scores-output")
    judge.set_defaults(func=cmd_judge_elyza)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        parser.exit(2, f"ERROR: {exc}\n")
