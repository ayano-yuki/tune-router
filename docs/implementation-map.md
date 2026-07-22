# Implementation Map

## Current Scope

このリポジトリの初期実装は、実験をすぐに増やすための土台に集中しています。

- dataset manifests
- training presets
- evaluation presets
- experiment matrix
- lightweight CLI for validation and run cards
- SFT/DPO training commands
- task-based benchmark evaluation command
- ELYZA judge score import command
- dataset/model pinning commands
- checkpoint resume and pruning commands
- Markdown/JSON report command
- static HTML dashboard command
- metrics export command
- automated ELYZA judge command
- uv-managed Python environment

`evaluate` は JGLUE、ELYZA-tasks-100、XL-Sum、形式遵守、推論コストのタスク別成果物を保存します。ELYZA の judge score は `score-elyza` で後から取り込みます。

## Modules

`src/tunescope/config.py`
: YAML の読み込み、必須フィールド検証、実験 ID の重複チェックを担当します。

`src/tunescope/cli.py`
: `validate`、`list-experiments`、`run-card`、`prepare-dataset`、`setup-datasets` を提供します。

`src/tunescope/dataset_setup.py`
: Hugging Face datasets の取得、サンプリング、標準 JSONL 化、manifest 生成を担当します。

`src/tunescope/training.py`
: TRL の `SFTTrainer` / `DPOTrainer` を使い、LoRA、QLoRA、Full SFT、DPO の実行と `train_metrics.json` 保存を担当します。

`src/tunescope/evaluation.py`
: モデル生成、JGLUE/XL-Sum 採点、ELYZA prediction 生成、judge score 取り込み、`eval_metrics.json`、`cost.json` 保存を担当します。

`src/tunescope/reporting.py`
: `experiments/results` から Markdown と JSON の比較レポートを生成します。

`src/tunescope/dashboard.py`
: レポート JSON から静的 HTML dashboard を生成します。

`src/tunescope/export.py`
: レポート JSON から CSV / JSONL の metrics export を生成します。

`src/tunescope/judge.py`
: ELYZA prediction に対して heuristic judge または外部 command judge を実行します。

`src/tunescope/config_edit.py`
: Hugging Face dataset revision の固定と、実験 config の base model 一括設定を担当します。

`src/tunescope/checkpoints.py`
: `checkpoint-*` の検出、latest checkpoint 解決、古い checkpoint の pruning を担当します。

`pyproject.toml`
: `uv sync --group dev` で開発環境を再現できるように、実行依存と dev dependency group を管理します。

`configs/datasets/*.yaml`
: データセット名、revision、split、用途、標準化後の形式を記録します。revision は実験開始前に Hugging Face の commit hash で固定します。

`configs/train/*.yaml`
: QLoRA、LoRA、DPO、Full SFT の共通ハイパーパラメータを管理します。

`configs/experiments/*.yaml`
: 個別実験の method、dataset、sample_count、rank、train preset、evaluation preset を管理します。

`experiments/manifests/initial_matrix.yaml`
: 最初に検証する実験群の一覧です。重複条件がある場合は `reuses_result_from` で明示します。

`docs/run-all-experiments.md`
: 全実験の実行順、結果保存先、run card 保存、最終レポート生成コマンドをまとめます。

## Optional Extensions

現時点の CLI は、データ準備、学習、評価、ELYZA 採点、自動 judge、artifact registry、checkpoint 管理、レポート生成、HTML dashboard、metrics export まで一通り実装済みです。今後の任意拡張は、クラウド実験基盤や特定組織向けの tracker API への直接 push です。
