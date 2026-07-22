# TuneScope

TuneScope は、同一の日本語 Instruction データと評価セットを使い、Base、LoRA、QLoRA、Full SFT、DPO の性能・計算コスト・回答傾向・一般能力の変化を再現可能に比較するための実験プロジェクトです。

最初の実施範囲は次の 3 条件に絞ります。

| ID | 手法 | データ | 件数 | rank |
| --- | --- | --- | ---: | ---: |
| B0 | Base | なし | 0 | - |
| Q2 | QLoRA | llm-jp/llm-jp-instructions | 500 | 32 |
| Q3 | QLoRA | llm-jp/llm-jp-instructions | 2,000 | 32 |

この 3 条件で評価差が見えない場合は、DPO や Full SFT に進む前に、ベースモデル、プロンプト形式、学習率、評価方法を見直します。

## Repository Layout

```text
tune-scope/
├── configs/
│   ├── datasets/      # Hugging Face dataset manifests
│   ├── evaluation/    # evaluation task settings
│   ├── experiments/   # experiment definitions
│   └── train/         # reusable training presets
├── datasets/
│   └── manifests/     # generated or pinned data manifests only
├── docs/
│   ├── experiment-plan.md
│   ├── implementation-map.md
│   └── run-all-experiments.md
├── experiments/
│   ├── manifests/     # experiment matrices
│   └── results/       # metrics and run artifacts
├── reports/
├── src/tunescope/
└── tests/
```

データ本体、モデル重み、アダプター、評価出力の大きな成果物は Git に入れません。必要な情報は dataset name、revision、split、sample_count、seed、抽出条件として manifest に残します。

## Quick Start

```powershell
uv sync --group dev
uv run tunescope validate
uv run tunescope list-experiments
uv run tunescope run-card Q2
uv run tunescope setup-datasets --priority-only --allow-floating-revision
```

PyPI への接続で証明書エラーが出る環境では、`uv --system-certs sync --group dev` を使います。

## Core Decisions

- SFT の第一候補は `llm-jp/llm-jp-instructions`。
- DPO の第一候補は `llm-jp/hh-rlhf-12k-ja`。
- 評価は JGLUE、ELYZA-tasks-100、XL-Sum 日本語 subset、形式遵守率、推論速度、GPU メモリを分けて記録する。
- 最初は QLoRA のデータ量比較を優先し、LoRA rank、LoRA vs QLoRA、DPO、Full SFT は評価基盤が固まってから追加する。

## Commands

`uv run tunescope validate`
: 設定ファイルの整合性と初期実験 ID の重複を確認します。

`uv run tunescope list-experiments`
: `experiments/manifests/initial_matrix.yaml` の実験一覧を表示します。

`uv run tunescope run-card <ID>`
: 実験 ID から run card の Markdown 下書きを表示します。

`uv run tunescope prepare-dataset llm_jp_instructions --sample-count 500 --allow-floating-revision`
: 単一データセットを Hugging Face から取得し、標準 JSONL と manifest を生成します。

`uv run tunescope setup-datasets --priority-only --allow-floating-revision`
: 初期優先実験 `B0`、`Q2`、`Q3` に必要なデータセットだけを準備します。`B0` はデータなしなのでスキップされます。

`uv run tunescope pin-dataset-revisions`
: Hugging Face の現在の dataset commit を取得し、`configs/datasets/*.yaml` の `revision` に書き込みます。

`uv run tunescope set-base-model --model <hf-model-id> --method qlora_sft --method lora_sft --method full_sft --pin-revision`
: 対象実験の `base_model` を一括設定し、必要なら model revision も固定します。

`uv run tunescope train-sft --experiment-id Q2 --output-dir experiments\results\Q2`
: SFT 系実験を実行し、モデルと `train_metrics.json` を保存します。

`uv run tunescope train-dpo --experiment-id D1 --output-dir experiments\results\D1`
: DPO 実験を実行し、モデルと `train_metrics.json` を保存します。

`uv run tunescope evaluate --experiment-id Q2 --output-dir experiments\results\Q2`
: JGLUE / ELYZA / XL-Sum / 形式遵守 / 推論コストのタスク別 prediction と metrics、`eval_metrics.json`、`cost.json` を保存します。

`uv run tunescope score-elyza --experiment-id Q2 --output-dir experiments\results\Q2 --scores reports\elyza_scores_Q2.csv`
: 人手評価または LLM Judge の ELYZA スコアを prediction と `eval_metrics.json` に反映します。

`uv run tunescope report --matrix experiments\manifests\initial_matrix.yaml --results-dir experiments\results --output reports\initial_matrix.md`
: 実験結果を Markdown と JSON のレポートに集約します。

`uv run tunescope prune-checkpoints --experiment-id Q2 --keep-last 2`
: 古い `checkpoint-*` ディレクトリを削除し、直近 checkpoint だけを残します。

`uv run tunescope list-artifacts`
: 登録済みの model / adapter artifact を一覧表示します。

`uv run tunescope judge-elyza --experiment-id Q2 --output-dir experiments\results\Q2`
: ELYZA prediction に自動 judge score を付与します。既定はローカル heuristic judge です。

`uv run tunescope dashboard --report-json reports\initial_matrix.json --output reports\dashboard.html`
: レポート JSON から静的 HTML ダッシュボードを生成します。

`uv run tunescope export-metrics --report-json reports\initial_matrix.json --output reports\metrics.csv`
: tracker 連携や表計算用に metrics を CSV / JSONL へ書き出します。

`.\scripts\setup_datasets.ps1 -PriorityOnly -AllowFloatingRevision`
: PowerShell から同じセットアップを実行します。

`.\scripts\run_all_experiments.ps1 -DryRun`
: 全実験コマンドの dry run を実行し、run card とレポートを保存します。

`uv run pytest`
: 設定検証テストを実行します。

[docs/run-all-experiments.md](docs/run-all-experiments.md) に、全実験の実行順、結果保存先、レポート生成コマンドをまとめています。

## Dataset Setup

生成物は `datasets/prepared/<dataset_id>/*.jsonl` と `datasets/manifests/*.yaml` に出力されます。データ本体は `.gitignore` で除外し、manifest だけを残します。

再現性のため、通常は `configs/datasets/*.yaml` の `revision` を Hugging Face の commit hash に固定してから実行します。まだ revision を固定していない検証段階では、明示的に `--allow-floating-revision` を付けます。
