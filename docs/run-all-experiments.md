# Run All Experiments

このドキュメントは、TuneScope の全実験を実行し、結果を `experiments/results` と `reports` に保存するためのコマンド集です。

## Status

現在の CLI は次を実装しています。

- `validate`
- `list-experiments`
- `run-card`
- `prepare-dataset`
- `setup-datasets`
- `train-sft`
- `train-dpo`
- `evaluate`
- `report`

`evaluate` は `configs/evaluation/default.yaml` の task 定義を読み、JGLUE、ELYZA-tasks-100、XL-Sum、形式遵守、推論コストをタスク別に保存します。

## Experiment IDs

`experiments/manifests/initial_matrix.yaml` の全実験:

| ID | 実行 | 備考 |
| --- | --- | --- |
| B0 | yes | Base model baseline |
| Q1 | yes | QLoRA, 100 samples |
| Q2 | yes | QLoRA, 500 samples |
| Q3 | yes | QLoRA, 2,000 samples |
| Q4 | yes | QLoRA, all samples |
| R1 | yes | QLoRA rank 8 |
| R2 | no | Q3 と同一条件なので `reuses_result_from: Q3` |
| R3 | yes | QLoRA rank 64 |
| L1 | yes | LoRA rank 32 |
| D1 | yes | DPO |
| F1 | yes | Full SFT |

実際に学習を走らせる ID:

```text
B0 Q1 Q2 Q3 Q4 R1 R3 L1 D1 F1
```

結果を読むだけの ID:

```text
R2 -> Q3
```

## One-Time Setup

```powershell
uv sync --group dev
uv run tunescope validate
uv run tunescope list-experiments
```

PyPI の証明書エラーが出る場合:

```powershell
uv --system-certs sync --group dev
```

## Pin Revisions

再現性のため、通常実験の前に `configs/datasets/*.yaml` の `revision` を Hugging Face の commit hash に固定します。

```powershell
uv run tunescope pin-dataset-revisions
```

ベースモデルも固定します。

```powershell
uv run tunescope set-base-model --model <hf-model-id> --method base --method qlora_sft --method lora_sft --method full_sft --pin-revision
uv run tunescope set-base-model --model experiments\results\Q3\model --experiment-id D1
```

固定前に試験実行する場合だけ `--allow-floating-revision` を付けます。

```powershell
uv run tunescope setup-datasets --priority-only --allow-floating-revision
```

全実験用データを準備する場合:

```powershell
uv run tunescope setup-datasets --allow-floating-revision
```

PowerShell ラッパーを使う場合:

```powershell
.\scripts\setup_datasets.ps1 -AllowFloatingRevision
```

## Save Run Cards

各実験の実行前チェックリストを `reports/run-cards` に保存します。

```powershell
New-Item -ItemType Directory -Force reports\run-cards
uv run tunescope run-card B0 | Set-Content -Encoding UTF8 reports\run-cards\B0.md
uv run tunescope run-card Q1 | Set-Content -Encoding UTF8 reports\run-cards\Q1.md
uv run tunescope run-card Q2 | Set-Content -Encoding UTF8 reports\run-cards\Q2.md
uv run tunescope run-card Q3 | Set-Content -Encoding UTF8 reports\run-cards\Q3.md
uv run tunescope run-card Q4 | Set-Content -Encoding UTF8 reports\run-cards\Q4.md
uv run tunescope run-card R1 | Set-Content -Encoding UTF8 reports\run-cards\R1.md
uv run tunescope run-card R2 | Set-Content -Encoding UTF8 reports\run-cards\R2.md
uv run tunescope run-card R3 | Set-Content -Encoding UTF8 reports\run-cards\R3.md
uv run tunescope run-card L1 | Set-Content -Encoding UTF8 reports\run-cards\L1.md
uv run tunescope run-card D1 | Set-Content -Encoding UTF8 reports\run-cards\D1.md
uv run tunescope run-card F1 | Set-Content -Encoding UTF8 reports\run-cards\F1.md
```

## Result Layout

結果は実験 ID ごとに保存します。

```text
experiments/results/
├── B0/
├── Q1/
├── Q2/
├── Q3/
├── Q4/
├── R1/
├── R3/
├── L1/
├── D1/
└── F1/
```

各ディレクトリに保存する推奨ファイル:

```text
run.yaml
train_metrics.json
eval_metrics.json
generation_samples.jsonl
cost.json
notes.md
```

`R2` は `experiments/results/Q3` を参照し、独立した学習結果は作りません。

学習コマンドは model / adapter artifact を `experiments/manifests/artifacts.yaml` に登録します。

```powershell
uv run tunescope list-artifacts
```

## Standard Commands

まとめて実行する場合:

```powershell
.\scripts\run_all_experiments.ps1 -AllowFloatingRevision
```

コマンド配線だけ確認する場合:

```powershell
.\scripts\run_all_experiments.ps1 -DryRun
```

### Baseline

```powershell
uv run tunescope evaluate --experiment-id B0 --output-dir experiments\results\B0
```

評価データの revision が未固定の検証段階では、評価にも `--allow-floating-revision` を付けます。

### QLoRA Data Size

```powershell
uv run tunescope train-sft --experiment-id Q1 --output-dir experiments\results\Q1
uv run tunescope evaluate --experiment-id Q1 --output-dir experiments\results\Q1

uv run tunescope train-sft --experiment-id Q2 --output-dir experiments\results\Q2
uv run tunescope evaluate --experiment-id Q2 --output-dir experiments\results\Q2

uv run tunescope train-sft --experiment-id Q3 --output-dir experiments\results\Q3
uv run tunescope evaluate --experiment-id Q3 --output-dir experiments\results\Q3

uv run tunescope train-sft --experiment-id Q4 --output-dir experiments\results\Q4
uv run tunescope evaluate --experiment-id Q4 --output-dir experiments\results\Q4
```

### LoRA Rank

```powershell
uv run tunescope train-sft --experiment-id R1 --output-dir experiments\results\R1
uv run tunescope evaluate --experiment-id R1 --output-dir experiments\results\R1

uv run tunescope evaluate --experiment-id R2 --reuse-result-from Q3 --output-dir experiments\results\Q3

uv run tunescope train-sft --experiment-id R3 --output-dir experiments\results\R3
uv run tunescope evaluate --experiment-id R3 --output-dir experiments\results\R3
```

### LoRA vs QLoRA

```powershell
uv run tunescope train-sft --experiment-id L1 --output-dir experiments\results\L1
uv run tunescope evaluate --experiment-id L1 --output-dir experiments\results\L1
```

### DPO

```powershell
uv run tunescope train-dpo --experiment-id D1 --output-dir experiments\results\D1
uv run tunescope evaluate --experiment-id D1 --output-dir experiments\results\D1
```

### Full SFT

```powershell
uv run tunescope train-sft --experiment-id F1 --output-dir experiments\results\F1
uv run tunescope evaluate --experiment-id F1 --output-dir experiments\results\F1
```

## Generate Reports

ELYZA-tasks-100 の人手評価または LLM Judge スコアを取り込む場合:

```powershell
uv run tunescope score-elyza --experiment-id Q2 --output-dir experiments\results\Q2 --scores reports\elyza_scores_Q2.csv
```

`--scores` は CSV または JSONL で、`id`、`score`、任意の `comment` を含めます。`id` がない場合は prediction の行順で対応付けます。

比較レポートの出力先:

```powershell
uv run tunescope report --matrix experiments\manifests\initial_matrix.yaml --results-dir experiments\results --output reports\initial_matrix.md
```

レポートには、生メトリクス、baseline 差分、artifact サイズ、学習時間あたりの改善量が含まれます。

HTML dashboard と tracker 用 export を生成する場合:

```powershell
uv run tunescope dashboard --report-json reports\initial_matrix.json --output reports\dashboard.html
uv run tunescope export-metrics --report-json reports\initial_matrix.json --output reports\metrics.csv
```

ELYZA prediction をローカル heuristic judge で自動採点する場合:

```powershell
uv run tunescope judge-elyza --experiment-id Q2 --output-dir experiments\results\Q2
```

古い checkpoint を整理する場合:

```powershell
uv run tunescope prune-checkpoints --experiment-id Q2 --keep-last 2
```

推奨する最初のレポート:

```powershell
uv run tunescope report --experiment-id B0 --experiment-id Q2 --experiment-id Q3 --results-dir experiments\results --output reports\base_vs_qlora_500_2000.md
```

## Minimal First Run

GPU コストを抑えて最初に確認する場合:

```powershell
uv run tunescope validate
uv run tunescope setup-datasets --priority-only --allow-floating-revision
uv run tunescope evaluate --experiment-id B0 --output-dir experiments\results\B0
uv run tunescope train-sft --experiment-id Q2 --output-dir experiments\results\Q2
uv run tunescope evaluate --experiment-id Q2 --output-dir experiments\results\Q2
uv run tunescope train-sft --experiment-id Q3 --output-dir experiments\results\Q3
uv run tunescope evaluate --experiment-id Q3 --output-dir experiments\results\Q3
uv run tunescope report --experiment-id B0 --experiment-id Q2 --experiment-id Q3 --results-dir experiments\results --output reports\base_vs_qlora_500_2000.md
```
