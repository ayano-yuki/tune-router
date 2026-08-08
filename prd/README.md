# TuneRouter Prd

TuneRouterの6カテゴリ分類スコアをrouting signalとして使うGraph Orchestratorです。
決定的selectorに加え、outcomeから構造化planを学習するQwen Instruct + LoRA FT、DAG実行、並列専門家、
verification/repair loop、予算停止、JSONL trace、Query x Candidate結果に対するoffline replayを実装しています。

## Setup

```powershell
uv sync --project .\prd
```

## Dataset

日本語データセットは`prd/artifacts/`を正本として使用します。

| File | Records | Purpose |
| --- | ---: | --- |
| `artifacts/train.json` | 3,323 | 既存6カテゴリ分類器の学習、orchestrator bootstrap |
| `artifacts/dev.json` | 716 | 調整、plan生成評価 |
| `artifacts/test.json` | 761 | 固定評価 |
| `artifacts/dataset.json` | 4,800 | 全レコード |

このデータは`Storage`、`Network`、`Coding`、`Security`、`Database`、`General`のdomain supervisionを提供します。Fugu型orchestratorのoutcome FTでは、これに全候補model/graphのquality、cost、latency結果を紐付けて教師planを生成します。

`prd/artifacts/`内は次のように分離します。データセット4ファイルはGit管理し、生成物サブディレクトリは`prd/.gitignore`で除外します。

```text
artifacts/
├── dataset.json
├── train.json
├── dev.json
├── test.json
├── runtime/             # trace・評価レポート
├── ft-data/             # SFT・preferenceデータ
└── orchestrator-lora/   # LoRA adapter
```

## Graph selection

既存の分類サーバを起動した状態で実行します。

```powershell
uv run --project .\prd tune-orchestrator select `
  "Kubernetes上のPostgreSQLが遅い。PVCはNFSです" `
  --router-url http://127.0.0.1:18001/v1
```

分類器なしで閾値を確認する場合は固定スコアを渡せます。

```powershell
uv run --project .\prd tune-orchestrator select `
  "PostgreSQL on NFS is slow" `
  --scores '{"Database":0.46,"Storage":0.36,"Network":0.10,"General":0.01,"Coding":0.04,"Security":0.03}'
```

## Execute a graph

外部モデルを呼ばず、実行経路とtraceを確認します。

```powershell
uv run --project .\prd tune-orchestrator run `
  "PostgreSQL on NFS is slow" --mock `
  --scores '{"Database":0.46,"Storage":0.36,"Network":0.10,"General":0.01,"Coding":0.04,"Security":0.03}'
```

実モデルはOpenAI互換エンドポイントへ接続します。`config/model-endpoints.example.yaml`を環境に合わせ、
`--model-config`で指定してください。traceは既定で`prd/artifacts/runtime/traces.jsonl`へ追記されます。

## Offline evaluation

入力はPRDの`candidate_results` schemaを持つJSONまたはJSONLです。任意のrouter predictionも同時に比較できます。

```powershell
uv run --project .\prd tune-orchestrator evaluate `
  --candidate-results .\prd\eval\candidate-results.example.json `
  --predictions .\prd\eval\router-predictions.example.jsonl `
  --out .\prd\artifacts\runtime\evaluation
```

Random、Always Small、Always Large、Best Single、Rule-based、Graph Selector、追加predictionを同じ結果表で比較し、
Quality、Routing Accuracy、Regret、Cost、Latency、success rate、multi-agent過不足、Pareto CSVを出力します。

実行traceの完全性、verifier pass、repair成功率、ループ回数、cost/latencyは次で集計できます。

```powershell
uv run --project .\prd tune-orchestrator trace-report `
  --traces .\prd\artifacts\runtime\traces.jsonl
```

## Learned Orchestrator FT

全候補outcomeと任意のレビュー済みtraceから、SFTとpreferenceデータを生成します。

```powershell
uv run --project .\prd tune-orchestrator prepare-ft-data `
  --candidate-results .\prd\eval\candidate-results.example.json `
  --traces .\prd\artifacts\runtime\traces.jsonl `
  --out .\prd\artifacts\ft-data
```

学習依存を導入し、Qwen2.5-1.5B-InstructをLoRA SFTします。

```powershell
uv sync --project .\prd --extra training --system-certs

uv run --project .\prd --extra training tune-orchestrator train-ft `
  --train .\prd\artifacts\ft-data\train.jsonl `
  --dev .\prd\artifacts\ft-data\dev.jsonl `
  --output .\prd\artifacts\orchestrator-lora `
  --bf16
```

構造化plan生成を評価します。

```powershell
uv run --project .\prd --extra training tune-orchestrator evaluate-ft `
  --adapter .\prd\artifacts\orchestrator-lora `
  --data .\prd\artifacts\ft-data\dev.jsonl
```

ローカルadapterで単発選択する場合:

```powershell
uv run --project .\prd --extra training tune-orchestrator select-ft `
  "NFS上のPostgreSQLでI/O待ちが急増した" `
  --adapter .\prd\artifacts\orchestrator-lora `
  --scores '{"Database":0.46,"Storage":0.36,"Network":0.10,"General":0.01,"Coding":0.04,"Security":0.03}'
```

OpenAI互換endpointとして配信したFTモデルは、通常の`select`/`run`に`--orchestrator-url`を付けて使用できます。
不正plan、未許可model/graph、timeout時はdeterministic selectorへfallbackします。

詳細設計は`doc/learned-orchestrator-ft-design.md`を参照してください。

## Tests

```powershell
$env:PYTHONPATH = ".\prd\src"
python -m unittest discover -s .\prd\tests -v
```
