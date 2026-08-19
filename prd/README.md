# TuneRouter Prd

TuneRouterの6カテゴリ分類スコアをrouting signalとして使うGraph Orchestratorです。
決定的selectorに加え、outcomeから構造化planを学習するQwen Instruct + LoRA FT、DAG実行、並列専門家、
verification/repair loop、予算停止、JSONL trace、Query x Candidate結果に対するoffline replayを実装しています。

注意: `prd` のFTはrouter分類器ではなく、graph/model/delegation/verifier方針を生成するlearned orchestratorです。
`sever/` + `poc/artifacts/qwen-router-lora` は開発用bootstrap routerであり、本番ではOpenAI互換のrouter serviceとして別管理してください。

## Setup

```powershell
uv sync --project .\prd
```

Public interfaceは `tune-orchestrator` CLIです。`prd/src` 直下の `tune_*` Python moduleは内部実装であり、互換性を保証するpublic Python APIではありません。

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

未採用graphを裏で比較実行するshadow explorationは`--shadow-mode`で有効化します。

```powershell
uv run --project .\prd tune-orchestrator run `
  "PostgreSQL on NFS is slow" `
  --router-url http://127.0.0.1:18001/v1 `
  --model-config .\prd\config\model-endpoints.yaml `
  --shadow-mode deterministic-baseline `
  --shadow-max-count 1
```

review済みtraceからbounded contextual bandit stateを作り、runtimeで安全な候補集合内の選択に使えます。

```powershell
uv run --project .\prd tune-orchestrator build-bandit-state `
  --traces .\prd\artifacts\runtime\traces.jsonl `
  --out .\prd\artifacts\runtime\bandit-state.json
```

runtime投入前にoffline replayで昇格可否を確認します。

```powershell
uv run --project .\prd tune-orchestrator replay-bandit `
  --traces .\prd\artifacts\runtime\traces.jsonl `
  --bandit-state .\prd\artifacts\runtime\bandit-state.json `
  --out .\prd\artifacts\runtime\bandit-replay.json `
  --report .\prd\artifacts\runtime\bandit-replay.md
```

```powershell
uv run --project .\prd tune-orchestrator gate-bandit `
  --replay .\prd\artifacts\runtime\bandit-replay.json `
  --min-evaluated-requests 30 `
  --max-loss-rate 0.05
```

```powershell
uv run --project .\prd tune-orchestrator run `
  "PostgreSQL on NFS is slow" `
  --router-url http://127.0.0.1:18001/v1 `
  --model-config .\prd\config\model-endpoints.yaml `
  --bandit-state .\prd\artifacts\runtime\bandit-state.json `
  --bandit-traffic-percent 10
```

canary開始後はlive traceでhealth gateを回します。

```powershell
uv run --project .\prd tune-orchestrator monitor-bandit `
  --traces .\prd\artifacts\runtime\traces.jsonl `
  --min-bandit-traces 30 `
  --max-bandit-failure-rate 0.05
```

次のtraffic比率はrollout controllerで生成できます。

```powershell
uv run --project .\prd tune-orchestrator plan-bandit-rollout `
  --promotion .\prd\artifacts\runtime\bandit-promotion.json `
  --monitor .\prd\artifacts\runtime\bandit-monitor.json `
  --bandit-state .\prd\artifacts\runtime\bandit-state.json `
  --current-traffic-percent 10 `
  --step-percent 10 `
  --out .\prd\artifacts\runtime\bandit-rollout.json
```

```powershell
uv run --project .\prd tune-orchestrator verify-bandit-rollout `
  --rollout .\prd\artifacts\runtime\bandit-rollout.json `
  --bandit-state .\prd\artifacts\runtime\bandit-state.json `
  --promotion .\prd\artifacts\runtime\bandit-promotion.json `
  --monitor .\prd\artifacts\runtime\bandit-monitor.json `
  --require-monitor
```

```powershell
uv run --project .\prd tune-orchestrator build-bandit-release `
  --rollout .\prd\artifacts\runtime\bandit-rollout.json `
  --bandit-state .\prd\artifacts\runtime\bandit-state.json `
  --promotion .\prd\artifacts\runtime\bandit-promotion.json `
  --monitor .\prd\artifacts\runtime\bandit-monitor.json `
  --require-monitor
```

```powershell
uv run --project .\prd tune-orchestrator activate-bandit-release `
  --manifest .\prd\artifacts\runtime\bandit-release.json `
  --bandit-state .\prd\artifacts\runtime\bandit-state.json `
  --out .\prd\artifacts\runtime\bandit-current.json
```

反映したreleaseはregistryへ記録します。これによりincident時に「直前の正常release」を機械的に選べます。

```powershell
uv run --project .\prd tune-orchestrator record-bandit-release `
  --current .\prd\artifacts\runtime\bandit-current.json `
  --manifest .\prd\artifacts\runtime\bandit-release.json `
  --registry .\prd\artifacts\runtime\bandit-release-registry.json
```

rollback候補の選定:

```powershell
uv run --project .\prd tune-orchestrator select-bandit-rollback `
  --registry .\prd\artifacts\runtime\bandit-release-registry.json `
  --current-release-id bandit-release-current-id `
  --out .\prd\artifacts\runtime\bandit-rollback.json
```

rollbackを適用する場合は、候補manifest digestと任意のstate digestを検証してからcurrent pointerを書き換えます。

```powershell
uv run --project .\prd tune-orchestrator apply-bandit-rollback `
  --rollback .\prd\artifacts\runtime\bandit-rollback.json `
  --manifest .\prd\artifacts\runtime\bandit-release-previous.json `
  --bandit-state .\prd\artifacts\runtime\bandit-state-previous.json `
  --out .\prd\artifacts\runtime\bandit-current.json
```

runtime起動前にはcurrent pointer、manifest、state、registryをまとめて検証します。

```powershell
uv run --project .\prd tune-orchestrator verify-bandit-current `
  --current .\prd\artifacts\runtime\bandit-current.json `
  --bandit-state .\prd\artifacts\runtime\bandit-state.json `
  --registry .\prd\artifacts\runtime\bandit-release-registry.json `
  --require-registry
```

検証済みの起動入力はruntime bundleとして固定できます。

```powershell
uv run --project .\prd tune-orchestrator build-bandit-runtime-bundle `
  --current .\prd\artifacts\runtime\bandit-current.json `
  --bandit-state .\prd\artifacts\runtime\bandit-state.json `
  --current-verification .\prd\artifacts\runtime\bandit-current-verification.json `
  --registry .\prd\artifacts\runtime\bandit-release-registry.json `
  --graphs .\prd\graphs `
  --model-config .\prd\config\model-endpoints.yaml
```

```powershell
uv run --project .\prd tune-orchestrator verify-bandit-runtime-bundle `
  --bundle .\prd\artifacts\runtime\bandit-runtime-bundle.json `
  --current .\prd\artifacts\runtime\bandit-current.json `
  --bandit-state .\prd\artifacts\runtime\bandit-state.json `
  --current-verification .\prd\artifacts\runtime\bandit-current-verification.json `
  --registry .\prd\artifacts\runtime\bandit-release-registry.json `
  --graphs .\prd\graphs `
  --model-config .\prd\config\model-endpoints.yaml
```

```powershell
uv run --project .\prd tune-orchestrator run `
  "PostgreSQL on NFS is slow" `
  --router-url http://127.0.0.1:18001/v1 `
  --model-config .\prd\config\model-endpoints.yaml `
  --bandit-state .\prd\artifacts\runtime\bandit-state.json `
  --bandit-release-current .\prd\artifacts\runtime\bandit-current.json
```

`bandit-rollout.json` は `bandit-state` のdigestにbindされます。runtimeで違うstateを指定するとfail closedします。
また、既定で24時間の有効期限が入り、期限切れconfigもruntimeで拒否されます。

実運用前の起動前検査は`doctor`で実行します。graph、router、model alias、credential、model endpointの
到達性をまとめて確認できます。

```powershell
uv run --project .\prd tune-orchestrator doctor `
  --router-url http://127.0.0.1:18001/v1 `
  --model-config .\prd\config\model-endpoints.yaml
```

実モデルへ短いchat completionを送ってOpenAI互換応答まで確認する場合:

```powershell
uv run --project .\prd tune-orchestrator doctor `
  --router-url http://127.0.0.1:18001/v1 `
  --model-config .\prd\config\model-endpoints.yaml `
  --probe-model-chat
```

PRD learned orchestrator FT adapterの配置も検査する場合:

```powershell
uv run --project .\prd tune-orchestrator doctor `
  --router-url http://127.0.0.1:18001/v1 `
  --model-config .\prd\config\model-endpoints.yaml `
  --adapter .\prd\artifacts\orchestrator-lora
```

FT orchestratorをOpenAI互換endpointとして配信している場合:

```powershell
uv run --project .\prd tune-orchestrator doctor `
  --router-url http://127.0.0.1:18001/v1 `
  --model-config .\prd\config\model-endpoints.yaml `
  --orchestrator-url http://127.0.0.1:18003/v1 `
  --orchestrator-model tune-orchestrator-ft `
  --probe-orchestrator
```

業務運用の起動、接続、PRD FT昇格、監視、障害切り分けは`doc/production-runbook.md`を参照してください。

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

このcommandは現行MVPの評価harnessです。全候補Outcome Matrix、utility profile別Oracle/regret、routing collapse、KNN/MLPを含む共通baseline suiteへの拡張要件は、`TODO.md` のPRD-016〜PRD-018を正本とします。実装順も、運用機構の追加より先に固定モデルに対するQuality/Cost/Latency上の増分価値を検証する順へ変更しています。

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

ローカルadapterで選択からgraph実行まで行う場合:

```powershell
uv run --project .\prd --extra training tune-orchestrator run-ft `
  "NFS上のPostgreSQLでI/O待ちが急増した" `
  --adapter .\prd\artifacts\orchestrator-lora `
  --router-url http://127.0.0.1:18001/v1 `
  --model-config .\prd\config\model-endpoints.yaml
```

OpenAI互換endpointとして配信したFTモデルは、通常の`select`/`run`に`--orchestrator-url`を付けて使用できます。
FTモデルは固定graphに加え、`plan_type: bounded_graph`の制約付きnode/edge planも生成できます。
不正plan、未許可model/graph、cycle、high-risk verifier bypass、timeout時はdeterministic selectorへfallbackします。

詳細設計は`doc/learned-orchestrator-ft-design.md`を参照してください。

## Model Router Learning

router分類器は、事前学習データと運用trace由来の継続学習データを同じschemaへ正規化してから学習します。

```powershell
uv run --project .\prd tune-orchestrator prepare-router-pretrain `
  --source .\prd\artifacts\train.json `
  --out .\prd\artifacts\router-pretrain
```

review済みtraceから継続学習データを作ります。`approved`、`preferred`、高ratingのtraceだけを採用します。

```powershell
uv run --project .\prd tune-orchestrator prepare-router-continual `
  --traces .\prd\artifacts\runtime\traces.jsonl `
  --out .\prd\artifacts\router-continual
```

base dataとcontinual dataを混ぜます。`--continual-ratio` で新規運用データの比率を制限し、急な忘却を避けます。

```powershell
uv run --project .\prd tune-orchestrator merge-router-data `
  --base .\prd\artifacts\router-pretrain\dataset.json `
  --continual .\prd\artifacts\router-continual\dataset.json `
  --out .\prd\artifacts\router-merged `
  --continual-ratio 0.35
```

軽量prototypeでデータ品質を即確認できます。

```powershell
uv run --project .\prd tune-orchestrator train-router-prototype `
  --train .\prd\artifacts\router-merged\train.json

uv run --project .\prd tune-orchestrator evaluate-router-prototype `
  --model .\prd\artifacts\router-prototype.json `
  --data .\prd\artifacts\router-merged\dev.json
```

実モデルをLoRA sequence classificationとして学習します。

```powershell
uv run --project .\prd --extra training tune-orchestrator train-router `
  --train .\prd\artifacts\router-merged\train.json `
  --dev .\prd\artifacts\router-merged\dev.json `
  --output .\prd\artifacts\router-lora `
  --base-model Qwen/Qwen2.5-0.5B
```

## Tests

```powershell
$env:PYTHONPATH = ".\prd\src"
python -m unittest discover -s .\prd\src\test -v
```
