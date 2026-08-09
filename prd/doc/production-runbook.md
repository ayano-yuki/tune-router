# TuneRouter Prd Production Runbook

このRunbookは、`prd/` のGraph OrchestratorをPoC相当の検証状態から、業務運用で扱える構成へ昇格させるための手順です。

重要な切り分け:

- Router classifier: 入力文を `Storage`、`Network`、`Coding`、`Security`、`Database`、`General` のscoreへ変換する外部依存です。
- PRD learned orchestrator FT: router score、risk、budget、model catalogを見て、graph、delegation、verifier、synthesis方針をJSON planとして生成する `prd` 側のFTです。
- Specialist models: 実際に回答、検証、統合を行うOpenAI互換LLM群です。

`sever/` と `poc/artifacts/qwen-router-lora` は開発用bootstrap routerです。本番構成では、PoC成果物を本番依存にせず、同じOpenAI互換contractを満たすrouter serviceとして管理してください。

## Production Topology

```text
User Request
  |
  v
Router classifier endpoint
  - POST /v1/chat/completions
  - response.router.scores
  |
  v
PRD Graph Orchestrator
  - deterministic selector, or
  - PRD learned orchestrator FT
  |
  v
Specialist model endpoints
  - storage-specialist
  - network-specialist
  - coding-specialist
  - security-specialist
  - database-specialist
  - general-fallback
  - verifier
  - general-synthesizer
  |
  v
answer + prd/artifacts/runtime/traces.jsonl
```

## Setup

```bash
cd /path/to/tune-scope

uv sync --project ./prd
uv sync --project ./prd --extra training --system-certs
```

`sever/` は開発用routerを起動する場合だけ同期します。

```bash
uv sync --project ./sever --system-certs
```

テストは必ず `uv run` 経由で実行します。

```bash
uv run --project ./prd python -m unittest discover -s ./prd/src/test -v
```

## Router Contract

PRDはrouter実装そのものには依存しません。必要なのはOpenAI互換のchat completion responseに `router.scores` が含まれることです。

期待レスポンス例:

```json
{
  "model": "router",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Storage"
      }
    }
  ],
  "router": {
    "confidence": 0.84,
    "scores": {
      "Storage": 0.84,
      "Network": 0.01,
      "Coding": 0.03,
      "Security": 0.02,
      "Database": 0.10,
      "General": 0.00
    }
  }
}
```

開発用bootstrap routerを使う場合:

```bash
uv run --project ./sever --system-certs python ./sever/src/openai_classifier_server.py \
  --host 127.0.0.1 \
  --port 18001 \
  --model-name router \
  --adapter ./poc/artifacts/qwen-router-lora
```

本番ではこのadapter pathをRunbookやsystemd unitに固定しないでください。router serviceは独立した成果物としてversion、評価結果、rollback手順を持たせます。

Router疎通:

```bash
curl http://127.0.0.1:18001/v1/models
```

```bash
curl -s http://127.0.0.1:18001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"router","messages":[{"role":"user","content":"Kubernetes上のPostgreSQLが遅い。PVCはNFSです"}]}'
```

## Router Learning Lifecycle

routerは6カテゴリ分類器です。PRDでは、PoC artifactを直接本番依存にせず、事前学習データと運用trace由来の継続学習データを明示的に作成します。

1. 事前学習データを正規化する

```bash
uv run --project ./prd tune-orchestrator prepare-router-pretrain \
  --source ./prd/artifacts/train.json \
  --out ./prd/artifacts/router-pretrain
```

2. review済みtraceから継続学習データを作る

```bash
uv run --project ./prd tune-orchestrator prepare-router-continual \
  --traces ./prd/artifacts/runtime/traces.jsonl \
  --out ./prd/artifacts/router-continual \
  --min-rating 4
```

採用条件:

- `review_label` が `approved`、`preferred`、`success`
- または `user_rating >= --min-rating`
- router labelは `graph.selected_labels`、`router.primary_labels`、`graph.delegations[].label` の順に抽出する
- low rating、未review、未知labelは学習に入れない

3. base dataとcontinual dataをmergeする

```bash
uv run --project ./prd tune-orchestrator merge-router-data \
  --base ./prd/artifacts/router-pretrain/dataset.json \
  --continual ./prd/artifacts/router-continual/dataset.json \
  --out ./prd/artifacts/router-merged \
  --continual-ratio 0.35
```

`--continual-ratio` は運用データの混入率上限です。過大にすると古い一般化能力を忘れやすくなるため、昇格前は0.2から0.35程度で比較してください。

4. 依存なしprototypeでデータ品質を確認する

```bash
uv run --project ./prd tune-orchestrator train-router-prototype \
  --train ./prd/artifacts/router-merged/train.json \
  --out ./prd/artifacts/router-prototype.json

uv run --project ./prd tune-orchestrator evaluate-router-prototype \
  --model ./prd/artifacts/router-prototype.json \
  --data ./prd/artifacts/router-merged/dev.json \
  --out ./prd/artifacts/router-prototype-evaluation.json
```

5. 実routerをLoRA sequence classificationとして学習する

```bash
uv run --project ./prd --extra training tune-orchestrator train-router \
  --train ./prd/artifacts/router-merged/train.json \
  --dev ./prd/artifacts/router-merged/dev.json \
  --output ./prd/artifacts/router-lora \
  --base-model Qwen/Qwen2.5-0.5B \
  --epochs 2
```

router adapterの昇格条件:

- held-out dev/testで既存routerのmacro F1を下回らない
- `Storage` vs `Database`、`Network` vs `Security` など境界事例のregressionが許容範囲
- `General` への逃げ過ぎ、または専門labelへの過剰確信が増えていない
- `select` / `run` のtraceで `router.scores` 分布、margin、graph選択が期待範囲
- rollback可能なrouter artifact versionとして保存されている

## PRD Learned Orchestrator FT Lifecycle

PRDのFTはrouterではなく、graph orchestration planを学習します。

1. Outcome dataを作る

```bash
uv run --project ./prd tune-orchestrator prepare-ft-data \
  --candidate-results ./prd/eval/candidate-results.example.json \
  --traces ./prd/artifacts/runtime/traces.jsonl \
  --out ./prd/artifacts/ft-data
```

2. LoRA SFTする

```bash
uv run --project ./prd --extra training tune-orchestrator train-ft \
  --train ./prd/artifacts/ft-data/train.jsonl \
  --dev ./prd/artifacts/ft-data/dev.jsonl \
  --output ./prd/artifacts/orchestrator-lora \
  --bf16
```

3. FT adapterを評価する

```bash
uv run --project ./prd --extra training tune-orchestrator evaluate-ft \
  --adapter ./prd/artifacts/orchestrator-lora \
  --data ./prd/artifacts/ft-data/dev.jsonl \
  --predictions ./prd/artifacts/ft-evaluation.json
```

review済みtraceが同一queryに複数ある場合、`prepare-ft-data` はSFT用データに加えて
`trajectory_preferences.jsonl` も生成します。これは、実行trajectory単位のchosen/rejected比較です。

```bash
ls ./prd/artifacts/ft-data
```

期待される生成物:

- `train.jsonl`
- `dev.jsonl`
- `preferences.jsonl`
- `trajectory_preferences.jsonl`
- `metadata.json`

`trajectory_preferences.jsonl` のpromptにはrating、review label、final answer、node outputを入れません。
review情報やnode実行状況はmetadataに残し、preference optimizationや監査で利用します。

昇格条件の最低ライン:

- `valid_plan_rate` が業務閾値以上
- `graph_accuracy` と `selected_labels_exact_match` がdeterministic baselineを下回らない
- high-risk requestでverifierを迂回しない
- offline replayでmean regret、cost、latencyが悪化しない
- shadow traceでfallback率と `node_failed` が許容範囲

## Running With PRD FT

local adapterを直接使い、graph実行まで行う場合:

```bash
uv run --project ./prd --extra training tune-orchestrator run-ft \
  "Kubernetes上のPostgreSQLが遅い。PVCはNFSです" \
  --adapter ./prd/artifacts/orchestrator-lora \
  --router-url http://127.0.0.1:18001/v1 \
  --model-config ./prd/config/model-endpoints.yaml
```

外部modelをまだ起動していない段階では `--mock` でcontrol planeだけ確認します。

```bash
uv run --project ./prd --extra training tune-orchestrator run-ft \
  "Kubernetes上のPostgreSQLが遅い。PVCはNFSです" \
  --adapter ./prd/artifacts/orchestrator-lora \
  --router-url http://127.0.0.1:18001/v1 \
  --mock \
  --debug
```

FT orchestratorをOpenAI互換endpointとして配信している場合は、通常の `run` に `--orchestrator-url` を渡します。

```bash
uv run --project ./prd tune-orchestrator run \
  "Kubernetes上のPostgreSQLが遅い。PVCはNFSです" \
  --router-url http://127.0.0.1:18001/v1 \
  --orchestrator-url http://127.0.0.1:18003/v1 \
  --orchestrator-model tune-orchestrator-ft \
  --model-config ./prd/config/model-endpoints.yaml
```

不正JSON、未許可graph、未許可model、high-risk verifier bypass、timeout時はdeterministic selectorへfallbackします。`trace.graph.selector_type` と `trace.graph.fallback_reason` を確認してください。

## Specialist Endpoint Integration

`prd/config/model-endpoints.example.yaml` をコピーして、本番環境用に編集します。

```bash
cp ./prd/config/model-endpoints.example.yaml ./prd/config/model-endpoints.yaml
```

単一のOpenAI互換LLMを全aliasで共有する最小例:

```yaml
defaults:
  base_url: http://127.0.0.1:18002/v1
  timeout_seconds: 30
  input_cost_per_million: 0
  output_cost_per_million: 0

models:
  storage-specialist:
    model: Qwen/Qwen2.5-7B-Instruct
  network-specialist:
    model: Qwen/Qwen2.5-7B-Instruct
  coding-specialist:
    model: Qwen/Qwen2.5-Coder-7B-Instruct
  security-specialist:
    model: Qwen/Qwen2.5-7B-Instruct
  database-specialist:
    model: Qwen/Qwen2.5-7B-Instruct
  general-fallback:
    model: Qwen/Qwen2.5-7B-Instruct
  verifier:
    model: Qwen/Qwen2.5-7B-Instruct
  general-synthesizer:
    model: Qwen/Qwen2.5-7B-Instruct
```

API keyが必要なgatewayを使う場合:

```yaml
defaults:
  base_url: https://llm-gateway.example.internal/v1
  api_key_env: LLM_GATEWAY_API_KEY
  timeout_seconds: 45
```

```bash
export LLM_GATEWAY_API_KEY='...'
```

## Preflight

deterministic selectorとspecialist endpointを確認:

```bash
uv run --project ./prd tune-orchestrator doctor \
  --router-url http://127.0.0.1:18001/v1 \
  --model-config ./prd/config/model-endpoints.yaml
```

local PRD FT adapterの配置も確認:

```bash
uv run --project ./prd tune-orchestrator doctor \
  --router-url http://127.0.0.1:18001/v1 \
  --model-config ./prd/config/model-endpoints.yaml \
  --adapter ./prd/artifacts/orchestrator-lora
```

FT orchestrator endpointのplan生成まで確認:

```bash
uv run --project ./prd tune-orchestrator doctor \
  --router-url http://127.0.0.1:18001/v1 \
  --model-config ./prd/config/model-endpoints.yaml \
  --orchestrator-url http://127.0.0.1:18003/v1 \
  --orchestrator-model tune-orchestrator-ft \
  --probe-orchestrator
```

各specialist aliasへ短いchat completionを送る:

```bash
uv run --project ./prd tune-orchestrator doctor \
  --router-url http://127.0.0.1:18001/v1 \
  --model-config ./prd/config/model-endpoints.yaml \
  --probe-model-chat
```

`doctor` が確認するもの:

- graph定義
- router contract
- `model-endpoints.yaml` のalias網羅性
- credential環境変数
- specialist endpointの `/v1/models`
- 任意でspecialist chat completion
- 任意でPRD FT local adapter配置
- 任意でPRD FT endpointのJSON plan生成

## Production Execution Modes

Mode A: deterministic selector

```bash
uv run --project ./prd tune-orchestrator run \
  "Kubernetes上のPostgreSQLが遅い。PVCはNFSです" \
  --router-url http://127.0.0.1:18001/v1 \
  --model-config ./prd/config/model-endpoints.yaml
```

Mode B: local PRD FT adapter

```bash
uv run --project ./prd --extra training tune-orchestrator run-ft \
  "Kubernetes上のPostgreSQLが遅い。PVCはNFSです" \
  --adapter ./prd/artifacts/orchestrator-lora \
  --router-url http://127.0.0.1:18001/v1 \
  --model-config ./prd/config/model-endpoints.yaml
```

Mode C: PRD FT endpoint

```bash
uv run --project ./prd tune-orchestrator run \
  "Kubernetes上のPostgreSQLが遅い。PVCはNFSです" \
  --router-url http://127.0.0.1:18001/v1 \
  --orchestrator-url http://127.0.0.1:18003/v1 \
  --orchestrator-model tune-orchestrator-ft \
  --model-config ./prd/config/model-endpoints.yaml
```

PRD FTは固定graph planに加えて、制約付きのbounded graph planも返せます。採用された場合は
`trace.graph.id` が `bounded_graph` になり、実行されたnode/edge planは `trace.graph.generated_graph` に保存されます。

確認:

```bash
uv run --project ./prd tune-orchestrator run \
  "Kubernetes上のPostgreSQLが遅い。PVCはNFSです" \
  --router-url http://127.0.0.1:18001/v1 \
  --orchestrator-url http://127.0.0.1:18003/v1 \
  --orchestrator-model tune-orchestrator-ft \
  --model-config ./prd/config/model-endpoints.yaml \
  --debug
```

見る場所:

- `trace.graph.selector_type`
- `trace.graph.id`
- `trace.graph.generated_graph`
- `trace.graph.fallback_reason`

## Shadow Exploration

未採用graphを裏で実行し、ユーザーにはserved answerだけを返す場合は `--shadow-mode` を使います。

deterministic baselineをshadow実行:

```bash
uv run --project ./prd tune-orchestrator run \
  "Kubernetes上のPostgreSQLが遅い。PVCはNFSです" \
  --router-url http://127.0.0.1:18001/v1 \
  --orchestrator-url http://127.0.0.1:18003/v1 \
  --orchestrator-model tune-orchestrator-ft \
  --model-config ./prd/config/model-endpoints.yaml \
  --shadow-mode deterministic-baseline \
  --shadow-max-count 1 \
  --shadow-max-cost 0.05 \
  --shadow-max-latency-ms 30000
```

single/parallelの代替候補も含める:

```bash
uv run --project ./prd tune-orchestrator run \
  "Kubernetes上のPostgreSQLが遅い。PVCはNFSです" \
  --router-url http://127.0.0.1:18001/v1 \
  --model-config ./prd/config/model-endpoints.yaml \
  --shadow-mode alternatives \
  --shadow-max-count 2 \
  --shadow-max-cost 0.05
```

既定ではhigh-risk requestではshadowを実行しません。明示的に許可する場合:

```bash
--shadow-include-high-risk
```

保存先:

- `trace.shadow_executions[].reason`
- `trace.shadow_executions[].status`
- `trace.shadow_executions[].decision`
- `trace.shadow_executions[].trace`

shadow実行の失敗はserved answerを壊しません。失敗理由は `trace.shadow_executions[].error` に保存されます。

## Bounded Contextual Bandit

review済みtraceから、context別のgraph/model選択統計を作れます。

```bash
uv run --project ./prd tune-orchestrator build-bandit-state \
  --traces ./prd/artifacts/runtime/traces.jsonl \
  --out ./prd/artifacts/runtime/bandit-state.json
```

`build-bandit-state` はreview済みのserved traceに加えて、review済みの
`trace.shadow_executions[].trace` も観測として取り込みます。shadow outcomeを学習へ戻す場合は、
shadow trace側にも `evaluation.user_rating` または `evaluation.review_label` を付けてからstateを作ります。

runtime投入前にoffline replayを実行します。

```bash
uv run --project ./prd tune-orchestrator replay-bandit \
  --traces ./prd/artifacts/runtime/traces.jsonl \
  --bandit-state ./prd/artifacts/runtime/bandit-state.json \
  --out ./prd/artifacts/runtime/bandit-replay.json \
  --report ./prd/artifacts/runtime/bandit-replay.md \
  --min-observations 3
```

見る指標:

- `summary.mean_reward_delta`
- `summary.loss_rate`
- `summary.switch_rate`
- `summary.skipped`
- `details[].selected_source`

最低昇格条件:

- `mean_reward_delta >= 0`
- `loss_rate` が業務閾値以下
- `skipped.insufficient_observations` が多すぎない
- high-risk requestを対象にする場合は `--include-high-risk` のreplay結果を別途レビューする

CIまたはrelease jobでは `gate-bandit` でpromotion gateを評価します。

```bash
uv run --project ./prd tune-orchestrator gate-bandit \
  --replay ./prd/artifacts/runtime/bandit-replay.json \
  --out ./prd/artifacts/runtime/bandit-promotion.json \
  --report ./prd/artifacts/runtime/bandit-promotion.md \
  --min-evaluated-requests 30 \
  --min-mean-reward-delta 0.0 \
  --max-loss-rate 0.05 \
  --max-switch-rate 0.50 \
  --max-skip-rate 0.80
```

`status=fail` の場合、既定ではexit code 1で終了します。手元確認だけで失敗終了を避ける場合は
`--no-fail` を付けます。

runtimeで適用:

```bash
uv run --project ./prd tune-orchestrator run \
  "Kubernetes上のPostgreSQLが遅い。PVCはNFSです" \
  --router-url http://127.0.0.1:18001/v1 \
  --orchestrator-url http://127.0.0.1:18003/v1 \
  --orchestrator-model tune-orchestrator-ft \
  --model-config ./prd/config/model-endpoints.yaml \
  --bandit-state ./prd/artifacts/runtime/bandit-state.json \
  --bandit-min-observations 3
```

canary rolloutでは、最初に低いtraffic比率から開始します。

```bash
uv run --project ./prd tune-orchestrator run \
  "Kubernetes上のPostgreSQLが遅い。PVCはNFSです" \
  --router-url http://127.0.0.1:18001/v1 \
  --orchestrator-url http://127.0.0.1:18003/v1 \
  --orchestrator-model tune-orchestrator-ft \
  --model-config ./prd/config/model-endpoints.yaml \
  --bandit-state ./prd/artifacts/runtime/bandit-state.json \
  --bandit-traffic-percent 10 \
  --bandit-rollout-salt prod-2026w32
```

rollout外のrequestはserved decisionを維持し、`trace.graph.selection_metadata.bandit_canary` に
候補graph、score、bucket、`sampled=false` を保存します。rollout対象になったrequestは
`selector_type=bandit_policy` として実行されます。

canary開始後はlive traceのhealth gateを実行します。

```bash
uv run --project ./prd tune-orchestrator monitor-bandit \
  --traces ./prd/artifacts/runtime/traces.jsonl \
  --out ./prd/artifacts/runtime/bandit-monitor.json \
  --report ./prd/artifacts/runtime/bandit-monitor.md \
  --min-bandit-traces 30 \
  --max-bandit-failure-rate 0.05 \
  --max-relative-failure-rate 0.10 \
  --max-bandit-p95-latency-ms 60000 \
  --max-bandit-mean-cost-usd 1.0
```

`monitor-bandit` が `status=fail` を返した場合はrolloutを停止し、`--bandit-traffic-percent 0`
または `--bandit-state` の無効化でdeterministic/learned served decisionへ戻します。

rollout controllerで次のtraffic比率を生成します。

初回canary:

```bash
uv run --project ./prd tune-orchestrator plan-bandit-rollout \
  --promotion ./prd/artifacts/runtime/bandit-promotion.json \
  --bandit-state ./prd/artifacts/runtime/bandit-state.json \
  --current-traffic-percent 0 \
  --step-percent 10 \
  --max-traffic-percent 50 \
  --max-age-hours 24 \
  --rollout-salt prod-2026w32 \
  --out ./prd/artifacts/runtime/bandit-rollout.json \
  --report ./prd/artifacts/runtime/bandit-rollout.md
```

2段目以降:

```bash
uv run --project ./prd tune-orchestrator plan-bandit-rollout \
  --promotion ./prd/artifacts/runtime/bandit-promotion.json \
  --monitor ./prd/artifacts/runtime/bandit-monitor.json \
  --bandit-state ./prd/artifacts/runtime/bandit-state.json \
  --current-traffic-percent 10 \
  --step-percent 10 \
  --max-traffic-percent 50 \
  --min-monitor-bandit-traces 30 \
  --max-age-hours 24 \
  --rollout-salt prod-2026w32 \
  --out ./prd/artifacts/runtime/bandit-rollout.json
```

runtimeではcontrollerが生成したconfigを指定します。

```bash
uv run --project ./prd tune-orchestrator run \
  "Kubernetes上のPostgreSQLが遅い。PVCはNFSです" \
  --router-url http://127.0.0.1:18001/v1 \
  --orchestrator-url http://127.0.0.1:18003/v1 \
  --orchestrator-model tune-orchestrator-ft \
  --model-config ./prd/config/model-endpoints.yaml \
  --bandit-state ./prd/artifacts/runtime/bandit-state.json \
  --bandit-rollout-config ./prd/artifacts/runtime/bandit-rollout.json
```

`plan-bandit-rollout` のaction:

- `advance`: 次のtraffic比率へ進める
- `hold`: 現在比率を維持する
- `rollback`: `runtime.enabled=false` またはrollback trafficへ戻す。既定ではexit code 1

runtimeへ反映する前に、rollout artifact chainを検証します。

```bash
uv run --project ./prd tune-orchestrator verify-bandit-rollout \
  --rollout ./prd/artifacts/runtime/bandit-rollout.json \
  --bandit-state ./prd/artifacts/runtime/bandit-state.json \
  --promotion ./prd/artifacts/runtime/bandit-promotion.json \
  --monitor ./prd/artifacts/runtime/bandit-monitor.json \
  --require-monitor \
  --out ./prd/artifacts/runtime/bandit-rollout-verification.json \
  --report ./prd/artifacts/runtime/bandit-rollout-verification.md
```

初回canaryでmonitorがまだ無い場合は `--monitor` と `--require-monitor` を外します。

検証済みの一式をrelease manifestへ固定します。

```bash
uv run --project ./prd tune-orchestrator build-bandit-release \
  --rollout ./prd/artifacts/runtime/bandit-rollout.json \
  --bandit-state ./prd/artifacts/runtime/bandit-state.json \
  --promotion ./prd/artifacts/runtime/bandit-promotion.json \
  --monitor ./prd/artifacts/runtime/bandit-monitor.json \
  --verification ./prd/artifacts/runtime/bandit-rollout-verification.json \
  --require-monitor \
  --out ./prd/artifacts/runtime/bandit-release.json \
  --report ./prd/artifacts/runtime/bandit-release.md
```

runtimeへはrelease manifestを渡すのを推奨します。

```bash
uv run --project ./prd tune-orchestrator activate-bandit-release \
  --manifest ./prd/artifacts/runtime/bandit-release.json \
  --bandit-state ./prd/artifacts/runtime/bandit-state.json \
  --out ./prd/artifacts/runtime/bandit-current.json \
  --channel production
```

activation後はrelease registryへ記録します。registryは同じrelease id、channel、manifest digestの再登録を1件へ畳み込みます。

```bash
uv run --project ./prd tune-orchestrator record-bandit-release \
  --current ./prd/artifacts/runtime/bandit-current.json \
  --manifest ./prd/artifacts/runtime/bandit-release.json \
  --registry ./prd/artifacts/runtime/bandit-release-registry.json
```

障害時は現在のrelease idを除外し、同じchannelの直近pass済みreleaseをrollback候補として選びます。

```bash
uv run --project ./prd tune-orchestrator select-bandit-rollback \
  --registry ./prd/artifacts/runtime/bandit-release-registry.json \
  --current-release-id bandit-release-current-id \
  --channel production \
  --out ./prd/artifacts/runtime/bandit-rollback.json \
  --report ./prd/artifacts/runtime/bandit-rollback.md
```

rollbackを適用する場合は、候補に記録されたmanifest digestと実ファイルを照合してからcurrent pointerを書き換えます。
`--bandit-state` を指定すると、候補releaseに紐づくstate digestも検証します。

```bash
uv run --project ./prd tune-orchestrator apply-bandit-rollback \
  --rollback ./prd/artifacts/runtime/bandit-rollback.json \
  --manifest ./prd/artifacts/runtime/bandit-release-previous.json \
  --bandit-state ./prd/artifacts/runtime/bandit-state-previous.json \
  --out ./prd/artifacts/runtime/bandit-current.json \
  --registry ./prd/artifacts/runtime/bandit-release-registry.json
```

runtime起動前の最終gateとして、current pointer、release manifest、bandit state、release registryを照合します。

```bash
uv run --project ./prd tune-orchestrator verify-bandit-current \
  --current ./prd/artifacts/runtime/bandit-current.json \
  --bandit-state ./prd/artifacts/runtime/bandit-state.json \
  --registry ./prd/artifacts/runtime/bandit-release-registry.json \
  --require-registry \
  --out ./prd/artifacts/runtime/bandit-current-verification.json
```

最後にruntime bundleを作成します。bundleはcurrent pointer、release manifest、bandit state、current verification、
release registry、graph定義、model configのdigestを固定します。

```bash
uv run --project ./prd tune-orchestrator build-bandit-runtime-bundle \
  --current ./prd/artifacts/runtime/bandit-current.json \
  --bandit-state ./prd/artifacts/runtime/bandit-state.json \
  --current-verification ./prd/artifacts/runtime/bandit-current-verification.json \
  --registry ./prd/artifacts/runtime/bandit-release-registry.json \
  --graphs ./prd/graphs \
  --model-config ./prd/config/model-endpoints.yaml \
  --out ./prd/artifacts/runtime/bandit-runtime-bundle.json
```

systemd unit、Kubernetes Job、release pipelineなどの起動直前にbundle照合を入れます。

```bash
uv run --project ./prd tune-orchestrator verify-bandit-runtime-bundle \
  --bundle ./prd/artifacts/runtime/bandit-runtime-bundle.json \
  --current ./prd/artifacts/runtime/bandit-current.json \
  --bandit-state ./prd/artifacts/runtime/bandit-state.json \
  --current-verification ./prd/artifacts/runtime/bandit-current-verification.json \
  --registry ./prd/artifacts/runtime/bandit-release-registry.json \
  --graphs ./prd/graphs \
  --model-config ./prd/config/model-endpoints.yaml \
  --out ./prd/artifacts/runtime/bandit-runtime-bundle-verification.json
```

runtimeへはcurrent pointerを渡すのを推奨します。

```bash
uv run --project ./prd tune-orchestrator run \
  "Kubernetes上のPostgreSQLが遅い。PVCはNFSです" \
  --router-url http://127.0.0.1:18001/v1 \
  --orchestrator-url http://127.0.0.1:18003/v1 \
  --orchestrator-model tune-orchestrator-ft \
  --model-config ./prd/config/model-endpoints.yaml \
  --bandit-state ./prd/artifacts/runtime/bandit-state.json \
  --bandit-release-current ./prd/artifacts/runtime/bandit-current.json
```

artifact binding:

- `bandit-rollout.json.artifacts.bandit_state_digest` に `bandit-state.json` のSHA-256 digestを保存する
- `promotion_digest`、`monitor_digest` も保存し、`verify-bandit-rollout` で照合する
- `bandit-release.json` は検証済みrollout一式とruntime設定を固定する
- `bandit-current.json` は現在有効なrelease manifestのpathとdigestを保存する
- `bandit-release-registry.json` はactivation履歴、manifest digest、state digest、traffic比率を保存する
- `bandit-rollback.json` は現在releaseを除外した直近の正常release候補を保存する
- runtimeは `--bandit-state` と `--bandit-rollout-config` のdigest一致を検証する
- runtimeは `--bandit-release-manifest` と `--bandit-state` のdigest一致、有効期限、release statusも検証する
- runtimeは `--bandit-release-current` のmanifest digest、release id、有効期限も検証する
- digest不一致、format不一致、binding欠落は既定でfail closed
- `bandit-rollout.json.expires_at` を過ぎたconfigも既定でfail closed
- controller外で作った古いconfigを一時的に許可する場合のみ `--bandit-allow-unbound-rollout` を使う
- 期限切れconfigを緊急時にだけ使う場合は `--bandit-allow-expired-rollout` を明示する
- banditが適用されたtraceには `selection_metadata.bandit.rollout_config_digest` が保存される
- rollout外候補traceには `selection_metadata.bandit_canary.rollout_config_digest` が保存される

探索的に未観測候補も許可する場合:

```bash
--bandit-explore-unobserved --bandit-exploration-weight 0.2
```

既定ではhigh-risk requestにbandit policyを適用しません。明示的に許可する場合:

```bash
--bandit-include-high-risk
```

banditがdecisionを上書きした場合、`trace.graph.selector_type` は `bandit_policy` になり、
`trace.graph.selection_metadata.bandit` にcontext、arm、score、元のserved graphが保存されます。

banditの制約:

- policy gateを迂回しない
- 未登録modelを作らない
- 任意tool実行をしない
- 観測数不足のcontextではserved decisionを維持する
- 候補はserved decision、deterministic baseline、single/parallel alternativeに限定する

## Observability

traceは既定で次へ追記されます。

```text
prd/artifacts/runtime/traces.jsonl
```

集計:

```bash
uv run --project ./prd tune-orchestrator trace-report \
  --traces ./prd/artifacts/runtime/traces.jsonl \
  --out ./prd/artifacts/runtime/trace-report.md
```

最低限見る指標:

- `trace_completeness`
- `stop_reasons.completed`
- `stop_reasons.node_failed`
- `latency_p95_ms`
- `mean_cost`
- `verifier_pass_rate`
- `repair_trigger_rate`
- `repair_success_rate`
- `shadow_trace_rate`
- `shadow_success_rate`
- `bandit_trace_rate`
- `bandit_switches`
- `bandit_contexts`
- `bandit_canary_eligible`
- `bandit_canary_sampled`
- `selector_type`: `deterministic`、`learned_orchestrator`、`deterministic_fallback`
- `fallback_reason`: FT planが拒否された理由

## Failure Details

`--debug` の `trace.evaluation.failure_detail` を確認します。

| failure_detail | 主な意味 | 対応 |
| --- | --- | --- |
| `model_endpoint_unreachable` | specialist endpointに接続できない | LLMサーバ、port、firewall、`base_url` を確認 |
| `model_credentials_missing` | `api_key_env` の環境変数がない | 環境変数を設定 |
| `model_response_invalid` | OpenAI互換レスポンス形式またはJSON schemaが不正 | gateway応答、verifier JSON出力を確認 |
| `model_endpoint_not_found` | `/v1/chat/completions` が404 | `base_url` またはgateway pathを確認 |
| `model_alias_missing` | graphが要求するaliasがconfigにない | `model-endpoints.yaml` にaliasを追加 |

FT planの失敗は `trace.graph.fallback_reason` を確認します。

## Release Checklist

- `uv run --project ./prd python -m unittest discover -s ./prd/src/test -v` が成功
- `uv run --project ./prd tune-orchestrator validate-graphs` が成功
- `doctor --model-config ...` が `status: pass`
- PRD FTを使う場合、`doctor --adapter ...` または `doctor --orchestrator-url ... --probe-orchestrator` が成功
- `evaluate-ft` のmetricが昇格基準を満たす
- offline replayでdeterministic selectorより悪化しない
- router serviceがPoC成果物ではなく本番管理されたartifactになっている
- specialist endpointの全aliasがconfigに存在する
- trace出力先の容量、保全、ローテーション方針がある
- `--max-cost`、`--max-latency-ms`、`--max-steps` がSLAに合っている

## Troubleshooting

port確認:

```bash
sudo ss -ltnp | grep ':18001'
sudo ss -ltnp | grep ':18002'
sudo ss -ltnp | grep ':18003'
```

routerは通るが `node_failed`:

```bash
uv run --project ./prd tune-orchestrator doctor \
  --router-url http://127.0.0.1:18001/v1 \
  --model-config ./prd/config/model-endpoints.yaml \
  --probe-model-chat
```

PRD FTが使われていない:

```bash
uv run --project ./prd tune-orchestrator run \
  "Kubernetes上のPostgreSQLが遅い。PVCはNFSです" \
  --router-url http://127.0.0.1:18001/v1 \
  --orchestrator-url http://127.0.0.1:18003/v1 \
  --orchestrator-model tune-orchestrator-ft \
  --model-config ./prd/config/model-endpoints.yaml \
  --debug
```

`trace.graph.selector_type` が `learned_orchestrator` ならPRD FTが採用されています。`deterministic_fallback` の場合は `trace.graph.fallback_reason` を確認します。

local adapterで確認:

```bash
uv run --project ./prd --extra training tune-orchestrator run-ft \
  "Kubernetes上のPostgreSQLが遅い。PVCはNFSです" \
  --adapter ./prd/artifacts/orchestrator-lora \
  --router-url http://127.0.0.1:18001/v1 \
  --mock \
  --debug
```
