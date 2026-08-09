# Learned Orchestrator FT Design

## Objective

Tune Orchestratorを、6カテゴリ分類器のtop-1を固定ルールへ渡すシステムから、リクエストごとにgraph、担当モデル、委譲目的、検証、統合方針を生成する学習済みオーケストレータへ拡張する。

Sakana Fuguに近づくため、学習対象を単純なdomain labelではなく、最終outcomeを改善するorchestration planとする。ただし初期段階では、任意コードや未登録graphを生成させず、登録済みgraph/modelの範囲で学習する。

## Architecture

```text
User Request
    |
    +--> Hard Policy Gate ---------------------------> safe response / handoff
    |
    +--> Existing FT Classifier --> label scores
                                    |
                                    v
                         Learned Orchestrator LLM
                         Qwen2.5-1.5B + LoRA
                                    |
                         structured orchestration plan
                                    |
                         schema / graph / model validation
                           |                    |
                         valid                invalid
                           |                    |
                           v                    v
                      Graph Executor     Deterministic Fallback
                           |
                   specialists / verifier / repair / synthesis
                           |
                    answer + outcome trace
                           |
                  offline table / human review
                           |
                    next SFT / preference data
```

## Training Target

入力はrequest、既存分類器scores、risk、budget、利用可能graph/modelに限定する。候補ごとの実測qualityは推論時に利用できないため入力へ含めない。

教師出力は以下を持つJSON planである。

- `graph_id`
- `primary_labels`
- `selected_labels`
- `confidence`
- `risk_level`
- `reason`
- `delegations[].label`
- `delegations[].model`
- `delegations[].objective`
- `synthesis_strategy`

Graph Executorはdelegation objectiveを各specialistのsystem promptへ追加し、learned planを実際の実行へ反映する。

## Data Generation

日本語domain supervisionの正本は`prd/artifacts/`とする。`train.json` 3,323件、`dev.json` 716件、`test.json` 761件、全体4,800件で、既存6カテゴリ分類とlearned orchestratorのbootstrap/evaluationへ使用する。

Fugu型のoutcome学習では、domain labelだけを正解にせず、各Queryへ候補model/graphの実行結果を付加する。データソースは次の二種類とする。

1. Offline outcome table

全Queryを全候補model/graphで実行した`candidate_results`から教師planを選ぶ。最高qualityとの差が`0.02`以内の候補を同等品質帯とし、cost/latency penaltyが小さい候補を選ぶ。

2. Reviewed execution trace

`user_rating >= 4`または`review_label`が`approved`、`success`、`preferred`のtraceのみをSFTへ戻す。未評価traceや低評価traceは正例として使わない。

生成物:

- `artifacts/ft-data/train.jsonl`
- `artifacts/ft-data/dev.jsonl`
- `artifacts/ft-data/preferences.jsonl`
- `artifacts/ft-data/trajectory_preferences.jsonl`
- `artifacts/ft-data/metadata.json`

`preferences.jsonl`は同一requestに対するchosen/rejected planを保持し、DPOなどの次段階で利用する。
`trajectory_preferences.jsonl`は同一requestに対するreview済み実行traceを比較し、chosen/rejected trajectoryから
plan preferenceを作る。promptにはoutcome、rating、node outputを含めず、review情報はmetadataにのみ保存する。

## Training

初期構成:

| Item | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Method | LoRA causal LM SFT |
| LoRA rank / alpha | `16 / 32` |
| Max length | `2048` |
| Epochs | `2` |
| Learning rate | `2e-4` |
| Loss target | assistant JSON plan tokens only |

system/user prompt tokenはlossからmaskし、assistant planだけを学習する。base modelとhyperparameterはCLIから変更可能とする。

## Runtime Safety

学習モデルを信頼境界にしない。

- destructive/credential exfiltration policy gateはモデル呼び出し前に実行
- graph idはallowlist検証
- selected labelsは既存6ラベルに限定
- delegation modelはmodel catalogのallowlist検証
- high riskで未検証`single_specialist`を禁止
- JSON/schema違反、timeout、未許可値はdeterministic fallback
- fallback reasonをtraceへ保存

## Evaluation Gates

モデル単体:

- valid plan rate
- graph accuracy
- selected labels exact match / F1
- model allowlist violation
- fallback rate

outcome:

- quality
- mean regret
- cost / latency
- unnecessary multi-agent rate
- missed collaboration rate
- verifier / repair success
- policy bypass count

SFT modelはplan accuracyだけで昇格させない。全候補結果上のoffline replayでdeterministic selectorよりmean regretを悪化させず、quality-cost Paretoを改善した場合のみshadow deploymentへ進める。

## Path Toward Fugu-like Orchestration

現在のFTは、登録済みgraph集合から1つを選び、selected labelsとdelegation objectiveを生成する段階である。
これは業務投入に必要な安全性と比較可能性を優先したMVPであり、Fugu相当の自由なagent assemblyではない。

Fugu-likeな方向へ進める場合も、最初から任意のagent graphを生成させない。運用で壊れにくくするため、次の順で
「選択肢を増やす」「構造を増やす」「学習信号を濃くする」「探索を制御する」を分離して進める。

### Stage 1: Model Capability Catalog

実装済み。`model-endpoints.yaml` の各model aliasにcapability metadataを持たせ、
learned orchestratorの入力、plan検証、FTデータ生成、`doctor`検査へ接続する。
同一domain内で複数modelを選べるようにするが、選択対象は事前登録されたmodel aliasだけに限定する。

追加する入力:

- `model_catalog[].alias`
- `model_catalog[].domains`
- `model_catalog[].strengths`
- `model_catalog[].context_window`
- `model_catalog[].latency_tier`
- `model_catalog[].cost_tier`
- `model_catalog[].safety_profile`
- `model_catalog[].supports_json`

期待するplan変更:

- `delegations[].model` がdomain固定のdefault aliasではなく、catalog内のaliasから選ばれる
- `reason` にmodel選択理由を含める
- `allowed_models` にないaliasは従来通りdeterministic fallback
- `domains` が合わないaliasを選んだ場合もdeterministic fallback

昇格条件:

- model allowlist violationが0
- domain別のbest-single baselineよりmean regretが悪化しない
- 高cost modelの不要選択率が許容閾値以下

### Stage 2: Bounded Graph Composition

実装済み。固定graph idだけでなく、制約付きのnode/edge planを生成、検証、既存`GraphExecutor`で実行できる。
ここでも任意コードや任意tool呼び出しは許可しない。`plan_type: bounded_graph` のみを受け付ける。

生成可能なnode type:

- `specialist`
- `verifier`
- `repair`
- `synthesizer`
- `clarifier`
- `policy`

制約:

- DAGのみ
- `max_nodes`、`max_parallelism`、`max_steps` をbudgetで制限
- high-risk requestは必ず`verifier`を通る
- high-risk requestのfinal outputはverifier evidenceに依存する
- 外部model aliasはcatalog allowlist内のみ
- final nodeは`synthesizer`、`specialist`、`clarifier`、`policy`のいずれか
- schema validationに失敗した場合は既存固定graph selectorへfallback

この段階の出力は、既存の`graph_id` planとは別formatにする。

```json
{
  "plan_type": "bounded_graph",
  "nodes": [
    {
      "id": "storage_1",
      "role": "specialist",
      "model": "storage-long-context",
      "dependencies": [],
      "objective": "Check PVC, NFS latency, mount options, and I/O saturation."
    },
    {
      "id": "database_1",
      "role": "specialist",
      "model": "database-specialist",
      "dependencies": [],
      "objective": "Check PostgreSQL waits, checkpoints, locks, and query symptoms."
    },
    {
      "id": "synthesis_1",
      "role": "synthesizer",
      "model": "general-synthesizer",
      "dependencies": ["storage_1", "database_1"],
      "objective": "Prioritize evidence and produce a safe incident response plan."
    }
  ],
  "final_node": "synthesis_1"
}
```

昇格条件:

- generated graph validation pass rateが業務閾値以上
- cycle、unknown dependency、unknown model、policy bypassが0
- 固定graph baselineよりquality-cost Paretoが改善
- fallback後も回答品質がdeterministic baselineを下回らない

実装上の保存先:

- `RouteDecision.generated_graph`
- `trace.graph.generated_graph`
- approved traceからのFT再投入

### Stage 3: Trajectory-Level Preference Data

実装済み。SFTの正例は最終planだけを教師にするが、review済みtrace同士を比較して
`trajectory_preferences.jsonl` へchosen/rejected preferenceを出力できる。
同一queryに対して複数traceがあり、`user_rating`または`review_label`で優劣が付く場合のみ生成する。

収集対象:

- router scores
- selected plan
- 各node prompt
- 各node output
- verifier result
- repair loop count
- final answer
- latency、cost、token usage
- human review label
- incident outcome or task success label

preference単位:

- plan A vs plan B
- verifierあり vs verifierなし
- single specialist vs parallel specialists
- repair前 answer vs repair後 answer
- synthesizer strategyの差分

学習への使い方:

- SFT: approved trajectoryのplan生成
- preference optimization: chosen/rejected planまたはtrajectory
- offline replay: 同一queryでのregret、cost、latency比較

実装済みの制約:

- promptにはreview label、rating、final answer、node outputを入れない
- chosen/rejectedの比較根拠はmetadataへ保存する
- high-risk traceはverifier nodeが含まれない限りtrajectory preferenceに使わない
- bounded graph traceも`generated_graph`からplanを復元してpreference化する

昇格条件:

- review未完了traceを正例に混ぜない
- high-risk traceはverifier結果とreview labelが揃うまで学習対象外
- preference dataのchosen/rejectedにoutcome leakageを入れない

### Stage 4: Shadow Exploration

実装済み。本番trafficで未選択候補のoutcomeを継続収集する。ただし、ユーザーに返す回答は現行winnerだけにする。
`run` / `run-ft` に `--shadow-mode` を付けると、比較用graphを実行し、親traceの
`shadow_executions[]` に保存する。

探索方法:

- 低risk、低cost、低latencyの範囲でshadow graphを実行
- expensive modelはsampling rateを強く制限
- high-risk requestはshadowでもpolicy/verifier制約を維持
- shadow結果は人間レビューまたはoffline judgeを通してから学習へ戻す

記録するtrace field:

- `shadow_executions[].reason`
- `shadow_executions[].status`
- `shadow_executions[].decision`
- `shadow_executions[].trace`

実装済みの探索mode:

- `deterministic-baseline`: learned/bounded graphがserveされた場合にdeterministic selectorをshadow実行
- `alternatives`: deterministic baselineに加え、single/parallelの強制代替候補を最大件数までshadow実行

実装済みの安全制約:

- 既定ではhigh-risk requestでshadowを実行しない
- `--shadow-include-high-risk` を指定しても既存policy/verifier制約は維持
- shadow failureはserveされた回答を壊さず、`shadow_executions[].status=failed` として保存
- trace reportにshadow trace rate、shadow runs、shadow success rateを出力

昇格条件:

- shadow実行が本番SLAを侵害しない
- shadow cost budgetを超えない
- shadow結果が学習へ入る前にreview gateを通る

### Stage 5: Online Regret Optimization

実装済み。SFTだけでなく、review済みtraceからbounded contextual bandit stateを構築し、
実行時に安全な候補集合の中からobserved rewardが高いarmを選べる。
現時点では、policy bypassや未登録model探索は行わず、serve候補、deterministic baseline、single/parallel alternativeの範囲に限定する。

推奨順序:

1. Deterministic selectorをbaselineとして固定
2. Learned orchestrator SFTをshadowで比較
3. Preference optimizationでplan preferenceを改善
4. Bounded contextual banditでcatalog/model/graph選択のregretを改善
5. Canary rolloutでtraffic比率を段階的に上げる

実装済みのCLI:

- `build-bandit-state`
- `replay-bandit`
- `gate-bandit`
- `monitor-bandit`
- `plan-bandit-rollout`
- `verify-bandit-rollout`
- `build-bandit-release`
- `activate-bandit-release`
- `run --bandit-state ...`
- `run-ft --bandit-state ...`

bandit context:

- risk level
- top router label
- secondary labels

bandit arm:

- graph id
- selected labels
- delegation model aliases
- generated graph digest

reward source:

- `evaluation.user_rating`
- `evaluation.review_label`
- cost penalty
- latency penalty
- failed stop reason penalty

offline replay:

- served traceとreview済みshadow traceを同一requestの候補集合として比較する
- `leave_one_out` を既定で有効にし、同じrequestの観測をcontext統計から抜いてから選択を再現する
- `mean_reward_delta`、`loss_rate`、`switch_rate`、skip理由を出力する
- replay結果はJSONとMarkdownで保存できる

promotion gate:

- `gate-bandit` がreplay結果を閾値評価し、pass/failをJSONとMarkdownで保存する
- CIやrelease jobで `status=fail` をexit code 1として扱える
- 最低観測数、平均reward改善、loss rate、switch rate、skip rateを明示閾値にする

canary rollout:

- `--bandit-traffic-percent` でtraffic比率を制限する
- `--bandit-rollout-salt` とrequest/context/armから決定的bucketを作る
- rollout外ではserved decisionを維持し、`selection_metadata.bandit_canary` に候補とbucketを保存する
- rollout内でのみ `selector_type=bandit_policy` に切り替える

live rollout monitor:

- `monitor-bandit` がlive traceからbandit適用trace、baseline trace、canary eligible/sample数を集計する
- reviewが揃う前でもstop reason、failure rate、P95 latency、mean costでhealth gateを評価する
- baseline failure rateとの差分を `relative_failure_rate` として監視する
- monitor fail時はtraffic percentを0へ戻すか、bandit state指定を外してrollbackする

rollout controller:

- `plan-bandit-rollout` がpromotion/monitor結果から `advance`、`hold`、`rollback` を決める
- 初回rolloutはpromotion passだけで開始できる
- 2段目以降はmonitor passと十分なbandit trace数を要求する
- 出力は `bandit-rollout.json` として保存し、runtimeの `--bandit-rollout-config` で読み込める
- rollback actionでは既定でexit code 1にし、release jobを停止できる

artifact binding:

- `bandit-rollout.json` は使用した `bandit-state`、promotion、monitorのdigestを保存する
- `verify-bandit-rollout` がdigest chain、format、有効期限をrelease前に検証する
- `build-bandit-release` が検証済みartifact chainとruntime設定をrelease manifestへ固定する
- `activate-bandit-release` がrelease manifestを検証し、現在有効な `bandit-current.json` を更新する
- runtimeは `--bandit-state` と rollout config 内の `bandit_state_digest` を照合する
- runtimeは `--bandit-release-manifest` を読み、release status、state digest、有効期限を検証してからbanditを有効化する
- runtimeは `--bandit-release-current` を読み、current pointerとrelease manifestのdigest一致を検証する
- digest不一致またはbinding欠落は既定でfail closedする
- rollout configは `expires_at` を持ち、期限切れも既定でfail closedする
- 適用traceには `rollout_config_digest` を残し、後からどのcontroller出力で選択されたかを追跡できる
  release manifest経由では `release_id` と `release_manifest_digest` も保存する

banditで探索してよい範囲:

- model alias
- graph template
- verifier有無。ただしhigh-riskは常に必須
- synthesis strategy
- low-risk requestのparallelism

banditで探索してはいけない範囲:

- policy restriction bypass
- 未登録model
- 任意tool実行
- secretやcredentialを含むprompt加工
- budget上限の変更

実装済みの安全制約:

- 既定ではhigh-risk requestにbandit policyを適用しない
- banditは候補集合を新規生成せず、既存validatorを通った候補だけを選ぶ
- 観測数が`--bandit-min-observations`未満のcontextではserved decisionを維持
- 選択結果は`trace.graph.selection_metadata.bandit`へ保存する
- `build-bandit-state` はreview済みshadow outcomeを観測として取り込む
- `replay-bandit` で昇格判定してからruntime適用する
- `gate-bandit` 通過後も、canary trafficを段階的に上げる
- `monitor-bandit` でlive canaryの失敗率、latency、costを継続監視する
- `plan-bandit-rollout` が生成したruntime configだけを本番runに渡す
- rollout configとbandit stateのartifact bindingをruntimeで検証する
- rollout configの期限をruntimeで検証する
- release manifestを本番投入単位として保存し、runtimeで検証する
- current release pointerを本番runtimeの参照点にする

### Persistent Safety Gates

どの段階でも次は維持する。

- deterministic fallback
- graph/model allowlist
- budget enforcement
- policy gate before learned model
- high-risk verifier requirement
- JSON/schema validation
- trace completeness requirement
- Oracle replayによるoffline比較
- rollback可能なmodel/config versioning

Fugu-like orchestrationの完成度は、自由度の高さではなく、制約下でのregret改善、障害時のfallback品質、
安全性違反ゼロ、運用者が説明できるtraceの4点で判断する。
