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
- `artifacts/ft-data/metadata.json`

`preferences.jsonl`は同一requestに対するchosen/rejected planを保持し、DPOなどの次段階で利用する。

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

現在のFTは固定graph集合からのplan生成であり、Fugu相当の自由なagent assemblyではない。段階的に以下を追加する。

1. model capability catalogを学習入力へ追加し、同一domain内の複数modelを選択可能にする。
2. 固定graph idに加え、制約付きnode/edge compositionを生成する。
3. verifier、repair、synthesisを含むtrajectory全体へpreference labelを付ける。
4. shadow explorationで未選択候補のoutcomeを継続収集する。
5. SFT、preference optimization、bounded contextual banditの順にonline regretを改善する。

安全性と比較可能性を維持するため、各段階でdeterministic fallbackとOracle replayを残す。
