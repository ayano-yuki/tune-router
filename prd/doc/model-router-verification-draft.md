# Model Router Verification Draft

> Status: exploratory design. Production向けのnormativeなartifact contract、utility定義、baseline/promotion要件は `../TODO.md` のPRD-016〜PRD-018を正本とする。本書と差異がある場合はPRD-016〜PRD-018を優先する。

## 1. Purpose

本ドキュメントは、TuneScope Model Orchestrator / Model Router の検証方法の草案である。

検証の主目的は、FT Router の分類精度を上げることではない。モデルルーターの価値は、各Queryに対して品質、コスト、レイテンシを考慮しながら、最終的に最も良い実行先または実行グラフを選べるかで決まる。

したがって、評価の中心は次の問いに置く。

> Best Single Model と比べて、Qualityを維持または改善しながらCost/Latencyをどれだけ削減できるか。OracleとのRegretをどこまで縮められるか。

## 2. Evaluation Principles

### 2.1 Classification Accuracy Is Not Enough

現行PoCの `Storage` / `Network` / `Coding` / `Security` / `Database` / `General` 分類は重要なrouting signalである。ただし、正しくカテゴリ分類できたことは、良いルーティングと同義ではない。

例:

- Routerが最高品質モデルを外しても、選択モデルのQualityがほぼ同じなら実害は小さい
- Routing Accuracyが高くても、常に高コストモデルを選ぶならRouter導入価値は弱い
- 複合タスクでは単一カテゴリ正解より、複数専門家を組み合わせた最終回答品質が重要

### 2.2 Evaluate Router Decisions by Outcome

Routerの出力を直接評価するのではなく、Routerが選んだモデル/グラフで得られた最終結果を評価する。

評価対象:

- 選択したモデルまたは実行グラフ
- 生成された回答
- 回答品質
- 推論コスト
- ルーター込みのE2Eレイテンシ
- Oracleとの差分
- Best Singleとの差分

## 3. Metrics

### 3.1 Routing Accuracy

各Queryで最も高いQualityを出したモデル/グラフをOracleとし、Routerが同じ選択をできた割合。

```text
routing_accuracy = count(router_choice == oracle_choice) / total_queries
```

用途:

- RouterがOracle選択をどれだけ当てているかを見る
- 分類器やgraph selectorの基本性能を見る

限界:

- 2位モデルのQualityが1位と僅差でも不正解になる
- Cost/Latencyを考慮しない

### 3.2 Quality Score

Routerが選択したモデル/グラフで実際に得られた回答品質。

タスク別の評価例:

| Task Type | Primary Quality Metric |
| --- | --- |
| Coding | unit test pass rate, benchmark score, static check result |
| Infra / SRE | human rubric, runbook correctness, diagnosis coverage |
| Database | diagnosis correctness, query/fix validity, safety |
| Security | defensive usefulness, policy compliance, evidence quality |
| QA | exact match, F1, retrieval-grounded correctness |
| General | human preference, LLM judge, rubric score |

原則として、可能なものは自動評価を優先し、自動評価が難しいものはrubric付き人手評価またはLLM judgeを使う。

### 3.3 Regret

現行MVPではOracleとRouter選択のQuality差を使う。

```text
regret(query) = quality(oracle_choice) - quality(router_choice)
mean_regret = average(regret(query))
```

Production benchmarkではPRD-017に従い、profile別のQuality/Cost/Latency/Failureを統合したutility差を正規のregretとする。quality-only regretは診断metricとして併記する。

重要性:

- Routing AccuracyよりRouterの実害を表しやすい
- 「外したがほぼ同品質」と「外して大きく品質低下」を区別できる
- 学習ループやonline learningの主要指標にできる

### 3.4 Cost

Router本体と選択先モデル/グラフの合計コスト。

```text
total_cost = router_cost + selected_model_cost + verifier_cost + synthesizer_cost + retry_cost
```

集計:

- mean cost/query
- p50 / p95 cost/query
- cost per successful answer
- cost normalized quality

比較観点:

- 同じQualityならどれだけ安いか
- 同じCostならどれだけQualityが高いか
- Always Largeに対する削減率

### 3.5 Latency

Router込みの応答時間。

計測:

- router latency
- selected model latency
- graph execution latency
- verifier / repair / synthesis latency
- TTFT
- E2E latency p50 / p95 / p99

複数モデルを並列実行するグラフでは、合計呼び出し時間ではなくユーザーが待つE2E latencyを重視する。

### 3.6 Pareto Efficiency

Quality、Cost、Latencyを同時に見る。

主要な可視化:

- Quality vs Cost
- Quality vs Latency
- Regret vs Cost
- Cost reduction at fixed quality
- Quality improvement at fixed cost

Routerの価値は、Pareto frontier上でBest SingleやRule-basedより良い位置に出るかで判断する。

## 4. Required Baselines

Router単体では良し悪しを判断できないため、最低限以下を同一Datasetで比較する。

| Baseline | Description | Purpose |
| --- | --- | --- |
| Random | 候補モデルからランダム選択 | 下限確認 |
| Always Small | 最軽量/最安モデル固定 | 低コスト基準 |
| Best Single | 全Query平均で最もQualityが高い単一モデル固定 | Router導入価値の最低合格ライン |
| Always Large | 最高性能/最高コストモデル固定 | 品質上限に近い実運用基準 |
| Rule-based Router | キーワードやカテゴリルールで選択 | シンプル実装との比較 |
| Current FT Router | 現行Qwen LoRA分類器 | 現PoCの実力測定 |
| Graph Selector | 複数カテゴリ/検証/統合を使うMVP Router | PRD拡張案の評価 |
| Oracle | Queryごとに最高Qualityの選択肢を選ぶ | 理論上限 |

合格の基本条件:

- Best Singleより低いQualityなら不合格
- Best Singleと同等Qualityなら、CostまたはLatencyで明確に勝つ必要がある
- Always Largeより低いQualityでも、Regretが小さくCost削減が大きければ成功とみなせる

## 5. Router Algorithms to Compare

初期比較対象:

| Router | Input Features | Notes |
| --- | --- | --- |
| Random | none | sanity check |
| Rule-based | keywords, regex, domain rules | 最低限の運用ルール |
| Category Router | current 6-label classifier | 現行PoC |
| Embedding + kNN | query embedding | ラベル境界の滑らかな比較 |
| Logistic Regression / SVM | embedding + metadata | 軽量教師ありbaseline |
| MLP Router | embedding + router scores | 非線形baseline |
| FT LLM Router | fine-tuned small model | 現行路線の拡張 |
| LLM-as-a-Router | prompt-based decision | learned orchestration前の強いbaseline |
| Graph Selector | classifier scores + rules + risk + budget | MVP orchestrator |

将来比較:

- contextual bandit
- online learning router
- learned graph selector
- learned orchestrator LLM

## 6. Offline Evaluation Dataset

### 6.1 Core Requirement

オフライン評価の鍵は、全Queryを全候補モデル/グラフで実行しておくことである。

```text
Query 001
  Model A -> quality / cost / latency / response
  Model B -> quality / cost / latency / response
  Model C -> quality / cost / latency / response
  Graph X -> quality / cost / latency / response
```

これにより、Routerを毎回実モデル実行せず、過去の実行結果テーブル上で高速に比較できる。

### 6.2 Dataset Splits

既存:

- `poc/artifacts/test.json`
- `prd/artifacts/test.json`

追加:

- `eval/orchestration-single-domain.json`
- `eval/orchestration-boundary.json`
- `eval/orchestration-multi-domain.json`
- `eval/orchestration-risky-security.json`
- `eval/orchestration-clarification.json`

推奨比率:

| Set | Ratio | Purpose |
| --- | ---: | --- |
| single-domain | 35% | 単一専門モデル選択の確認 |
| boundary | 25% | Storage/Databaseなど境界検証 |
| multi-domain | 25% | graph orchestrationの価値確認 |
| risky-security | 10% | safety gate確認 |
| clarification | 5% | 低信頼時の質問返し確認 |

### 6.3 Record Schema

```json
{
  "query_id": "eval-000001",
  "query": "Kubernetes上のPostgreSQLが急に遅くなった。PVCはNFS...",
  "domain_labels": ["Database", "Storage", "Network"],
  "task_type": "infra_diagnosis",
  "risk_level": "normal",
  "candidate_results": [
    {
      "candidate_id": "database-specialist",
      "candidate_type": "model",
      "response": "...",
      "quality": 0.82,
      "cost": 0.012,
      "latency_ms": 4800,
      "quality_method": "human_rubric",
      "errors": []
    },
    {
      "candidate_id": "parallel_experts-v0.1.0",
      "candidate_type": "graph",
      "response": "...",
      "quality": 0.91,
      "cost": 0.034,
      "latency_ms": 9200,
      "quality_method": "human_rubric",
      "errors": []
    }
  ],
  "oracle": {
    "candidate_id": "parallel_experts-v0.1.0",
    "quality": 0.91
  }
}
```

### 6.4 Router Prediction Schema

```json
{
  "query_id": "eval-000001",
  "router_id": "ft-router-v1",
  "selected_candidate_id": "database-specialist",
  "router_latency_ms": 95,
  "router_confidence": 0.74,
  "router_scores": {
    "Database": 0.44,
    "Storage": 0.31,
    "Network": 0.18
  },
  "selected_graph": "single_specialist",
  "reason": "top1 above threshold"
}
```

## 7. Quality Rubric

### 7.1 General 0-1 Rubric

| Score | Meaning |
| ---: | --- |
| 1.0 | 完全に正しく、実行可能で、安全で、追加確認事項も適切 |
| 0.8 | 主要な答えは正しく、軽微な不足のみ |
| 0.6 | 部分的に有用だが、重要な観点が抜けている |
| 0.4 | 表面的で、実務上の切り分けや根拠が弱い |
| 0.2 | 誤りが多く、使用すると混乱を招く |
| 0.0 | 危険、無関係、または回答不能 |

### 7.2 Infra / SRE Rubric

評価観点:

- 原因候補の網羅性
- 切り分け順序の妥当性
- コマンド/確認項目の具体性
- 破壊的操作を避けているか
- Storage / Network / DB / App の境界を誤っていないか

### 7.3 Coding Rubric

評価観点:

- テストが通るか
- 変更範囲が適切か
- 既存設計に沿っているか
- バグ修正が根本原因に対応しているか
- 不要なリファクタをしていないか

### 7.4 Security Rubric

評価観点:

- 防御目的に限定されているか
- 危険な手順や攻撃実装を避けているか
- 証拠、影響、緩和策が明確か
- scope / authorization を前提にしているか
- policy gate が正しく作動したか

## 8. Offline Evaluation Procedure

### Step 1: Candidate Poolを定義

最初は以下を候補にする。

- `storage-specialist`
- `network-specialist`
- `coding-specialist`
- `security-specialist`
- `database-specialist`
- `general-fallback`
- `single_specialist`
- `specialist_with_verifier`
- `parallel_experts`

### Step 2: 全Queryを全候補で実行

各Queryについて、候補モデル/候補グラフをすべて実行する。

記録:

- response
- quality
- cost
- latency
- errors
- safety status

### Step 3: Oracleを作る

Queryごとに、Qualityが最大の候補をOracleとする。

同点または僅差の場合:

```text
if quality_diff <= 0.02:
  choose lower cost candidate as oracle
if cost_diff also small:
  choose lower latency candidate
```

これにより、品質差がほぼないときに高コストモデルをOracleにし続ける偏りを避ける。

### Step 4: 各Routerをoffline replay

RouterにはQueryのみを入力し、選択候補を出させる。

offline tableから選択候補の結果を引き、以下を計算する。

- selected quality
- selected cost
- selected latency
- regret
- routing accuracy
- safety violation

### Step 5: Baseline比較表を作る

```text
Router | Quality | Routing Acc | Mean Regret | Cost | Latency | Safety
```

必須比較:

- Random
- Always Small
- Best Single
- Always Large
- Rule-based
- FT Router
- Graph Selector
- Oracle

### Step 6: Pareto Frontierを描く

最低限のプロット:

- Quality vs Cost
- Quality vs Latency
- Mean Regret vs Cost
- Quality improvement vs Best Single
- Cost reduction vs Always Large

## 9. Graph Orchestrator Specific Evaluation

Model Routerだけでなく、Fugu型のgraph orchestrationを評価するため、以下を追加する。

### 9.1 Graph Selection Accuracy

Oracle modelだけでなく、Oracle graphを定義する。

例:

- 単一専門家で十分なら `single_specialist`
- 複数領域が必要なら `parallel_experts`
- 高リスク/高影響なら `specialist_with_verifier`
- 情報不足なら `clarify_first`

### 9.2 Unnecessary Multi-Agent Rate

単一モデルで十分なQueryに対し、複数モデルグラフを選んだ割合。

```text
unnecessary_multi_agent_rate =
  count(selected_graph_is_multi_agent and single_specialist_quality_close_to_oracle)
  / total_queries
```

これはコスト増の主要原因なので必ず見る。

### 9.3 Missed Collaboration Rate

複数専門家が必要なQueryに対し、単一専門家だけを選んでQualityが落ちた割合。

```text
missed_collaboration_rate =
  count(oracle_graph_is_multi_agent and selected_graph_is_single and regret > threshold)
  / total_queries
```

### 9.4 Loop Effectiveness

verification / repair loopの有効性を測る。

- verifier pass rate
- repair triggered rate
- repair success rate
- average loop count
- loop cost overhead
- loop latency overhead
- loop-induced quality improvement

## 10. Online Evaluation and Learning Loop

MVP後はonline evaluationへ進む。

### 10.1 Logging

本番/検証環境で以下を記録する。

```json
{
  "query_id": "online-000001",
  "query": "...",
  "available_candidates": ["small", "large", "parallel_experts"],
  "selected_candidate": "small",
  "router_id": "graph-selector-v0.1.0",
  "router_confidence": 0.71,
  "quality": null,
  "user_feedback": null,
  "cost": 0.006,
  "latency_ms": 1800,
  "exploration": false
}
```

### 10.2 Exploration

一部の低リスクトラフィックで、別候補を試す。

初期案:

```text
90% exploitation: 現在のbest router選択
10% exploration: 別候補または別graphを試す
```

制約:

- Security high riskではexplorationしない
- 高コスト候補のexploration率には上限を置く
- ユーザー影響が大きい業務ではshadow evaluationを優先する

### 10.3 Online Metrics

- cumulative regret
- online regret
- exploration cost
- quality/cost improvement over time
- drift detection
- new model adoption speed

## 11. Acceptance Criteria for Verification MVP

検証基盤MVPは、以下を満たしたら完了とする。

- 全Query×全候補モデル/グラフの評価テーブルを作成できる
- Random / Best Single / Rule-based / FT Router / Graph Selector / Oracleを同一データで比較できる
- Quality / Routing Accuracy / Mean Regret / Cost / Latencyを出力できる
- Quality-Cost Pareto chart用のCSVを出力できる
- Routerごとの失敗分類を出力できる
- 既存PoCの6カテゴリ分類精度と、新しいoutcome評価を別指標として管理できる
- Graph Orchestratorについて、unnecessary multi-agent rate と missed collaboration rate を出せる

## 12. Initial Experiment Plan

### Experiment 1: Current FT Router Outcome Evaluation

目的:

- 現行Qwen LoRA routerが、分類精度だけでなくoutcome上どれだけ有効か確認する

比較:

- Best Single
- Always Large
- Rule-based
- Current FT Router
- Oracle

成功条件:

- Best Single比でQuality同等以上、またはQuality低下が `0.02` 以下
- Costを `20%` 以上削減
- Mean Regret `0.05` 以下

### Experiment 2: Boundary Cases

目的:

- `Storage` / `Database`、`Coding` / `Database` 境界でtop-2 routingが有効か確認する

比較:

- top-1 single specialist
- top-2 parallel experts
- specialist with verifier
- Oracle

成功条件:

- top-2 parallel expertsがtop-1 single specialistよりQualityで `+0.05` 以上改善
- Cost増加に対してPareto上妥当

### Experiment 3: Graph Selector MVP

目的:

- Deterministic graph selectorが、単一ルーティングより良いQuality-Cost trade-offを作れるか確認する

比較:

- Current FT Router
- Graph Selector v0.1
- Always Large
- Oracle

成功条件:

- Current FT Router比でmulti-domain setのQuality `+0.08` 以上
- unnecessary multi-agent rate `20%` 未満
- Mean RegretがCurrent FT Routerより低い

## 13. Deliverables

- `eval/candidate-results.jsonl`
- `eval/router-predictions.jsonl`
- `eval/router-comparison.csv`
- `eval/pareto-quality-cost.csv`
- `eval/pareto-quality-latency.csv`
- `eval/failure-analysis.md`
- `eval/verification-report.md`

## 14. Open Questions

- 初期candidate modelは実モデルを使うか、まずは既存PoCのtarget model名に対するmock回答で評価基盤だけ作るか。
- Quality評価は人手rubricを主にするか、LLM judgeを併用するか。
- Oracle決定時にQualityとCostの重みをどう置くか。
- Graph候補をOracle対象に含めるタイミングをいつにするか。
- Online explorationを本番前にshadow modeでどこまで再現できるか。
