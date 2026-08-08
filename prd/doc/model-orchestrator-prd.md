# TuneScope Model Orchestrator PRD

## 1. Summary

TuneScope Model Orchestrator は、現行PoCの6カテゴリ分類ルータを拡張し、ユーザーリクエストごとに最適なモデル/エージェント構成を動的に組み立てるモデルオーケストレーション基盤である。

現行PoCは `Qwen/Qwen2.5-0.5B` + LoRA により、入力を `Storage`、`Network`、`Coding`、`Security`、`Database`、`General` に分類し、固定された専門モデルへ振り分ける。評価では OSS-only データで accuracy `0.873`、macro F1 `0.877` を達成している。一方で、現実の問い合わせは単一カテゴリだけでは閉じない。たとえば「Kubernetes上のPostgreSQLが遅い」は Storage / Network / Database / Coding の境界をまたぎ、単発分類よりも、複数専門家の協調、検証、再試行、統合が重要になる。

本PRDでは、Sakana Fugu の「学習済みオーケストレータLLMが、リクエストごとにモデル選択・委譲・検証・統合を行う」考え方を参考に、TuneScopeを以下の方向へ拡張する。

- Flat routing から adaptive orchestration へ移行する
- ルートを単一ラベルではなく、実行グラフとして表現する
- 実行を一回のモデル呼び出しではなく、計画、委譲、検証、修復、統合のループとして扱う
- deterministic selectorを安全なbaseline/fallbackとして維持しながら、実行outcomeとレビュー済みtraceでオーケストレータLLMをFTする

## 2. Background

### 2.1 Current PoC

現行PoCの主な成果は以下。

- OSS由来データのみで6ラベル分類データを構築
- 各カテゴリ800件、合計4,800件
- train/dev/test = 3,323 / 716 / 761
- Qwen2.5-0.5B sequence classification + LoRA fine-tuning
- FTあり: accuracy `0.873`、macro F1 `0.877`
- FTなし: accuracy `0.239`、macro F1 `0.163`
- 主な誤分類は `Storage` / `Database`、`Coding` / `Database` 境界に集中

このPoCは「小型ルータで専門モデルへ安く速く振り分ける」仮説を検証できている。ただし、出力は単一ラベルであり、複数カテゴリや段階的解決には対応していない。

### 2.2 Inspiration

Sakana Fugu は、単一APIの背後で複数モデルを動的に組み合わせるマルチエージェントシステムである。公開情報では、Fuguは手作業で固定したワークフローではなく、タスクごとにエージェントを assemble / route / coordinate する学習済みオーケストレータとして説明されている。

Claude Code / Codex は、実装形態としてはFuguほどの内部 learned orchestration ではないが、実務上重要な示唆がある。

- Claude Code: 計画は高性能モデル、実行はコスト効率のよいモデルに切り替える `opusplan` 型の分担
- Codex: モデルがツール呼び出しを行い、結果を受けて再度推論する agent loop
- Codex / OpenAI API: reasoning effort、prompt caching、context compaction など、長いループを現実的なコストで回すための設計

TuneScopeでは、これらを「グラフエンジニアリング」と「ループエンジニアリング」に分解して取り込む。

## 3. Product Goals

### 3.1 Goals

1. ユーザーリクエストに対して、単一モデルではなく最適な実行グラフを選べるようにする。
2. 現行PoCの分類器を、オーケストレーションの初期判断シグナルとして再利用する。
3. Storage / Network / Coding / Security / Database / General の専門性を維持しつつ、複合タスクに対応する。
4. ルーティング精度だけでなく、最終回答品質、コスト、レイテンシ、検証成功率を評価対象にする。
5. 全候補outcomeとレビュー済み実行ログから、学習済みオーケストレータを継続的にFTできるデータ基盤を作る。
6. 学習済みモデルに、graph選択だけでなくモデル選択、委譲目的、検証要否、統合方針を構造化planとして生成させる。

### 3.2 Non-Goals

- 初期FTリリースでは、未登録モデルや任意グラフを自己生成する完全なFugu型orchestrationは実装しない。
- 専門モデル本体のSFTを主目的にしない。
- すべての問い合わせで複数モデルを呼ばない。
- ユーザーに内部グラフの複雑さを露出しない。
- セキュリティ境界を超える自律実行、破壊的操作、外部システム操作は対象外とする。

## 4. Target Users

- インフラ、SRE、DBA、セキュリティ、開発支援を1つのAI窓口に集約したいチーム
- 問い合わせの内容に応じて専門モデルを使い分けたい運用者
- コストやレイテンシを抑えつつ、難しい問い合わせでは複数モデルの検証を使いたい管理者
- ルーティング品質を継続改善したいML/Platformチーム

## 5. Core Concepts

### 5.1 Orchestrator

Orchestrator は、ユーザーリクエスト、分類器スコア、履歴、予算、ポリシーを受け取り、実行グラフを選択するコンポーネントである。

Stage Aでは、以下の合成判断を行うdeterministic selectorを実装する。

- 現行Qwenルータの label probabilities
- ルールベースの task feature 抽出
- モデルごとの capability / cost / latency metadata
- confidence threshold
- risk policy
- user / workspace policy

Stage Bでは、全Query x 全候補のoutcomeとレビュー済みtraceから、軽量なorchestrator LLMをLoRA FTする。モデルはユーザー要求、分類スコア、リスク、予算、利用可能モデルを入力し、以下をJSON planとして生成する。

- graph id
- primary / selected labels
- specialist model assignments
- specialistごとのdelegated objective
- verification requirement
- synthesis strategy

policy gateは学習モデルの前段に置き、出力schema違反、未許可model/graph、timeout時はStage Aへfallbackする。

### 5.2 Graph Engineering

Graph Engineering は、問い合わせ解決をノードとエッジで設計する考え方である。

ノード例:

- `intent_classifier`: 現行PoCの分類器
- `planner`: タスク分解
- `specialist_storage`
- `specialist_network`
- `specialist_coding`
- `specialist_security`
- `specialist_database`
- `general_answerer`
- `critic`
- `verifier`
- `synthesizer`
- `fallback`

エッジ例:

- classify -> single specialist
- classify -> planner -> specialist team
- specialist -> verifier -> synthesize
- verifier failed -> repair loop
- low confidence -> clarify question
- high risk -> security review or human handoff

代表グラフ:

| Graph | 用途 | 構成 |
| --- | --- | --- |
| `single_specialist` | 高信頼・単一カテゴリ | classifier -> specialist -> answer |
| `specialist_with_verifier` | 高影響・検証必要 | classifier -> specialist -> verifier -> answer |
| `parallel_experts` | 複合カテゴリ | classifier -> planner -> N specialists -> synthesizer |
| `plan_execute_review` | 実装/調査タスク | planner -> executor -> reviewer -> repair -> final |
| `clarify_first` | 低信頼・情報不足 | classifier -> clarifier |
| `safe_refusal_or_handoff` | 高リスク/禁止領域 | policy -> safe response / human handoff |

### 5.3 Loop Engineering

Loop Engineering は、各グラフの中で反復処理を明示的に管理する考え方である。

主要ループ:

- Planning loop: 問い合わせを解ける単位へ分解する
- Delegation loop: 各サブタスクを適切な専門家へ割り当てる
- Verification loop: 回答、根拠、手順、リスクを検証する
- Repair loop: 検証失敗時に修正依頼を出す
- Synthesis loop: 複数回答を統合し、矛盾を解消する
- Learning loop: 実行結果、ユーザー評価、失敗理由を蓄積して改善する

各ループには停止条件を持たせる。

- 最大ステップ数
- 最大トークン/コスト
- 最大レイテンシ
- verifier pass
- confidence convergence
- human handoff
- policy stop

## 6. User Experience

### 6.1 Default Experience

ユーザーは内部ルーティングを意識せず、通常のチャット/ API として利用する。

入力:

```text
Kubernetes上のPostgreSQLが急に遅くなった。PVCはNFSで、Pod再起動後も改善しない。どこから切り分けるべき？
```

内部:

1. ルータが `Database`, `Storage`, `Network` を高スコアと判定
2. `parallel_experts` グラフを選択
3. DB specialist が query / lock / checkpoint を確認
4. Storage specialist が NFS latency / mount option / IOPS を確認
5. Network specialist が MTU / packet loss / DNS / CNI を確認
6. Synthesizer が優先順位つき切り分け手順を統合

出力:

- 最初に確認すべき順序
- 可能性の高い原因
- コマンド例
- 危険な操作の注意
- 追加情報が必要な項目

### 6.2 Debuggability Experience

管理者/開発者は、必要に応じて内部トレースを確認できる。

- selected graph
- router scores
- selected agents
- loop count
- verifier result
- cost / latency
- fallback reason

ユーザー向け通常UIでは、内部モデル名や細かいスコアは初期状態では出さない。

## 7. Functional Requirements

### 7.1 Routing

- FR-1: 入力テキストに対して、現行PoC分類器の `scores` を取得できる。
- FR-2: `scores`、閾値、ポリシーから `primary_labels` と `secondary_labels` を算出できる。
- FR-3: 高信頼の単一カテゴリは `single_specialist` を選択できる。
- FR-4: 複数カテゴリが近接する場合は `parallel_experts` または `specialist_with_verifier` を選択できる。
- FR-5: 低信頼または情報不足の場合は `clarify_first` を選択できる。
- FR-6: 高リスクなSecurity系依頼は policy gate を通過させる。

### 7.2 Graph Execution

- FR-7: 実行グラフをJSON/YAMLで定義できる。
- FR-8: ノードごとに model, prompt, input mapping, output schema, budget を設定できる。
- FR-9: ノード間の依存関係をDAGとして実行できる。
- FR-10: 並列可能な専門家ノードを並列実行できる。
- FR-11: 各ノードの出力をトレースとして保存できる。
- FR-12: 失敗ノードは fallback または retry policy に従って処理できる。

### 7.3 Loop Control

- FR-13: グラフごとに最大ステップ数を設定できる。
- FR-14: verifier failed の場合、指定回数まで repair loop を実行できる。
- FR-15: 予算超過時は degraded answer または clarification に切り替えられる。
- FR-16: loop stop reason を記録できる。

### 7.4 Answer Synthesis

- FR-17: 複数専門家の回答を統合し、重複と矛盾を整理できる。
- FR-18: 矛盾が残る場合は、確度と追加確認事項を明示できる。
- FR-19: 最終回答はユーザーの意図に合わせ、手順、原因候補、根拠、リスクを含められる。

### 7.5 Evaluation

- FR-20: 既存の分類評価に加え、end-to-end評価セットを定義できる。
- FR-21: 実行グラフ単位で accuracy / quality / cost / latency を比較できる。
- FR-22: ルーティング失敗、専門家失敗、検証失敗、統合失敗を分類して記録できる。
- FR-23: ユーザー評価や人手レビュー結果を学習データ化できる。

## 8. Non-Functional Requirements

### 8.1 Performance

- P95 latency: `single_specialist` は5秒以内を目標
- P95 latency: `parallel_experts` は20秒以内を目標
- Cost budget: リクエストごとに上限を設定可能
- Cache: 同一prefix、同一graph、同一tool定義では prompt caching を阻害しない設計にする

### 8.2 Reliability

- 外部モデル障害時は fallback chain を使用する
- ノード単位のtimeoutを持つ
- 実行途中の部分結果を保持し、失敗時に診断可能にする
- graph definition は versioned artifact として管理する

### 8.3 Safety

- Securityカテゴリでは offensive / destructive / credential exfiltration などを policy gate で制御する
- 高リスク操作は human handoff または安全な説明に限定する
- モデル/プロバイダ除外ポリシーをサポートする
- トレース保存時は秘密情報をマスクする

### 8.4 Observability

- request_id
- graph_id / graph_version
- router scores
- selected nodes
- model id
- latency
- token/cost
- loop count
- verifier status
- stop reason
- user feedback

## 9. Data Model

### 9.1 Orchestration Trace

```json
{
  "trace_id": "orch-20260809-000001",
  "input": {
    "text": "Kubernetes上のPostgreSQLが急に遅くなった..."
  },
  "router": {
    "model": "qwen-router-lora",
    "scores": {
      "Database": 0.44,
      "Storage": 0.31,
      "Network": 0.18,
      "Coding": 0.04,
      "Security": 0.02,
      "General": 0.01
    },
    "primary_labels": ["Database"],
    "secondary_labels": ["Storage", "Network"]
  },
  "graph": {
    "id": "parallel_experts",
    "version": "0.1.0",
    "stop_reason": "verifier_passed"
  },
  "nodes": [
    {
      "id": "database_specialist",
      "model": "database-specialist",
      "status": "completed",
      "latency_ms": 4200
    },
    {
      "id": "storage_specialist",
      "model": "storage-specialist",
      "status": "completed",
      "latency_ms": 3900
    },
    {
      "id": "synthesizer",
      "model": "general-synthesizer",
      "status": "completed",
      "latency_ms": 2600
    }
  ],
  "evaluation": {
    "user_rating": null,
    "review_label": null,
    "failure_type": null
  }
}
```

### 9.2 Graph Definition

```json
{
  "id": "specialist_with_verifier",
  "version": "0.1.0",
  "entrypoint": "specialist",
  "nodes": {
    "specialist": {
      "role": "domain_specialist",
      "model_selector": "primary_label",
      "max_tokens": 2048
    },
    "verifier": {
      "role": "critic",
      "model": "verifier",
      "max_tokens": 1024
    },
    "repair": {
      "role": "repair",
      "model_selector": "primary_label",
      "max_iterations": 1
    }
  },
  "edges": [
    ["specialist", "verifier"],
    ["verifier:fail", "repair"],
    ["repair", "verifier"],
    ["verifier:pass", "final"]
  ]
}
```

## 10. MVP Scope

### 10.1 Deterministic Foundation

最初にdeterministic orchestratorを作り、学習済みモデルのbaseline、fallback、教師データ生成元として維持する。

含める:

- 現行Qwenルータの推論APIを routing signal として使用
- router scores から graph を選択する graph selector
- 3種類の実行グラフ
  - `single_specialist`
  - `specialist_with_verifier`
  - `parallel_experts`
- graph execution trace
- 手動評価用レポート
- 既存test.jsonに加えた複合タスク評価セット

この段階では含めない:

- 本番モデルプールとの完全統合
- UI
- 課金/レート制御の完全実装

### 10.2 MVP Graph Selection Rules

```text
if max_score < 0.45:
  graph = clarify_first
elif top1_score - top2_score >= 0.25 and risk != high:
  graph = single_specialist
elif risk == high:
  graph = specialist_with_verifier
else:
  graph = parallel_experts
```

カテゴリ境界が既知の `Storage` / `Database`、`Coding` / `Database` では、top2との差が小さい場合に `parallel_experts` を優先する。

### 10.3 Learned Orchestrator FT

FT対象はsequence classifierではなく、構造化orchestration planを生成するcausal LMとする。

入力:

- user request
- 現行6分類器のlabel probabilities
- risk level
- max cost / latency / steps
- available graph ids
- available model aliases

出力:

```json
{
  "graph_id": "parallel_experts",
  "primary_labels": ["Database"],
  "selected_labels": ["Database", "Storage"],
  "confidence": 0.88,
  "risk_level": "normal",
  "reason": "database and storage evidence are both required",
  "delegations": [
    {
      "label": "Database",
      "model": "database-specialist",
      "objective": "Check locks, waits, checkpoints, and query plans."
    },
    {
      "label": "Storage",
      "model": "storage-specialist",
      "objective": "Check NFS latency, saturation, and mount options."
    }
  ],
  "synthesis_strategy": "Reconcile evidence and order checks by diagnostic value."
}
```

初期base modelは `Qwen/Qwen2.5-1.5B-Instruct`、学習方式はLoRA causal LM SFTとする。全候補outcomeからquality差 `0.02` 以内を同等品質帯とし、その中でcost/latencyが小さい候補を教師planにする。同時にchosen/rejected planを生成し、次段階のDPOまたはpreference optimizationに備える。

高リスクpolicyはFTモデルへ委譲しない。schema validation、model allowlist、graph allowlistを通過したplanだけを実行し、違反時はdeterministic selectorへfallbackする。

## 11. Evaluation Plan

### 11.1 Metrics

分類単体:

- accuracy
- macro F1
- per-label precision / recall
- top-2 accuracy
- calibration error

オーケストレーション:

- final answer quality
- task success rate
- verifier pass rate
- repair success rate
- unnecessary multi-agent rate
- cost per successful answer
- latency P50/P95
- user rating
- human preference win rate

学習済みオーケストレータ:

- valid plan rate
- graph accuracy
- selected labels exact match / F1
- delegation model validity
- deterministic fallback rate
- Oracle regret
- Best Single / deterministic selectorに対するquality-cost Pareto改善

### 11.2 Evaluation Sets

既存:

- `poc/artifacts/test.json`
- `prd/artifacts/train.json`
- `prd/artifacts/dev.json`
- `prd/artifacts/test.json`

追加:

- `orchestration-single-domain.json`: 単一専門領域
- `orchestration-boundary.json`: Storage/Database、Coding/Databaseなど境界ケース
- `orchestration-multi-domain.json`: 複数専門家が必要なケース
- `orchestration-risky-security.json`: 安全制御が必要なケース
- `orchestration-clarification.json`: 情報不足ケース

### 11.3 Success Criteria

MVP acceptance:

- 既存分類器の単体macro F1が `0.87` 以上を維持
- boundary set で top-2 accuracy `0.92` 以上
- multi-domain set で human preference が flat single-specialist baseline に対して `+15%` 以上
- unnecessary multi-agent rate が `20%` 未満
- trace completeness `99%` 以上
- policy gate bypass 重大事故 `0`

Learned FT acceptance:

- valid plan rate `99%` 以上
- policy/allowlist違反planの実行 `0`
- deterministic selector比でmean regretを悪化させない
- multi-domain setでdeterministic selector比quality `+0.05` 以上、または同等qualityでcost `15%` 以上削減
- fallback rate `5%` 未満

## 12. Rollout Plan

### Phase 0: PRD and Design

- 本PRDを確定
- graph schema と trace schema をレビュー
- MVP評価セットのラベル方針を決める

### Phase 1: Deterministic Graph Selector

- 現行OpenAI-compatible router serverをrouting signalとして使用
- graph selectorを実装
- trace保存を追加
- CLIで単発実行できるようにする

### Phase 2: Graph Executor

- `single_specialist`
- `specialist_with_verifier`
- `parallel_experts`
- timeout / retry / fallback
- evaluation report

### Phase 3: Loop Optimization

- verifier rubricを導入
- repair loopを導入
- cost / latency budgetを導入
- prompt cachingを阻害しない入力順序を整理

### Phase 4: Learning Loop

- 全Query x 全候補outcomeからOracle planを生成
- レビュー済み成功traceをSFT exampleへ変換
- chosen/rejected planをpreference datasetへ変換
- Qwen2.5-1.5B-InstructをLoRA SFT
- valid plan / graph accuracy / outcome regretをoffline評価
- deterministic selectorをfallbackにしたshadow実行
- safety gate通過後にA/B比較

### Phase 4.5: Fugu-like Expansion

- model capability / cost / latency catalogを入力へ追加
- 固定graph選択から、制約付きnode compositionへ拡張
- verifier結果を使ったtrajectory-level preference学習
- explorationをshadow modeで行い、新しいモデル/graphのoutcomeを収集
- SFT -> preference optimization -> bounded online learningの順で改善

### Phase 5: Production Readiness

- モデル/プロバイダ除外ポリシー
- セキュリティ監査
- 管理者向け observability
- SLA / cost guardrail
- UI/SDK integration

## 13. Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| 複数モデル呼び出しでコストが増える | 高 | graph selectorで単一専門家を優先し、budget stopを入れる |
| ループが止まらない | 高 | max steps, max cost, max latency, verifier convergenceを必須化 |
| 複数専門家の回答が矛盾する | 中 | synthesizerに矛盾検出と未確定事項の明示を要求 |
| Storage/Database境界が改善しない | 中 | top-2 routingとboundary evalを導入し、単一ラベル精度だけで判断しない |
| Securityで危険な支援をしてしまう | 高 | policy gate、human handoff、trace reviewを導入 |
| learned orchestratorの学習データが偏る | 中 | traceに失敗理由と人手評価を保存し、offline evalを先に整備 |
| プロンプト/ツール変更でキャッシュ効率が落ちる | 中 | graph definitionを固定化し、可変情報を後段に置く |

## 14. Open Questions

- MVPの専門モデルは実モデルを呼ぶか、まずはmock/stubでgraph qualityを検証するか。
- verifierは汎用モデルにするか、カテゴリ別verifierを持つか。
- human preference評価は誰が、どのrubricで付けるか。
- 日本語入力では日本語データセットを主評価にするか、英日混在で評価するか。
- preference optimizationをDPOから始めるか、trajectory-level reward modelを先に作るか。

## 15. References

- Sakana Fugu product page: https://sakana.ai/fugu/
- Sakana Fugu Technical Report: https://arxiv.org/abs/2606.21228
- Sakana Fugu API get started: https://console.sakana.ai/get-started
- Claude Code models, usage, and limits: https://support.claude.com/en/articles/14552983-models-usage-and-limits-in-claude-code
- Claude Code CLI reference: https://code.claude.com/docs/en/cli-usage
- OpenAI, Unrolling the Codex agent loop: https://openai.com/index/unrolling-the-codex-agent-loop/
- OpenAI model guidance: https://developers.openai.com/api/docs/guides/latest-model
