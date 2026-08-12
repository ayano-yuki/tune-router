# PRD Production TODO

この文書は、現在のPRDをMVPから業務運用可能な水準へ引き上げるための実装バックログである。
前回の問題点 `1. router推論service本体がPRDにない` は、今回の依頼どおり対象外とする。

## Scope and rules

- 実装コードは `prd/src` 直下へ置く。`prd/src/tune_orchestrator` は作らない。
- testはすべて `prd/src/test` へ置く。`prd/tests` は再作成しない。
- 本文のsource module pathは、最初に実施するPRD-011完了後の `tune_*` 名で記載する。test file名は既存の `test_*.py` を維持する。
- `poc` のmodule、artifact、設定、相対pathをPRD runtimeから参照しない。
- JSON artifactには必ず `format`、`created_at`、入力artifactのSHA-256 digest、生成設定を含める。
- CLIのgate/verify系commandは、成功時exit code `0`、失敗時exit code `1` とする。調査用途に限り `--no-fail` でexit code `0`を許可する。
- runtimeで検証不能なartifactはfail closedとし、暗黙のfallbackで使用を継続しない。
- 既存のdeterministic selectorは、learned plan生成失敗時の安全なfallbackとして維持する。
- 各TODOは、その項目の「受け入れ条件」と「必須test」をすべて満たした時だけ完了扱いにする。

共通の完了確認command:

```powershell
uv sync --project .\prd --frozen
uv run --project .\prd python -m unittest discover -s .\prd\src\test -v
uv run --project .\prd tune-orchestrator validate-graphs
uv run --project .\prd tune-orchestrator --help
```

## Priority and implementation order

| Order | ID | Priority | Depends on | Outcome |
|---:|---|---|---|---|
| 1 | PRD-011 | P0 | - | flat module名の衝突を先に解消する |
| 2 | PRD-008 | P0 | PRD-011 | artifactのatomic write、lock、署名基盤を作る |
| 3 | PRD-015 | P0 | PRD-008 | endpoint、secret、training、traceをhardeningする |
| 4 | PRD-002 | P0 | PRD-011 | 明示的なrouter正解labelを継続学習へ入れる |
| 5 | PRD-006 | P0 | PRD-002 | 学習dataのschema、leak、偏り、機密情報をgateする |
| 6 | PRD-007 | P0 | PRD-006 | router scoreを校正し、選択閾値を固定する |
| 7 | PRD-003 | P0 | PRD-006, PRD-007 | router候補をbaseline比較で昇格判定する |
| 8 | PRD-014 | P0 | PRD-003, PRD-008 | 全componentを束ねたrelease/rollbackを実装する |
| 9 | PRD-004 | P0 | PRD-008, PRD-014 | 検証済みruntime releaseを実行時に強制する |
| 10 | PRD-012 | P1 | PRD-004, PRD-014, PRD-015 | `doctor`を本番起動の単一gateにする |
| 11 | PRD-005 | P1 | PRD-006, PRD-014 | 制約付き自由agent assemblyを完成させる |
| 12 | PRD-013 | P1 | PRD-004, PRD-014 | drift、alert、dashboardを追加する |
| 13 | PRD-009 | P0 | 上記各機能と並行 | 実HTTPを含むE2E回帰を固定する |
| 14 | PRD-010 | P0 | PRD-009 | CIとtraining smokeを必須化する |

P0はproduction traffic投入前に必須、P1は限定canaryを100% trafficへ進める前に必須とする。

## PRD-002 Explicit router supervision

- [ ] reviewed traceの自己選択labelではなく、明示的な正解labelを継続学習の第一教師信号にする。

### Data contract

traceの `evaluation.router_feedback` を次のschemaで固定する。

```json
{
  "schema_version": 1,
  "disposition": "approved|corrected|rejected|ambiguous",
  "gold_labels": ["Storage", "Database"],
  "reviewer": "operator-id",
  "reviewed_at": "2026-08-09T05:00:00Z",
  "reason": "NFSとPostgreSQLの複合問題",
  "source": "human|oracle|incident_review"
}
```

適用規則:

1. `corrected` は `gold_labels`、`reviewer`、`reviewed_at`、`reason` を必須とする。
2. `approved` で `gold_labels` が省略された場合だけ、served decisionの `graph.selected_labels` を正解とする。
3. `rejected` は `gold_labels` がある場合だけ学習へ入れる。正解不明のrejectはdata quality reportへ除外理由を記録する。
4. `ambiguous` は学習へ入れず、active-learning queueへ送る。
5. 明示feedbackがある場合、`user_rating` や既存router選択labelで上書きしない。
6. 明示feedbackがない場合のみ、`review_label in {preferred, approved, success}` または `user_rating >= min_rating` を弱い教師信号として許可する。
7. `include_failed_corrections=true` の場合、実行が失敗していても `corrected` または正解付き `rejected` を採用する。このflagを実際の分岐に使用する。
8. 現行classifierはtop-1分類なので、`gold_label=gold_labels[0]` とする。2件目以降は `metadata.secondary_gold_labels` に保持し、同一textを複数の単一label recordへ複製しない。

source別weightは `human corrected=2.0`、`incident_review=2.0`、`oracle=1.75`、`explicit approved=1.5`、`rating fallback=1.0` とする。fallback発生traceはさらに `0.7` を乗算する。

### Implementation

- `prd/src/tune_router_learning.py`
  - `RouterFeedback` dataclassと `validate_router_feedback()` を追加する。
  - `_examples_from_trace()` のlabel決定順を上記規則へ変更する。
  - `include_failed_corrections` を実際に使用する。
  - record metadataへ `feedback_disposition`、`feedback_source`、`reviewer`、`source_trace_id`、`secondary_gold_labels` を保存する。
  - dedupe keyを `normalized_text + primary gold label` とし、同じtraceの再取込を冪等にする。
  - label衝突時は黙って高weightを選ばず、conflict一覧をmetadataへ出す。
- `prd/src/tune_cli.py`
  - `prepare-router-corrections --source <jsonl> --out <dir>` を追加する。
  - `prepare-router-continual` のsummaryに `explicit_examples`、`weak_examples`、`ambiguous`、`rejected_without_gold`、`conflicts` を出す。
- `prd/src/tune_evaluation.py`
  - `trace-report` にfeedback coverageとexplicit correction rateを追加する。
- `prd/doc/production-runbook.md`
  - feedback登録例、reviewer責任、学習対象/除外規則を追記する。

### Acceptance criteria

- explicit correctionがserved labelより常に優先される。
- 低ratingまたはnode failureでも、正解付きcorrectionは設定に従って採用できる。
- ambiguous/reject-without-goldはtrain splitへ混入しない。
- 同一textに相反するprimary labelがある場合、dataset生成結果が `status=fail` になりconflictを列挙する。
- dataset artifactから教師labelの根拠と元traceを追跡できる。

### Required tests

- `prd/src/test/test_router_learning.py`: precedence、failed correction、ambiguous除外、multi-label保持、conflict、冪等性。
- `prd/src/test/test_evaluation.py`: feedback coverage集計。
- fixtureにはPIIを含まないsynthetic traceを使う。

## PRD-003 Router evaluation and promotion gate

- [ ] candidate routerをbaselineと同一test setで比較し、基準未達artifactを本番へ昇格できないようにする。

### Artifact contracts

prediction JSONLの各record:

```json
{
  "query_id": "stable-id",
  "gold_label": "Storage",
  "scores": {"Storage": 0.8, "Network": 0.01, "Coding": 0.02, "Security": 0.01, "Database": 0.15, "General": 0.01},
  "predicted_label": "Storage",
  "model_digest": "sha256",
  "dataset_digest": "sha256",
  "latency_ms": 12.3
}
```

evaluation artifact formatは `tune-router-evaluation-v1` とし、最低限 `total`、`accuracy`、`macro_f1`、label別precision/recall/F1、confusion matrix、NLL、Brier score、ECE、General予測率、p50/p95 latencyを含める。

promotion artifact formatは `tune-router-promotion-v1` とし、`status`、candidate/baseline/model/dataset digest、全checkのactual/threshold/passed、生成時刻を含める。

### Implementation

- `prd/src/tune_router_evaluation.py` を追加する。
  - `evaluate_router_predictions()` を実装する。
  - ECEは15個のequal-width confidence binで計算する。
  - scoreは全labelが存在し、有限、各値が `[0,1]`、総和が `1 +/- 1e-6` であることを検証する。
  - candidateとbaselineのpaired bootstrap 95% CIをseed固定で計算する。
  - `evaluate_router_promotion()` を実装する。
- `prd/src/tune_router_learning.py`
  - local LoRA adapterからprediction JSONLを生成するinference関数を追加する。
  - adapter metadataのbase model revision、label順、training digestを検証する。
- `prd/src/tune_cli.py` に以下を追加する。
  - `predict-router --adapter --base-model --data --out`
  - `evaluate-router --predictions --data --out --report`
  - `gate-router --candidate --baseline --out --report`
- gate既定値:
  - `min_examples=100`
  - `min_macro_f1_delta=0.0`
  - `max_accuracy_regression=0.005`
  - `max_per_label_f1_regression=0.03`
  - `max_ece=0.10`
  - `max_ece_regression=0.01`
  - protected labelは `Security`、`max_protected_recall_regression=0.01`
- candidateとbaselineの `dataset_digest`、label集合、query_id集合が一致しなければ比較せずfailする。

### Acceptance criteria

- candidate単体の良いmetricだけではpassせず、baselineとの差分も全checkを通る必要がある。
- label別退行、校正悪化、評価data不一致のいずれかでexit code `1`になる。
- `--no-fail` はartifactの `status=fail` を変えず、process exit codeだけ `0`にする。
- promotion artifactがPRD-014のrelease manifestへdigestでbindされる。

### Required tests

- `prd/src/test/test_router_evaluation.py`: perfect/poor prediction、score不正、dataset mismatch、各gate境界値、bootstrap再現性。
- `prd/src/test/test_cli.py`: gate pass/failとexit code。
- training dependencyなしでmetric/gate testが完結すること。

## PRD-004 Mandatory runtime artifact verification

- [ ] `run` と `run-ft` が、未検証または改変済みruntime artifactを使用できないようにする。

### Runtime behavior

- production modeは `--runtime-release <current.json>` を必須とする。
- `--bandit-state`、`--bandit-rollout-config`、`--bandit-release-manifest`、`--bandit-release-current` のいずれかを直接指定した場合も、対応する `--bandit-runtime-bundle` を必須とする。
- artifact検証はrouter/model endpointへの最初のHTTP requestより前に行う。
- digest mismatch、期限切れ、signature不正、path traversal、missing artifact、release status不正は処理を開始せずexit code `1`とする。
- local開発は `--runtime-mode development --mock` の組合せだけrelease省略を許可する。
- break-glassは `--unsafe-allow-unverified-artifacts --change-ticket <id>` の両方を必須にし、stderrとtraceへsecurity eventを残す。production既定値では無効にする。

### Implementation

- `prd/src/tune_cli.py`
  - `_load_and_verify_runtime_context()` を追加し、`cmd_run()` と `cmd_run_ft()` の先頭で呼ぶ。
  - 検証済みcontextからgraphs、model config、bandit state、policy、calibration pathを解決する。
  - release由来値と直接CLI引数が競合したらfailする。
  - traceへ `runtime_release_id`、manifest/current/bundle digest、verification statusを必ず保存する。
- `prd/src/tune_bandit.py`
  - runtime bundleの `paths` をbundle file基準の相対pathへ正規化する。
  - resolve後pathがrelease directory外へ出る `..`、symlink escape、絶対pathを拒否する。
- `prd/src/tune_executor.py`
  - verification済みcontextがないproduction executionを拒否するguardを追加する。

### Acceptance criteria

- artifactを1 byte変更した状態でmodel endpointへのrequestが0件のまま失敗する。
- expired releaseとsignature不正releaseがfail closedになる。
- development mockはreleaseなしで既存どおり動く。
- production traceだけを見て、使用した全artifact digestを再現できる。

### Required tests

- `prd/src/test/test_runtime_release.py`: valid、tampered、expired、missing、path traversal、引数競合、break-glass audit。
- `prd/src/test/test_executor.py`: production guard。
- fake HTTP serverのrequest counterで「検証失敗時に外部callなし」を確認する。

## PRD-005 Constrained free agent assembly

- [ ] 固定graph選択を超え、capabilityと制約からnode/edge/loopを組み立てるagent assemblyをproduction対応にする。

### Plan schema v2

`tune-orchestrator-plan-v2` を定義し、以下を必須にする。

- `plan_id`、`plan_type=bounded_graph`、`selected_labels`、`risk_level`、`final_node`。
- `nodes[]`: `id`、`role`、`objective`、`model`、`dependencies`、`input_keys`、`output_key`、`max_tokens`、`timeout_seconds`。
- 任意の `fallback_models`、`output_schema`、`retry_policy`。
- 任意の `loop`: `candidate_node`、`verifier_node`、`repair_role`、`max_iterations`。
- `constraints`: `max_nodes`、`max_parallelism`、`max_steps`、`max_cost_usd`、`max_latency_ms`。
- plannerが参照した `capability_catalog_digest` と `policy_digest`。

### Validation rules

- node idは `^[a-z][a-z0-9_]{0,63}$`、重複不可。
- dependencyは既存nodeのみを参照し、DAGであること。loop edgeは通常dependencyと分離する。
- final nodeへ到達しないorphan nodeを拒否する。
- model aliasはcatalogに存在し、node label/domain、JSON support、context window、safety profileを満たすこと。
- high riskではfinal pathにverifierまたはpolicy nodeを必須とし、bandit/shadowによるbypassを禁止する。
- synthesizerは2入力以上、repairはcandidateとverifier feedbackを入力に持つこと。
- node/loopを展開したworst-case steps、cost、latencyがbudgetを超えるplanを実行前に拒否する。
- rejected planはdeterministic fallbackへ落とし、全validation errorをtraceへ保存する。

### Implementation

- `prd/src/tune_composition.py`
  - schema v2 normalize、DAG、reachability、capability、worst-case budget validatorを追加する。
  - canonical planと `plan_digest` を生成する。
- `prd/src/tune_learned.py`
  - planner promptへmodel capability、runtime health、budget、policyを構造化入力として渡す。
  - schema v1/v2 parserを分離し、v1は移行期間だけread-only対応する。
- `prd/src/tune_executor.py`
  - generated graphでもparallel node、verifier、bounded repair loop、fallback modelを同じ意味で実行する。
  - nodeごとのdeadlineを全体deadlineから切り出し、残budgetを次nodeへ伝播する。
- `prd/src/tune_ft_data.py`
  - schema v2のSFT exampleを生成する。
  - chosen/rejected trajectoryへplan validity、quality、cost、latency、safety、loop countを含める。
  - 同一query/capability catalog/budget内だけをpreference pairにする。
- `prd/src/tune_training.py`
  - `train-ft-preference` を追加し、SFT adapterからpreference optimizationを開始できるようにする。
  - training metadataへbase/SFT adapter digest、dataset digest、algorithm、seedを保存する。
- `prd/src/tune_shadow.py`
  - 未選択planを低riskかつbudget内だけで探索し、served/shadowを同一parent traceへbindする。

### Acceptance criteria

- 同一domainの複数modelからcapabilityに合うaliasを選択できる。
- 2 specialist並列、synthesis、verifier、repair loopを含むgenerated planを実行できる。
- cycle、orphan、unknown model、budget超過、high-risk bypassは実行前に拒否される。
- planner障害時にもdeterministic fallbackが動き、fallback理由がtraceに残る。
- Oracle replayで固定graph baselineとgenerated planを同じquery集合上で比較できる。

### Required tests

- `prd/src/test/test_composition.py`: schema v2の正常系と全validation rule。
- `prd/src/test/test_executor.py`: parallel、loop、fallback、deadline、budget exhaustion。
- `prd/src/test/test_ft_data.py`: trajectory pairの比較可能性とdigest。
- `prd/src/test/test_learned.py`: invalid plan fallbackとcatalog binding。

## PRD-006 Training data quality gates

- [ ] router/orchestratorの学習dataを、schema、重複、leak、偏り、provenance、機密情報の観点でrelease前に検査する。

### Implementation

- `prd/src/tune_data_quality.py` を追加する。
  - `validate_router_dataset()` と `validate_ft_dataset()` を実装する。
  - textはUnicode NFKC、trim、連続空白圧縮した値で重複/leak判定する。原文は変更しない。
  - `query_id`、normalized text、`source_trace_id` のtrain/dev/test横断を検出する。
  - 同一normalized textのlabel衝突をerrorにする。
  - record weightがfiniteかつ `(0, 10]` であることを確認する。
  - label、graph id、model alias、plan schema versionをallow-list検証する。
  - email、credential形式、private key header、Bearer token、主要cloud key patternを検知し、record idだけをreportする。値そのものはreportへ出さない。
  - source、reviewer、dataset digest、生成configのprovenance欠落を検出する。
- `prd/src/tune_cli.py` に以下を追加する。
  - `validate-router-data --dataset <dataset.json> --out <json> --report <md>`
  - `validate-ft-data --data <dir> --out <json> --report <md>`
- default gate:
  - schema error、split leak、label conflict、secret検出はfail。
  - trainの各label最小20件、dev/testの各label最小5件。閾値はCLIで変更可能。
  - 最大label imbalance ratioは10.0。超過はfail。
  - exact duplicateはwarningとして件数を出し、出力datasetでは1件へ畳み込む。
- dataset metadataへquality report digestをbindし、quality statusがpassでないdatasetをtraining commandが拒否する。

### Acceptance criteria

- train/test leak、conflicting label、secret混入のfixtureがすべてfailする。
- reportにrecord本文や検出secretが出力されない。
- `train-router` と `train-ft` はquality report未指定またはdigest不一致のproduction trainingを拒否する。
- 同じinputとseedから同じsplit/digestが生成される。

### Required tests

- `prd/src/test/test_data_quality.py`: schema、NFKC duplicate、split leak、conflict、class imbalance、secret redaction、digest binding。
- `prd/src/test/test_training.py`: quality gate未通過dataの拒否。

## PRD-007 Router calibration and threshold tuning

- [ ] router scoreをheld-out dev dataで校正し、confidence/margin thresholdをartifactとして管理する。

### Calibration contract

- methodはまずscalar temperature scalingを採用する。
- input probabilityは `epsilon=1e-12` でclipし、`log(p) / temperature` をsoftmaxして校正scoreを得る。
- temperatureはdev setのNLL最小化で決定し、test setではfitしない。
- calibration artifact formatは `tune-router-calibration-v1` とし、`temperature`、labels順、model digest、fit dataset digest、fit前後のNLL/Brier/ECE、optimizer設定を保存する。
- selection policy artifactには `minimum_confidence`、`single_specialist_margin`、`secondary_score`、`max_selected_labels` と、tuning dataset digestを保存する。

### Implementation

- `prd/src/tune_router_calibration.py` を追加する。
  - `fit_temperature()`、`apply_temperature()`、`expected_calibration_error()` を実装する。
  - scipyへ依存せず、log-temperature上のcoarse-to-fine deterministic searchをseed不要で実装する。
  - threshold grid searchはmacro F1を最大化し、同点ならmulti-agent率、次にconfidence thresholdの高い設定を選ぶ。
- `prd/src/tune_clients.py`
  - router responseのlabel集合、finite値、score範囲、総和を検証する。
  - 設定時だけcalibrationを適用し、raw/calibrated scoreの両方をtraceへ渡せるようにする。
- `prd/src/tune_cli.py` に以下を追加する。
  - `fit-router-calibration --predictions <dev.jsonl> --out <json>`
  - `tune-router-thresholds --predictions <dev.jsonl> --calibration <json> --out <json>`
  - `select`、`run`、`run-ft` に `--router-calibration` と `--selection-policy` を追加する。
- calibration/model/dataset digest不一致はfail closedにする。

### Acceptance criteria

- synthetic over-confident predictionでfit後NLLが悪化しない。
- raw scoreとcalibrated score、temperature、artifact digestがtraceに残る。
- calibrationなしでは既存挙動が変わらない。
- test dataをfit inputに指定した場合、dataset metadataのsplitから検知して拒否する。

### Required tests

- `prd/src/test/test_router_calibration.py`: normalization、NLL改善、digest mismatch、determinism、threshold tie-break。
- `prd/src/test/test_clients.py`: NaN、missing label、sum不正、calibration適用。
- `prd/src/test/test_selector.py`: calibrated threshold境界。

## PRD-008 Atomic artifact registry and signatures

- [x] JSON fileの単純上書きを廃止し、並行更新、途中書込、改ざんを検出できるartifact基盤へ統一する。

### Implementation

- `prd/src/tune_artifacts.py` を追加する。
  - canonical JSONはUTF-8、key sort、separator `(',', ':')`、NaN禁止で生成する。
  - `artifact_digest()` をこのmoduleへ集約する。
  - `atomic_write_json()` は同一directoryのtemporary fileへwrite、flush、`fsync`、`os.replace` の順で更新する。
  - `<target>.lock` のprocess間exclusive lockを取得し、default timeout 10秒、timeout時はfailする。
  - registryに単調増加 `revision` を持たせ、expected revisionを使うcompare-and-swap更新を実装する。
  - read時にformat、digest、revisionを検証する。
- signatureはEd25519 detached signatureを採用する。
  - `prd[security]` extraへ `cryptography` を追加する。
  - private keyはpathまたはsecret managerから読み、artifactへ埋め込まない。
  - signature artifactは `format=tune-artifact-signature-v1`、artifact digest、key id、algorithm、base64 signature、signed_atを持つ。
  - CLIへ `sign-artifact --artifact --private-key --key-id --out` と `verify-artifact --artifact --signature --public-key` を追加する。
- `tune_bandit.py`、`tune_router_learning.py`、`tune_ft_data.py`、`tune_training.py`、`tune_evaluation.py`、`tune_cli.py` の直接 `Path.write_text()` を共通writerへ移行する。
- release manifest、current pointer、runtime bundle、promotion artifactはproductionで署名必須にする。

### Acceptance criteria

- processを強制終了しても既存artifactがpartial JSONにならない。
- 2 processの同時registry更新でentry lossが発生せず、revisionが連続する。
- 署名後にartifactを1 byte変更するとverifyがfailする。
- private keyの内容、path、exception detailがtrace/logへ出ない。
- WindowsとLinuxの両方でlock testが通る。

### Required tests

- `prd/src/test/test_artifacts.py`: canonical digest、atomic replace、lock timeout、CAS conflict、concurrent append、sign/verify/tamper。
- `prd/src/test/test_bandit.py`: registry writer移行後の冪等性とrevision。

## PRD-009 End-to-end production tests

- [ ] unit testだけでは検出できないCLI、HTTP、artifact chain、fallbackの回帰をE2Eで固定する。

### Test harness

- `prd/src/test/e2e_support.py` に標準library `ThreadingHTTPServer` を使ったfake OpenAI-compatible serverを実装する。
- endpointは `/v1/models` と `/v1/chat/completions` を提供し、router、specialist、verifier、orchestrator planをfixtureごとに切り替える。
- request count、headers、payload、response delay、HTTP error、malformed JSON、connection closeを注入できるようにする。
- test artifactは `tempfile.TemporaryDirectory()` 配下へ作り、repositoryの `artifacts/runtime` を汚さない。

### Required E2E scenarios

- [ ] deterministic `select` からsingle specialist実行まで成功する。
- [ ] multi-domain requestでparallel expertsとsynthesisが実行される。
- [ ] learned bounded graph v2が実行される。
- [ ] learned plan不正時にdeterministic fallbackし、理由がtraceに残る。
- [ ] verifier fail後にrepairし、再verifyで成功する。
- [ ] specialist接続拒否時にfallback modelへ切り替わる。
- [ ] timeout、max steps、max costがそれぞれ停止理由へ反映される。
- [ ] valid runtime releaseで実行でき、改変artifactではHTTP call前に失敗する。
- [ ] router calibrationとselection policyがruntime decisionへ反映される。
- [ ] bandit 0/10/100% rolloutがstable hashどおりに切り替わる。
- [ ] shadow executionがserved answerを変更せずparent traceへ記録される。
- [ ] unified rollback後に直前releaseのdigestで実行される。
- [ ] secretを含むupstream errorがstdout、stderr、traceでredactされる。

### Files and acceptance criteria

- scenario testは `prd/src/test/test_e2e_*.py` に分割する。
- network、GPU、外部model downloadなしで5分以内に完了する。
- testはCLI entry point相当のargument parsingを通し、内部関数だけの直呼びで代替しない。
- critical pathのbranch coverageを85%以上にする。coverage対象は `clients`、`composition`、`executor`、`artifacts`、`release`、`router_evaluation`。

## PRD-010 CI and training verification

- [ ] clean environmentでbuild、test、quality gate、artifact検証、training smokeが再現されるCIを作る。

### Mandatory CI

- `.github/workflows/prd-ci.yml` を追加し、WindowsとLinux、Python 3.11/3.12 matrixで実行する。
- command順:
  1. `uv sync --project ./prd --frozen --extra dev`
  2. `ruff check prd/src prd/build_backend.py`
  3. `python -m unittest discover -s ./prd/src/test -v`
  4. `tune-orchestrator validate-graphs`
  5. fixtureに対するdata quality、router gate、release verify。
  6. wheel build、temporary venvへのinstall、`tune-orchestrator --help` smoke。
- `prd[dev]` extraへversionを固定した `ruff`、`coverage` を追加する。
- CIは `poc` や `sever` のinstall済みpackageに依存しないclean venvで行う。

### Training verification

- mandatory CIではtraining dependency import、tokenization、weighted record、metadata生成をfake model/tokenizerでtestする。
- `.github/workflows/prd-training-smoke.yml` を手動起動と週次scheduleで追加する。
- training smokeは承認済みの小型model revisionを固定し、1 epoch/数recordでrouter LoRAとorchestrator LoRAを作成、load、1 inference、evaluationまで行う。
- model download cache keyにmodel idとrevisionを含める。
- adapter、training log、evaluation、quality reportをCI artifactとして保持する。保持期間は14日とする。
- GPU runnerがない場合は黙ってskipせずjobをblocked/failとして可視化する。

### Acceptance criteria

- PRに対するmandatory CIがbranch protectionのrequired checkになっている。
- lockfile差分があるのに `uv.lock` 未更新ならCIがfailする。
- buildしたwheel単体でCLIとpackaged graphが動く。
- actual training smokeの最後に生成adapterを再loadできる。

## PRD-011 Collision-safe flat source modules

- [x] `prd/src` 直下という制約を維持しながら、`models`、`cli`、`training` 等の一般名によるimport衝突を解消する。

### Rename map

subdirectory packageは作らず、以下へ一括renameする。

| Current | New |
|---|---|
| `bandit.py` | `tune_bandit.py` |
| `cli.py` | `tune_cli.py` |
| `clients.py` | `tune_clients.py` |
| `composition.py` | `tune_composition.py` |
| `constants.py` | `tune_constants.py` |
| `evaluation.py` | `tune_evaluation.py` |
| `executor.py` | `tune_executor.py` |
| `ft_data.py` | `tune_ft_data.py` |
| `graphs.py` | `tune_graphs.py` |
| `learned.py` | `tune_learned.py` |
| `models.py` | `tune_models.py` |
| `ops.py` | `tune_ops.py` |
| `router_learning.py` | `tune_router_learning.py` |
| `selector.py` | `tune_selector.py` |
| `shadow.py` | `tune_shadow.py` |
| `training.py` | `tune_training.py` |

### Implementation

- 全source/test importを `tune_*` 名へ更新する。相対importとbare generic module importを混在させない。
- `pyproject.toml` のentry pointを `tune_cli:main` へ変更する。
- `build_backend.py` のentry pointとwheel module一覧を更新する。
- `prd/src/__init__.py` はflat module構成ではpackage rootにならないため削除する。
- 旧module名のcompatibility shimは残さない。CLIだけをpublic interfaceとし、Python APIはinternalであることをREADMEへ明記する。
- wheel内に `tune_*.py` とgraph definitionsだけが入り、test moduleは入らないことを検証する。

### Acceptance criteria

- `uv run --project ./prd python -c "import tune_models, tune_cli"` がworkspaceとinstalled wheelの両方で成功する。
- `import models` 等のgeneric importへ依存するPRD codeが `rg` で0件になる。
- editable installとwheel installで同じmodule pathが解決される。
- `prd/src/tune_orchestrator` directoryが存在しない。

### Required tests

- `prd/src/test/test_packaging.py`: editable/wheel module一覧、entry point、packaged graph読込。
- 全既存test importを新module名へ更新する。

## PRD-012 Single production doctor and concise runbook

- [ ] production起動可否を `doctor` 1回で判定し、分散した手順と個別gateの見落としをなくす。

### Implementation

- `prd/config/production-profile.example.yaml` を追加し、次を設定可能にする。
  - runtime release current path、trusted public keys、channel。
  - router URL/model/timeoutとcalibration artifact。
  - model endpoint config、graph directory、trace directory。
  - probe有無、SLO、drift/alert rule path。
- `doctor --profile <yaml>` を正規interfaceとする。既存の個別引数はdevelopment互換として残す。
- `prd/src/tune_ops.py` のcheckを以下へ拡張する。
  - release/current/manifest/signature/digest/expiry。
  - router promotion、calibration、selection policy、orchestrator evaluation、bandit bundle。
  - graph/model capability整合性、credential環境変数、trace directoryのwrite可否と空き容量。
  - endpoint TLS policy、`/v1/models`、任意の最小chat probe。
  - releaseに `poc` pathやworkspace外pathが含まれないこと。
  - version互換性とclock skew。
- check resultは `id`、`status=pass|warn|fail`、`summary`、`actual`、`expected`、`remediation` を持つ。
- production profileではwarnも起動blockするかを `warnings_as_errors` で設定する。既定はtrue。
- `prd/doc/production-runbook.md` は「build/evaluate/promote/activate/doctor/run/rollback」の順に整理し、重複commandを削る。
- `prd/README.md` はquick startとrunbook linkだけに絞り、production詳細を二重管理しない。

### Acceptance criteria

- fresh hostでprofileとsecretだけを設定すれば、単一commandで不足項目が具体的に分かる。
- `doctor` passしたreleaseと `run --runtime-release` が同じartifact chainを検証する。
- check追加時にrunbookの手動checkを増やさず、machine-readable resultへ集約できる。
- `doctor` resultとprofile digestが起動traceに残る。

### Required tests

- `prd/src/test/test_ops.py`: profile load、全check、warning policy、remediation、offline/online probe。
- `prd/src/test/test_e2e_doctor.py`: valid release pass、各component改変fail。

## PRD-013 Runtime observability, drift, alerts, dashboard

- [ ] JSONL trace保存だけで終わらず、運用metric、drift判定、alert、可視化を生成する。

### Trace schema

- traceへ `format=tune-orchestrator-trace-v2` と `schema_version=2` を追加する。
- 必須fieldはtrace/request id、UTC timestamp、runtime release id/digest、router raw/calibrated scores、decision、plan digest、node timing/usage/status、stop reason、policy eventとする。
- user textとmodel outputの保存modeを `full|redacted|hash_only|none` から選べるようにし、production既定は `redacted` とする。
- `JsonlTraceSink` はPRD-008のlock/atomic appendを使い、日次fileへpartitionする。

### Metrics and drift

- `prd/src/tune_observability.py` を追加する。
- `build-observability-baseline --traces --out` で、label score/selection、graph、fallback、latency、costの基準分布とsource digestを保存する。
- `monitor-runtime --traces --baseline --rules --out --prometheus-out` を追加する。
- 最低限のmetric:
  - request count、completed/failure rate、stop reason別count。
  - latency p50/p95/p99、mean cost、token、node retry、fallback rate。
  - label/graph/model selection分布、multi-agent率、clarification率。
  - verifier pass、repair trigger/success、shadow/bandit trafficとloss。
  - feedback coverage、reviewed accuracy、macro F1、ECE。
- drift:
  - categorical分布はJensen-Shannon divergence。
  - confidence/latency/costはPSIを10 quantile binで計算する。
  - sample不足時はpass扱いにせず `insufficient_data` とする。

### Alerts and dashboard

- `prd/config/alert-rules.example.yaml` を追加し、window、minimum samples、warning/critical threshold、comparison directionを定義する。
- critical alertが1件以上なら `monitor-runtime` はexit code `1`、warningのみならartifactはwarnでexit code `0` とする。
- `render-runtime-dashboard --monitor <json> --out <html>` を追加し、外部CDNなしの単一HTMLへSLO、trend、drift、release marker、top failureを表示する。
- alert artifactへrule version、runtime release id、window、evidence trace idを保存し、同じalert key/windowの重複通知を抑制できる形にする。

### Acceptance criteria

- baselineと明確に異なるlabel分布fixtureでdrift alertが発生する。
- release切替前後を別seriesとして比較できる。
- secret/user raw textがPrometheus、alert、dashboardへ出ない。
- trace schema v1はmigration readerで読めるが、新規出力はv2だけにする。

### Required tests

- `prd/src/test/test_observability.py`: percentile、JSD、PSI、window、insufficient data、alert severity、redaction。
- `prd/src/test/test_trace_sink.py`: concurrent append、rotation、schema v1 migration。
- dashboard testはHTML内の必須sectionとsecret非包含を確認する。

## PRD-014 Unified release and rollback

- [ ] banditだけでなくrouter、calibration、orchestrator adapter、graphs、model config、policyを一つのreleaseとして昇格・rollbackできるようにする。

### Release contract

`tune-orchestrator-runtime-release-v1` manifestに以下を含める。

```json
{
  "release_id": "runtime-20260809-001",
  "channel": "production",
  "created_at": "...",
  "expires_at": "...",
  "status": "pass",
  "components": {
    "router": {},
    "router_calibration": {},
    "selection_policy": {},
    "orchestrator": {},
    "graphs": {},
    "model_config": {},
    "bandit": {}
  },
  "gates": {},
  "compatibility": {},
  "signature": {}
}
```

各componentは `path`、`digest`、`format`、`version` を持つ。学習componentはdataset/training/evaluation/promotion digestも持つ。`bandit` は未使用時のみnullを許可する。

### Implementation

- `prd/src/tune_release.py` を追加する。
  - `build_runtime_release()`、`validate_runtime_release()`、`activate_runtime_release()` を実装する。
  - current pointerはrelease id、manifest path/digest、channel、activated_at、revisionだけを持つ。
  - registryはappend-only logical historyとし、activation/rollback event、operator、change ticket、reasonを記録する。
  - 全writeはPRD-008のlock、CAS、signatureを使う。
- `prd/src/tune_cli.py` に以下を追加する。
  - `build-runtime-release`
  - `verify-runtime-release`
  - `activate-runtime-release`
  - `record-runtime-release`
  - `select-runtime-rollback`
  - `apply-runtime-rollback --reason --operator --change-ticket`
- promotion statusがpassでないrouter/orchestrator/bandit artifact、期限切れartifact、互換性不一致をreleaseへ含めない。
- rollback候補は同channel、signature valid、過去にhealth pass、現行とは異なる最新releaseとする。
- rollbackはcomponent単位に混ぜず、manifest全体を切り替える。部分rollbackが必要なら新releaseをbuildする。
- 既存bandit release commandは低level toolとして残し、production runbookではunified releaseだけを使う。

### Compatibility checks

- router label順とselector label集合。
- orchestrator plan schemaとexecutor supported schema。
- graph schemaとmodel capability alias。
- calibrationが参照するrouter model digest。
- bandit stateが参照するgraph/policy digest。
- package versionがreleaseの `min_runtime_version <= current < max_runtime_version` を満たすこと。

### Acceptance criteria

- 1つのcurrent pointerから実行に必要な全artifactを解決できる。
- routerだけ旧版、banditだけ新版のような未検証混在が起きない。
- rollback後の次requestから全componentが旧releaseへ戻る。
- activation競合はCASで片方だけ成功し、履歴を失わない。
- audit recordからwho/when/from/to/reason/change ticketを追跡できる。

### Required tests

- `prd/src/test/test_release.py`: build/verify/activate、compatibility、signature、CAS、registry、rollback選択、partial mix拒否。
- `prd/src/test/test_e2e_release.py`: release A実行、Bへactivate、B障害、A rollback、再実行。

## PRD-015 Security hardening

- [ ] 外部endpoint、artifact、training model、secret、traceに対するproduction security controlを実装する。

### Endpoint and HTTP client

- `prd/src/tune_clients.py` に共通 `HttpClientPolicy` を追加する。
- URLにuserinfo、fragment、query上のcredential、unsupported schemeがあれば拒否する。
- `http` はloopback addressだけ許可し、それ以外は `https` 必須とする。
- production profileの `allowed_hosts` にないhost、redirect先host変更、DNS解決後の禁止address帯を拒否する。
- response body上限をdefault 4 MiB、error body記録上限を1 KiBとする。
- retryは429/502/503/504と接続失敗だけに限定し、指数backoff+jitter、最大3回、全体deadline内で行う。
- TLS CA path、client certificateをprofileから設定可能にし、TLS verification無効化optionはproductionで禁止する。
- requestごとにcorrelation idを送り、Authorization headerとURL queryをlogへ出さない。

### Secrets and traces

- recursive redactorを実装し、dict/list/string内のAuthorization、API key、private key、JWT、credential-like queryをmaskする。
- exception message、upstream response、node output、debug traceにも同じredactorを適用する。
- API keyは環境変数またはsecret provider referenceだけを許可し、YAMLへのinline valueを `doctor` でfailする。
- production traceのfile permissionをowner-only相当で作成し、retention日数と削除責任をprofileへ明記する。

### Model and training supply chain

- `trust_remote_code` の既定値をfalseへ変更する。
- remote codeが必要なmodelは `--allow-remote-code`、固定revision、承認済みallow-listの3条件を必須にする。
- base model idだけでなくcommit revision、file digest、tokenizer digestをtraining metadataへ保存する。
- adapter load時は `safetensors` を優先し、pickle系weightはproductionで拒否する。
- CIでdependency audit、secret scan、artifact signature testを実行する。

### Runtime policy

- input文字数、node output文字数、JSON nesting depthを上限化する。
- model生成plan/outputを信頼せず、schema、allow-list、budget、policyを実行直前に再検証する。
- high-risk requestではunsigned override、shadow exploration、unapproved bandit armを禁止する。
- break-glass操作はoperator、change ticket、期限、理由を必須にし、security event traceへ記録する。

### Acceptance criteria

- non-loopback HTTP、redirectによるhost escape、oversized response、inline secretが拒否される。
- malicious upstream errorにsecretが含まれてもstdout/stderr/traceへ平文が出ない。
- unpinned remote-code modelとunsigned production releaseをloadできない。
- security rejectionは一般的な `node_failed` ではなく、機密情報を含まない安定したfailure codeで識別できる。

### Required tests

- `prd/src/test/test_security.py`: URL policy、redirect、DNS/address policy、response limit、retry対象、redaction、break-glass。
- `prd/src/test/test_training.py`: remote code opt-in、revision必須、unsafe weight拒否、metadata digest。
- `prd/src/test/test_e2e_security.py`: fake endpointを使ったsecret漏えいとoversized response検証。

## Final production exit criteria

全TODO完了後、production readyと判定する条件:

- [ ] PRD-002からPRD-015の全checkboxと受け入れ条件が完了している。
- [ ] mandatory CIがmain branchでpassし、required checkに設定されている。
- [ ] router、orchestrator、banditを含むunified releaseが署名済みで、全promotion gateがpassしている。
- [ ] `doctor --profile ./prd/config/production.yaml` がwarningなしでpassする。
- [ ] valid releaseのE2E、tamper時fail closed、rollback rehearsalが成功している。
- [ ] canary監視で定義済みminimum samplesを満たし、critical alertとdrift alertがない。
- [ ] production runbookに担当者、change ticket、rollback release id、incident連絡先が記録されている。
