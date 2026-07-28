# TuneRouter PoC

このPoCは、OSS由来の4分類データセットをローカルJSONへ正規化し、そのデータで `Qwen/Qwen2.5-0.5B` をルータ分類器としてFine-Tuningするためのものです。

専門モデル本体をSFTするのではなく、ユーザー質問を `code`、`security_log`、`iac_text`、`general` のどれに送るかを判定する小型ルータをFTします。

## 方針

- データ元はHugging Face上のOSS由来データセットを使う
- `--per-label` で指定した件数程度を各分類に作る
- train/dev/testをJSONとして分割する
- `Qwen/Qwen2.5-0.5B` を4ラベル分類器としてLoRA FTする
- 精度が悪い場合は、まずデータとラベル境界を直す

## OSSデータソース

| ラベル | データセット | ライセンス |
| --- | --- | --- |
| `code` | `SoyMaycol/CodeInstruct-20K` | `cc-by-4.0` |
| `security_log` | `Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset` | `apache-2.0` |
| `iac_text` | `galcan/terraform_sec` | `apache-2.0` |
| `general` | `databricks/databricks-dolly-15k` | `cc-by-sa-3.0` |

各レコードには `source`、`source_url`、`source_license`、`source_record_id` を残します。

## 分類

| ラベル | 想定ルート |
| --- | --- |
| `code` | コード生成、デバッグ、SQL、API実装 |
| `security_log` | CVE、ログ分析、検知、インシデント調査 |
| `iac_text` | Terraform、Kubernetes、Docker、IaC、技術文書作成 |
| `general` | 一般質問、説明、相談、要約 |

## セットアップ

`uv run` で依存関係を解決して実行します。初回はPyTorch、Transformers、PEFT、Qwen2.5-0.5Bのモデル取得が走るため時間がかかります。

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py prepare-data --per-label 1000 --out .\poc\artifacts
```

## データ作成

OSSデータセットから各ラベル1000件程度を抽出し、ローカルJSONを作ります。

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py prepare-data --per-label 1000 --out .\poc\artifacts
```

少量で試す場合:

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py prepare-data --per-label 40 --out .\poc\artifacts
```

合成データで配線だけ確認したい場合:

```powershell
uv run --project .\poc python .\poc\src\cli.py prepare-synthetic-data --per-label 40 --out .\poc\artifacts
```

## Qwen2.5-0.5BをFT

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py train-qwen --train .\poc\artifacts\train.json --dev .\poc\artifacts\dev.json --output .\poc\artifacts\qwen-router-lora --epochs 1
```

GPUで半精度を使う場合:

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py train-qwen --bf16 --epochs 1
```

環境によっては `--fp16` の方が合います。

## 評価

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py evaluate-qwen --adapter .\poc\artifacts\qwen-router-lora --data .\poc\artifacts\test.json
```

単発予測:

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py predict-qwen --adapter .\poc\artifacts\qwen-router-lora "Terraformのstate driftを検出する手順を整理して"
```

データ作成、FT、評価をまとめて実行する場合:

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py run --per-label 1000 --out .\poc\artifacts --epochs 1
```

## 生成物

| ファイル | 内容 |
| --- | --- |
| `dataset.json` | OSS由来の全データ |
| `train.json` | 学習用データ |
| `dev.json` | 調整・確認用データ |
| `test.json` | 固定評価用データ |
| `qwen-router-lora/` | Qwen2.5-0.5BのLoRA adapter |
| `predictions_test.json` | test予測結果 |
| `report.md` | 件数、精度、混同行列、誤分類例 |

## 実装構成

PoC本体は `src/` 配下に責務別で分けています。

| ファイル | 責務 |
| --- | --- |
| `src/cli.py` | サブコマンド定義と処理の組み立て |
| `src/config.py` | ラベル、ルーティング先、OSSデータソース |
| `src/oss_data.py` | Hugging Face OSSデータセットの取得と正規化 |
| `src/synthetic_data.py` | 配線確認用の合成データ生成 |
| `src/splitting.py` | train/dev/test分割 |
| `src/json_store.py` | ローカルJSONの読み書き |
| `src/qwen_router.py` | Qwen2.5-0.5B LoRA FTと評価の補助関数 |
| `src/reporting.py` | レポート生成 |
| `src/types_local.py` | 共通データ型 |

## JSONスキーマ

```json
{
  "metadata": {
    "format": "tune-router-json-v1",
    "data_origin": "oss_huggingface",
    "router_base_model": "Qwen/Qwen2.5-0.5B",
    "labels": ["code", "security_log", "iac_text", "general"],
    "split": "train",
    "requested_per_label": 1000,
    "seed": 42
  },
  "records": [
    {
      "question_id": "real-code-000001",
      "text": "質問本文",
      "gold_label": "code",
      "label_id": 0,
      "target_model": "qwen2.5-coder-7b",
      "tags": ["code", "debug"],
      "importance": "normal",
      "source": "internal_export",
      "source_url": "https://huggingface.co/datasets/example",
      "source_license": "apache-2.0",
      "source_record_id": "example-id",
      "template_id": "source-or-cluster-id",
      "split_group": "source-or-cluster-id",
      "split": "train",
      "rubric": {
        "minimum_quality": 0.75,
        "requires_human_review": false
      }
    }
  ]
}
```

実データへ置き換えるときも、このJSON構造を保てばそのまま `train-qwen` に流せます。
