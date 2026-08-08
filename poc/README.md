# TuneRouter PoC

このPoCは、下流アプリのRouterカテゴリに合わせた分類データをローカルJSONへ正規化し、そのデータで
`Qwen/Qwen2.5-0.5B` をルータ分類器としてFine-Tuningするためのものです。

専門モデル本体をSFTするのではなく、ユーザー質問を `Storage`、`Network`、`Coding`、`Security`、
`Database`、`General` のどれに送るかを判定する小型ルータをFTします。

`General` はフォールバック用の教師ラベルです。下流アプリ側でカテゴリ一覧から `General` を除外し、
照合失敗時の fallback として扱う構成でも、学習時には「該当なし」の例として持たせます。

## 方針

- データラベルは下流アプリのRouterカテゴリ名と同じ文字列にする
- train/dev/testをJSONとして分割する
- `Qwen/Qwen2.5-0.5B` を6ラベル分類器としてLoRA FTする
- 精度が悪い場合は、まずカテゴリ境界と学習データを直す

## 分類

| ラベル | 想定ルート |
| --- | --- |
| `Storage` | Ceph、ZFS、RAID、NAS/SAN、iSCSI、NFS、容量、ディスク障害 |
| `Network` | BGP、OSPF、VLAN、DNS、DHCP、VPN、MTU、L2/L3疎通 |
| `Coding` | 実装、デバッグ、テスト、リファクタリング、ビルド、API |
| `Security` | CVE、インシデント、認証認可、TLS、WAF、脆弱性対応 |
| `Database` | PostgreSQL、MySQL、Oracle、Redis、MongoDB、クエリ改善、レプリケーション |
| `General` | 上記に該当しない一般質問、相談、要約、比較 |

## セットアップ

`uv run` で依存関係を解決して実行します。初回はPyTorch、Transformers、PEFT、
Qwen2.5-0.5Bのモデル取得が走るため時間がかかります。

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py prepare-data --per-label 800 --out .\poc\artifacts
```

## データ作成

OSS由来データだけを収集し、カテゴリごとに質問・依頼文形式へ正規化します。
指定件数をOSSだけで満たせない場合、合成データでは補完せずエラーにします。

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py prepare-data --per-label 800 --out .\poc\artifacts
```

少量で試す場合:

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py prepare-data --per-label 40 --out .\poc\artifacts
```

### 利用データセット

学習データはOSS由来の質問・依頼文だけを使い、各レコードに `source_name`、`source_url`、
`source_license`、`source_record_id` を保持します。`Stack Exchange API` 由来のデータは質問タイトルと
本文を結合し、Hugging Face datasets 由来のデータは各データセットの質問・プロンプト系フィールドを
抽出して、共通のルータ分類JSONへ正規化します。

| ラベル | source_name | 元データセット/API | ライセンス |
| --- | --- | --- | --- |
| `Storage` | `stackexchange-storage` | Stack Exchange API (`serverfault`, `unix`, `superuser`) | `cc-by-sa-4.0` |
| `Storage` | `dolly-storage-fallback` | `databricks/databricks-dolly-15k` | `cc-by-sa-3.0` |
| `Network` | `stackexchange-network` | Stack Exchange API (`networkengineering`, `serverfault`, `superuser`) | `cc-by-sa-4.0` |
| `Network` | `netconfeval-config-generation` | `NetConfEval/NetConfEval` (`Configuration Generation`) | `mit` |
| `Coding` | `magicoder-oss-instruct` | `ise-uiuc/Magicoder-OSS-Instruct-75K` | `mit` |
| `Coding` | `codeinstruct-20k` | `SoyMaycol/CodeInstruct-20K` | `cc-by-4.0` |
| `Security` | `trendyol-cybersecurity-instruct` | `Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset` | `apache-2.0` |
| `Database` | `sql-create-context` | `b-mc2/sql-create-context` | `cc-by-4.0` |
| `Database` | `stackexchange-database` | Stack Exchange API (`dba`, `serverfault`) | `cc-by-sa-4.0` |
| `General` | `dolly-general` | `databricks/databricks-dolly-15k` | `cc-by-sa-3.0` |

現行の `poc/artifacts` は `--per-label 800`、`seed=42` で作成した英語ベースのOSS-onlyデータです。
全体は4,800件で、各ラベル800件ずつ含みます。分割は `gold_label` と `split_group` から安定ハッシュで
決めており、おおむねtrain/dev/test = 70/15/15です。

| ファイル | 用途 | 件数 |
| --- | --- | ---: |
| `poc/artifacts/dataset.json` | 全データ | 4,800 |
| `poc/artifacts/train.json` | 学習用 | 3,323 |
| `poc/artifacts/dev.json` | 調整・確認用 | 716 |
| `poc/artifacts/test.json` | 固定評価用 | 761 |

`poc/artifacts-ja` には同じレコード構成の日本語版を置いています。`metadata.data_origin` は
`oss_only_ja`、`metadata.translation.target_language` は `ja` で、元の英語データは
`text_en`、日本語化した入力文は `text` として保持します。件数とsplitは `poc/artifacts` と同じです。

中期的には、Storage / Network / Database 向けのOSSソースを見直し、実運用に近い質問ログや
レビュー済みの手作りデータを同じJSONスキーマで追加します。

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

LoRA adapterを使ったFT後のルータを評価します。

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py evaluate-qwen --adapter .\poc\artifacts\qwen-router-lora --data .\poc\artifacts\test.json
```

FTなしの分類ヘッド初期状態を比較評価する場合:

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py evaluate-base-qwen --data .\poc\artifacts\test.json --report .\poc\artifacts\report_base.md
```

単発予測:

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py predict-qwen --adapter .\poc\artifacts\qwen-router-lora "Cephのosdが頻繁にdownする原因を切り分けたい"
```

データ作成、FT、評価をまとめて実行する場合:

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py run --per-label 800 --out .\poc\artifacts --epochs 1
```

## 生成物

| ファイル | 内容 |
| --- | --- |
| `dataset.json` | 全データ |
| `train.json` | 学習用データ |
| `dev.json` | 調整・確認用データ |
| `test.json` | 固定評価用データ |
| `qwen-router-lora/` | Qwen2.5-0.5BのLoRA adapter |
| `report.md` | FTあり評価レポート |
| `report_base.md` | FTなし評価レポート |

## 実装構成

PoC本体は `src/` 配下に責務別で分けています。

| ファイル | 責務 |
| --- | --- |
| `src/cli.py` | サブコマンド定義と処理の組み立て |
| `src/config.py` | ラベル、ルーティング先、OSSデータソース |
| `src/oss_data.py` | OSSデータセット/APIの取得と質問形式への正規化 |
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
    "data_origin": "oss_only",
    "router_base_model": "Qwen/Qwen2.5-0.5B",
    "labels": ["Storage", "Network", "Coding", "Security", "Database", "General"],
    "split": "train",
    "requested_per_label": 1000,
    "seed": 42
  },
  "records": [
    {
      "question_id": "poc-storage-000001",
      "text": "Cephのosdが頻繁にdownする原因を切り分けたい。",
      "gold_label": "Storage",
      "label_id": 0,
      "target_model": "storage-specialist",
      "tags": ["Storage", "ceph"],
      "importance": "normal",
      "source": "Stack Exchange API",
      "source_name": "stackexchange-storage",
      "source_license": "cc-by-sa-4.0",
      "template_id": "stackexchange-storage",
      "split_group": "stackexchange-storage:1",
      "split": "train",
      "rubric": {
        "minimum_quality": 0.75,
        "requires_human_review": false
      }
    }
  ]
}
```

実データへ置き換えるときも、このJSON構造とカテゴリ名を保てばそのまま `train-qwen` に流せます。
