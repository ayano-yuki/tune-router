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
uv run --project .\poc python .\poc\src\cli.py prepare-synthetic-data --per-label 1000 --out .\poc\artifacts
```

## データ作成

カテゴリの配線確認や初回FTでは、下流カテゴリに合わせた合成データから始めます。

```powershell
uv run --project .\poc python .\poc\src\cli.py prepare-synthetic-data --per-label 1000 --out .\poc\artifacts
```

少量で試す場合:

```powershell
uv run --project .\poc python .\poc\src\cli.py prepare-synthetic-data --per-label 40 --out .\poc\artifacts
```

OSSデータセットを使う場合:

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py prepare-data --per-label 1000 --out .\poc\artifacts
```

Storage / Network / Database はOSSソースだけだと十分な件数や品質が出ない可能性があります。
実運用に近い質問ログやレビュー済みの手作りデータを、同じJSONスキーマで追加する前提です。

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
uv run --project .\poc --system-certs python .\poc\src\cli.py predict-qwen --adapter .\poc\artifacts\qwen-router-lora "Cephのosdが頻繁にdownする原因を切り分けたい"
```

データ作成、FT、評価をまとめて実行する場合:

```powershell
uv run --project .\poc --system-certs python .\poc\src\cli.py run --per-label 1000 --out .\poc\artifacts --epochs 1
```

## 生成物

| ファイル | 内容 |
| --- | --- |
| `dataset.json` | 全データ |
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
    "data_origin": "synthetic_poc",
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
      "source": "synthetic_poc",
      "template_id": "storage.ceph",
      "split_group": "storage.ceph:1",
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
