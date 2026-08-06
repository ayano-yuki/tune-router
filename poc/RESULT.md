# TuneRouter PoC Result

## Summary

OSS-only datasetで `Qwen/Qwen2.5-0.5B` のLoRA Fine-Tuningルータを評価した。

合成データは生成・補完に使わず、各カテゴリ800件をOSS由来データだけで作成した。
カテゴリ名入りの日本語プレフィックスはラベルリークになるため除去し、元データの質問文・依頼文を入力にした。

| condition | accuracy | macro_f1 | correct / total |
| --- | ---: | ---: | ---: |
| LoRA Fine-Tuningあり | 0.873 | 0.877 | 664 / 761 |
| Fine-Tuningなし | 0.239 | 0.163 | - |

FTありはFTなしに対して、accuracyで `+0.634`、macro_f1で `+0.714` 改善した。

## Dataset

最終データは下流Routerカテゴリに合わせた6ラベル構成。
各カテゴリ800件、全体4800件をOSS由来データだけで作成した。

| split | Storage | Network | Coding | Security | Database | General | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 525 | 555 | 556 | 569 | 551 | 567 | 3323 |
| dev | 130 | 119 | 112 | 122 | 120 | 113 | 716 |
| test | 145 | 126 | 132 | 109 | 129 | 120 | 761 |

## Source Mix

| Category | OSS | Synthetic | Total | Main source |
| --- | ---: | ---: | ---: | --- |
| Storage | 800 | 0 | 800 | Stack Exchange API |
| Network | 800 | 0 | 800 | Stack Exchange API |
| Coding | 800 | 0 | 800 | Magicoder OSS Instruct |
| Security | 800 | 0 | 800 | Trendyol Cybersecurity Instruction |
| Database | 800 | 0 | 800 | Stack Exchange API / sql-create-context |
| General | 800 | 0 | 800 | databricks-dolly-15k |

| Source type | Count | Ratio |
| --- | ---: | ---: |
| OSS | 4800 | 100.0% |
| Synthetic | 0 | 0.0% |
| Total | 4800 | 100.0% |

## Fine-Tuned Result

LoRA FT後の評価結果。

| actual \ predicted | Storage | Network | Coding | Security | Database | General |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Storage | 117 | 5 | 2 | 1 | 20 | 0 |
| Network | 5 | 119 | 0 | 1 | 1 | 0 |
| Coding | 3 | 6 | 108 | 1 | 13 | 1 |
| Security | 0 | 0 | 0 | 108 | 1 | 0 |
| Database | 13 | 4 | 7 | 2 | 98 | 5 |
| General | 0 | 1 | 1 | 1 | 3 | 114 |

## Interpretation

- OSS-only datasetでも、LoRA FTによりルータ分類は大きく改善した。
- `Security`、`Network`、`General` は比較的安定している。
- 主な誤分類は `Storage` / `Database`、`Coding` / `Database` の境界に集中している。
- `Database` はSQL、PostgreSQL運用、DBバックアップ、replication、stored procedure、query planなどが混在し、`Storage` や `Coding` と意味的に近い。
- `Storage` はNFS、backup、disk、Cephなどで概ね妥当だが、PostgreSQLやログ保存を含む質問では `Database` に吸われやすい。
- `Coding` は実装タスクとして妥当だが、JSON metadata、log processing、repositoryなどの語により `Database` 側へ誤分類される例がある。
- `Security` は精度上は安定しているが、攻撃実装寄りの質問が混ざるため、防御用途に寄せるなら追加フィルタが必要。

## Next Actions

- `Database` / `Storage` / `Coding` の境界をデータフィルタで締める。
- `Storage` から `postgres`、`mysql`、`database` が強い質問を除外または `Database` へ寄せる。
- `Database` は `sql-create-context` の一般的すぎる短文を減らし、DBA/Stack Exchange由来の運用質問を増やす。
- `Security` は `BadUSB`、`payload`、`bypass`、攻撃実装系の質問を除外し、防御・検知・ハードニング寄りにする。
- 修正後も同じtest分割で再評価し、特に `Database` recall / precision を確認する。

## Commands

Data preparation:

```bash
uv run --project ./poc --system-certs python ./poc/src/cli.py prepare-data \
  --per-label 800 \
  --out ./poc/artifacts \
  --max-source-scan 200000
```

Fine-Tuning:

```bash
uv run --project ./poc --system-certs python ./poc/src/cli.py train-qwen \
  --train ./poc/artifacts/train.json \
  --dev ./poc/artifacts/dev.json \
  --output ./poc/artifacts/qwen-router-lora \
  --epochs 1
```

Evaluation:

```bash
uv run --project ./poc --system-certs python ./poc/src/cli.py evaluate-qwen \
  --adapter ./poc/artifacts/qwen-router-lora \
  --data ./poc/artifacts/test.json \
  --train ./poc/artifacts/train.json \
  --dev ./poc/artifacts/dev.json \
  --report ./poc/artifacts/report.md
```

No Fine-Tuning baseline:

```bash
uv run --project ./poc --system-certs python ./poc/src/cli.py evaluate-base-qwen \
  --data ./poc/artifacts/test.json \
  --train ./poc/artifacts/train.json \
  --dev ./poc/artifacts/dev.json \
  --predictions ./poc/artifacts/predictions_base.json \
  --report ./poc/artifacts/report_base.md
```
