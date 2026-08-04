# TuneRouter PoC Result

## Summary

OSSで取得できるデータを最大限使い、不足分を合成データで補完した mixed dataset で、
`Qwen/Qwen2.5-0.5B` のLoRA Fine-Tuningルータを評価した。

| condition | accuracy | macro_f1 | correct / total |
| --- | ---: | ---: | ---: |
| LoRA Fine-Tuningあり | 0.908 | 0.910 | 785 / 865 |
| Fine-Tuningなし | 0.191 | 0.099 | 165 / 865 |

FTありはFTなしに対して、accuracyで `+0.717`、macro_f1で `+0.811` 改善した。

## Dataset

最終データは下流Routerカテゴリに合わせた6ラベル構成。
各カテゴリ1000件を目標に、OSS由来データで不足するカテゴリは合成データで補完した。

| split | Storage | Network | Coding | Security | Database | General | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 714 | 716 | 695 | 734 | 711 | 702 | 4272 |
| dev | 130 | 136 | 156 | 131 | 154 | 156 | 863 |
| test | 156 | 148 | 149 | 135 | 135 | 142 | 865 |

## Source Mix

| Category | OSS | Synthetic | Total | OSS割合 | Synthetic割合 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Storage | 316 | 684 | 1000 | 31.6% | 68.4% |
| Network | 102 | 898 | 1000 | 10.2% | 89.8% |
| Coding | 1000 | 0 | 1000 | 100.0% | 0.0% |
| Security | 1000 | 0 | 1000 | 100.0% | 0.0% |
| Database | 42 | 958 | 1000 | 4.2% | 95.8% |
| General | 1000 | 0 | 1000 | 100.0% | 0.0% |

| Source | Count | Ratio |
| --- | ---: | ---: |
| OSS | 3460 | 57.7% |
| Synthetic | 2540 | 42.3% |
| Total | 6000 | 100.0% |

## Fine-Tuned Result

LoRA FT後の評価結果。

| actual \ predicted | Storage | Network | Coding | Security | Database | General |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Storage | 127 | 2 | 0 | 0 | 0 | 27 |
| Network | 1 | 137 | 1 | 0 | 0 | 9 |
| Coding | 0 | 1 | 143 | 2 | 0 | 3 |
| Security | 0 | 0 | 3 | 127 | 5 | 0 |
| Database | 0 | 0 | 0 | 0 | 134 | 1 |
| General | 18 | 3 | 3 | 1 | 0 | 117 |

主な誤分類は `Storage` / `General`、`General` / `Storage` の境界に集中している。
OSS由来データの一部に、カテゴリ名と本文内容がずれる例が含まれているため、
次はOSS抽出条件とラベル品質を見直す。

## No Fine-Tuning Baseline

LoRA adapterを使わず、ベースモデルに未学習の分類ヘッドを載せた状態で評価した結果。

| actual \ predicted | Storage | Network | Coding | Security | Database | General |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Storage | 149 | 2 | 0 | 1 | 4 | 0 |
| Network | 140 | 0 | 3 | 1 | 2 | 2 |
| Coding | 6 | 2 | 3 | 6 | 117 | 15 |
| Security | 39 | 4 | 0 | 8 | 55 | 29 |
| Database | 115 | 7 | 10 | 0 | 2 | 1 |
| General | 124 | 0 | 1 | 3 | 11 | 3 |

この比較は「FT済みadapter」と「未学習分類ヘッド」の比較であり、
プロンプト分類によるゼロショットルーティングとの公平比較ではない。

## Interpretation

- mixed dataset上でも、LoRA FTによりルータ分類は大きく改善した。
- `Coding`、`Security`、`Database` は比較的安定している。
- `Storage` と `General` の境界は、OSS抽出データの品質確認が必要。
- `Network` はOSS件数が少なく合成比率が高いため、実運用ログやレビュー済み質問の追加が必要。
- 実運用前には、運用ログやユーザー操作由来質問を同じJSONスキーマで追加し、固定testで再評価する。

## Commands

Data preparation:

```bash
uv run --project ./poc --system-certs python ./poc/src/cli.py prepare-data \
  --per-label 1000 \
  --out ./poc/artifacts
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
