# TuneRouter PoC Report

## 方針

このPoCでは、下流アプリのRouterカテゴリに合わせた分類データをローカルJSONで管理し、`Qwen/Qwen2.5-0.5B` をLoRAでFine-Tuningします。

## データ件数

| split | Storage | Network | Coding | Security | Database | General | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 497 | 504 | 494 | 491 | 493 | 467 | 2946 |
| dev | 82 | 80 | 106 | 115 | 117 | 115 | 615 |
| test | 121 | 116 | 100 | 94 | 90 | 118 | 639 |

## 次の判断

- カテゴリ境界が下流アプリのRouter設定と一致しているか、実データで確認する
- 運用ログ、レビュー済み質問、ユーザー操作由来の質問を同じJSONスキーマに追加する
- Qwen2.5-0.5BのLoRA FT結果で誤分類例を見る
- 精度が不足する場合は、データ境界、学習率、epoch、LoRA rank、入力テンプレートを調整する
- 回答品質採点とOracleラベル生成は、複数モデル運用の価値が見えてから追加する
