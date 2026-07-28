# TuneRouter PoC Report

## 方針

このPoCでは、OSS由来の4分類データをローカルJSONで管理し、`Qwen/Qwen2.5-0.5B` をLoRAでFine-Tuningします。

## データ件数

| split | code | security_log | iac_text | general | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 692 | 685 | 702 | 699 | 2778 |
| dev | 161 | 175 | 156 | 154 | 646 |
| test | 147 | 140 | 142 | 147 | 576 |

## 次の判断

- OSS由来データの中身とライセンスを確認し、必要なら実データを同じJSONスキーマに追加する
- Qwen2.5-0.5BのLoRA FT結果で誤分類例を見る
- 精度が不足する場合は、データ境界、学習率、epoch、LoRA rank、入力テンプレートを調整する
- 回答品質採点とOracleラベル生成は、複数モデル運用の価値が見えてから追加する
