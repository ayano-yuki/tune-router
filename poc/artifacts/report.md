# TuneRouter PoC Report

## 方針

このPoCでは、OSS由来の4分類データをローカルJSONで管理し、`Qwen/Qwen2.5-0.5B` をLoRAでFine-Tuningします。

## データ件数

| split | code | security_log | iac_text | general | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 692 | 685 | 702 | 699 | 2778 |
| dev | 161 | 175 | 156 | 154 | 646 |
| test | 147 | 140 | 142 | 147 | 576 |

## 精度

| split | accuracy | macro_f1 | correct / total |
| --- | ---: | ---: | ---: |
| test | 0.998 | 0.998 | 575 / 576 |

## Test Confusion Matrix

| actual \ predicted | code | security_log | iac_text | general |
| --- | ---: | ---: | ---: | ---: |
| code | 146 | 0 | 0 | 1 |
| security_log | 0 | 140 | 0 | 0 |
| iac_text | 0 | 0 | 142 | 0 |
| general | 0 | 0 | 0 | 147 |

## 誤分類例

- `code` -> `general` (0.996): Create a tree data structure for the following information: a) Naruto b) Sasuke c) Boruto d) Sarada

## 次の判断

- OSS由来データの中身とライセンスを確認し、必要なら実データを同じJSONスキーマに追加する
- Qwen2.5-0.5BのLoRA FT結果で誤分類例を見る
- 精度が不足する場合は、データ境界、学習率、epoch、LoRA rank、入力テンプレートを調整する
- 回答品質採点とOracleラベル生成は、複数モデル運用の価値が見えてから追加する
