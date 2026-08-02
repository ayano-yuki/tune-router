# TuneRouter PoC Report

## 方針

このPoCでは、下流アプリのRouterカテゴリに合わせた分類データをローカルJSONで管理し、`Qwen/Qwen2.5-0.5B` をLoRAでFine-Tuningします。

## データ件数

| split | Storage | Network | Coding | Security | Database | General | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 692 | 690 | 780 | 642 | 643 | 594 | 4041 |
| dev | 170 | 202 | 121 | 191 | 197 | 211 | 1092 |
| test | 138 | 108 | 99 | 167 | 160 | 195 | 867 |

## 精度

| split | accuracy | macro_f1 | correct / total |
| --- | ---: | ---: | ---: |
| test | 1.000 | 1.000 | 867 / 867 |

## Test Confusion Matrix

| actual \ predicted | Storage | Network | Coding | Security | Database | General |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Storage | 138 | 0 | 0 | 0 | 0 | 0 |
| Network | 0 | 108 | 0 | 0 | 0 | 0 |
| Coding | 0 | 0 | 99 | 0 | 0 | 0 |
| Security | 0 | 0 | 0 | 167 | 0 | 0 |
| Database | 0 | 0 | 0 | 0 | 160 | 0 |
| General | 0 | 0 | 0 | 0 | 0 | 195 |

## 誤分類例

誤分類はありませんでした。実データではここに境界の悪い質問が出る想定です。

## 次の判断

- カテゴリ境界が下流アプリのRouter設定と一致しているか、実データで確認する
- 必要なら運用ログやレビュー済み質問を同じJSONスキーマに追加する
- Qwen2.5-0.5BのLoRA FT結果で誤分類例を見る
- 精度が不足する場合は、データ境界、学習率、epoch、LoRA rank、入力テンプレートを調整する
- 回答品質採点とOracleラベル生成は、複数モデル運用の価値が見えてから追加する
