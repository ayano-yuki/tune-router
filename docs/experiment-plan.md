# Experiment Plan

## Theme

日本語 Instruction Tuning における FT 手法の比較。

同じ日本語 Instruction データを異なる Fine-Tuning 手法で学習したときの、性能・コスト・回答傾向・一般能力の変化を比較します。

## Data

### SFT

Primary dataset:

- `llm-jp/llm-jp-instructions`

Additional candidates:

- `llm-jp/oasst2-33k-ja`
- `databricks-dolly-15k-ja`
- `izumi-lab/llm-japanese-dataset`

実験順は、まず `llm-jp-instructions` のみ、次に日本語翻訳 Instruction、最後に大規模合成データの追加とします。

### DPO

Primary dataset:

- `llm-jp/hh-rlhf-12k-ja`

Extension:

- `llm-jp/llm-jp-4-8b-thinking-dpo-data`

最初の DPO では HH-RLHF 翻訳データを使い、推論特化データの効果と DPO の効果を混ぜないようにします。

## Phases

### Phase 0: Baseline

ベースモデルをそのまま評価します。

- JGLUE
- ELYZA-tasks-100
- XL-Sum Japanese subset
- inference speed
- GPU memory
- output length
- JSON and format-following rate

### Phase 1: QLoRA Data Size

`llm-jp-instructions` から件数を変えて QLoRA 学習します。

- 100
- 500
- 2,000
- all

固定するもの:

- base model
- QLoRA settings
- epoch
- learning rate
- seed
- max sequence length
- evaluation condition

### Phase 2: LoRA Rank

データ件数を固定し、rank のみを変えます。

- r = 8
- r = 32
- r = 64

### Phase 3: LoRA vs QLoRA

同じデータ、同じ rank で、ベースモデルの量子化だけを変えます。

- LoRA: BF16 or FP16 base model
- QLoRA: 4-bit NF4 base model

### Phase 4: DPO

最良の SFT モデルを初期モデルにし、`llm-jp/hh-rlhf-12k-ja` で DPO します。

比較対象:

- Base Model
- SFT Model
- SFT + DPO Model

### Phase 5: Full SFT

計算資源が許す場合のみ実施します。

比較対象:

- LoRA SFT
- QLoRA SFT
- Full SFT

## First Matrix

| ID | Method | Data | Count | Rank |
| --- | --- | --- | ---: | ---: |
| B0 | Base | none | 0 | - |
| Q1 | QLoRA | llm-jp/llm-jp-instructions | 100 | 32 |
| Q2 | QLoRA | llm-jp/llm-jp-instructions | 500 | 32 |
| Q3 | QLoRA | llm-jp/llm-jp-instructions | 2,000 | 32 |
| Q4 | QLoRA | llm-jp/llm-jp-instructions | all | 32 |
| R1 | QLoRA | same fixed data | fixed | 8 |
| R2 | QLoRA | same fixed data | fixed | 32 |
| R3 | QLoRA | same fixed data | fixed | 64 |
| L1 | LoRA | same fixed data | fixed | 32 |
| D1 | DPO | llm-jp/hh-rlhf-12k-ja | all | 32 |
| F1 | Full SFT | same fixed data | fixed | - |

Q3 と R2 のように同一条件になる実験は、同じ結果を共有します。

