# TuneRouter

ファインチューニングを用いて、質問の内容に応じた最適な AI モデル選択の精度向上を目指すプロジェクト。

## 目的

複数の専門 AI モデルを用意した環境では、すべての質問を最大モデルへ投げるよりも、質問の種類に応じて適切なモデルを選ぶ方が、精度・速度・GPU メモリ効率のバランスを取りやすい。

TuneRouter では、入力された質問を `Storage`、`Network`、`Coding`、`Security`、`Database`、`General` に分類し、対応する専門モデルまたはフォールバックへ振り分けるための小さな判定モデルを学習する。

## 想定構成

初期検証では次の構成を基本にする。

| 役割 | 候補モデル | 想定 VRAM |
| --- | --- | --- |
| ルーターモデル | Qwen3-0.6B / Qwen3-1.7B / TinySwallow-1.5B-Instruct | 約 0.5GB - 2GB |
| Storage 専門 | Qwen2.5-7B-Instruct-1M / Qwen2.5-14B-Instruct-1M | 約 5GB - 10GB |
| Network 専門 | Qwen2.5-7B-Instruct-1M / Qwen2.5-14B-Instruct-1M | 約 5GB - 10GB |
| Coding 専門 | Qwen2.5-Coder-7B / DeepSeek-Coder-V2-Lite-Instruct / Codestral 7B | 約 5GB - 10GB |
| Security 専門 | Qwen2.5-7B-Instruct-1M / Qwen2.5-14B-Instruct-1M / Phi-3.5-MoE-instruct | 約 5GB - 12GB |
| Database 専門 | Qwen2.5-7B-Instruct-1M / Qwen2.5-14B-Instruct-1M | 約 5GB - 10GB |
| General / 軽量即答 | Qwen3-0.6B / Qwen3-1.7B / TinySwallow-1.5B-Instruct | 約 0.5GB - 2GB |
| 会話文脈・同時リクエスト用 | KV Cache / 余白 | 約 20GB - 30GB |

Storage / Network / Database はインフラ運用のRCA、設定確認、手順整理を含むため、まずは長文読解・ログ解析に強い Instruct 1M 系で比較する。Coding は Coder 系、Security は長文ログ解析とCVE評価を重視し、General は軽量即答モデルを優先する。
