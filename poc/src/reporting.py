from __future__ import annotations

import collections
from typing import Iterable

from config import LABELS, ROUTER_BASE_MODEL
from types_local import Metrics


def summarize_counts(records: Iterable[dict]) -> dict[str, int]:
    counts = collections.Counter(record["gold_label"] for record in records)
    return {label: counts.get(label, 0) for label in LABELS}


def report_markdown(
    train: list[dict],
    dev: list[dict],
    test: list[dict],
    test_metrics: Metrics | None = None,
    confusion: dict | None = None,
    mistakes: list[dict] | None = None,
) -> str:
    labels = list(LABELS)
    count_header = "| split | " + " | ".join(labels) + " | total |"
    count_align = "| --- | " + " | ".join("---:" for _ in labels) + " | ---: |"
    lines = [
        "# TuneRouter PoC Report",
        "",
        "## 方針",
        "",
        f"このPoCでは、下流アプリのRouterカテゴリに合わせた分類データをローカルJSONで管理し、`{ROUTER_BASE_MODEL}` をLoRAでFine-Tuningします。",
        "",
        "## データ件数",
        "",
        count_header,
        count_align,
    ]
    for name, rows in [("train", train), ("dev", dev), ("test", test)]:
        counts = summarize_counts(rows)
        count_cells = " | ".join(str(counts[label]) for label in labels)
        lines.append(f"| {name} | {count_cells} | {len(rows)} |")

    if test_metrics and confusion:
        lines.extend(
            [
                "",
                "## 精度",
                "",
                "| split | accuracy | macro_f1 | correct / total |",
                "| --- | ---: | ---: | ---: |",
                f"| test | {test_metrics.accuracy:.3f} | {test_metrics.macro_f1:.3f} | {test_metrics.correct} / {test_metrics.total} |",
                "",
                "## Test Confusion Matrix",
                "",
                "| actual \\ predicted | " + " | ".join(labels) + " |",
                "| --- | " + " | ".join("---:" for _ in labels) + " |",
            ]
        )
        for actual in LABELS:
            row = confusion[actual]
            cells = " | ".join(str(row[label]) for label in labels)
            lines.append(f"| {actual} | {cells} |")

        lines.extend(["", "## 誤分類例", ""])
        if not mistakes:
            lines.append("誤分類はありませんでした。実データではここに境界の悪い質問が出る想定です。")
        else:
            for mistake in mistakes[:20]:
                lines.append(
                    f"- `{mistake['gold_label']}` -> `{mistake['predicted_label']}` "
                    f"({mistake['confidence']:.3f}): {mistake['text']}"
                )

    lines.extend(
        [
            "",
            "## 次の判断",
            "",
            "- カテゴリ境界が下流アプリのRouter設定と一致しているか、実データで確認する",
            "- 必要なら運用ログやレビュー済み質問を同じJSONスキーマに追加する",
            "- Qwen2.5-0.5BのLoRA FT結果で誤分類例を見る",
            "- 精度が不足する場合は、データ境界、学習率、epoch、LoRA rank、入力テンプレートを調整する",
            "- 回答品質採点とOracleラベル生成は、複数モデル運用の価値が見えてから追加する",
        ]
    )
    return "\n".join(lines) + "\n"
