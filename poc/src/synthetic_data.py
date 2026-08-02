from __future__ import annotations

import random
import re

from config import EXTRA_TAGS, LABELS
from utils import stable_int


SEED_TEMPLATES = {
    "code": [
        ("code.debug", "{lang}で{feature}を実装したら{error}になります。原因と修正案を教えて。"),
        ("code.test", "{framework}で{feature}の単体テストを書く観点を整理してください。"),
        ("code.sql", "{db}で{metric}を集計するSQLを作ってください。"),
        ("code.review", "この{lang}コードのバグになりそうな箇所をレビューしてください。"),
    ],
    "security": [
        ("security.cve", "{product}の{cve}について、影響範囲と一次対応を整理してください。"),
        ("security.alert", "{source}ログに{indicator}が大量に出ています。攻撃か誤検知かを見たいです。"),
        ("security.siem", "SIEMで{alert}が発火しました。優先度とトリアージ観点を整理してください。"),
        ("security.rule", "{tool}向けに{attack}を検知するルールの考え方を作ってください。"),
    ],
    "iac_text": [
        ("iac_text.terraform", "Terraformで{cloud}の{resource}を作る構成例と注意点を教えて。"),
        ("iac_text.drift", "Terraformのstate driftを検出する手順と運用上の注意点を整理してください。"),
        ("iac_text.kubernetes", "Kubernetesの{resource}で{operation}するYAMLを作ってください。"),
        ("iac_text.runbook", "{service}障害時のrunbookを、確認順序が分かる形で作ってください。"),
    ],
    "general": [
        ("general.explain", "{topic}とは何かを、初心者にも分かるように説明してください。"),
        ("general.compare", "{option_a}と{option_b}の違いを比較して、選び方を教えて。"),
        ("general.summary", "次の文章を短く要約してください。テーマは{topic}です。"),
        ("general.plan", "{goal}を進めるための現実的な段取りを作ってください。"),
    ],
}


VALUE_BANK = {
    "lang": ["Python", "TypeScript", "Go", "Rust", "Java", "React", "Node.js"],
    "feature": ["認証処理", "CSVインポート", "非同期ジョブ", "検索機能", "キャッシュ"],
    "error": ["NullReferenceException", "型エラー", "timeout", "500エラー", "メモリリーク"],
    "framework": ["pytest", "Jest", "Vitest", "Go test", "JUnit", "Playwright"],
    "db": ["PostgreSQL", "MySQL", "BigQuery", "SQLite", "Snowflake"],
    "metric": ["月次売上", "エラー率", "ユーザー継続率", "p95レイテンシ", "在庫推移"],
    "product": ["OpenSSL", "Apache Struts", "nginx", "GitLab", "Windows Server", "Kubernetes"],
    "cve": ["CVE-2024-3094", "CVE-2023-34362", "CVE-2021-44228", "CVE-2022-22965"],
    "source": ["nginx access", "CloudTrail", "Windows Event", "VPC Flow", "IDS"],
    "indicator": ["不審なUser-Agent", "401", "大量のPOST", "海外IP", "短時間のスキャン"],
    "alert": ["Impossible Travel", "Privilege Escalation", "Suspicious Login", "Data Exfiltration"],
    "tool": ["Sigma", "YARA", "Suricata", "Splunk SPL", "KQL"],
    "attack": ["SQLインジェクション", "XSS", "権限昇格", "横展開", "credential stuffing"],
    "cloud": ["AWS", "Azure", "GCP"],
    "resource": ["VPC", "S3 bucket", "IAM role", "ALB", "Secret"],
    "operation": ["ログ収集", "バックアップ", "権限設定", "デプロイ", "ロールバック"],
    "service": ["API", "バッチ", "DB", "認証基盤", "Webフロント"],
    "topic": ["生成AI", "ゼロトラスト", "在庫管理", "プロジェクト管理", "データ分析"],
    "option_a": ["Python", "Go", "SaaS", "内製", "RAG", "Fine-Tuning"],
    "option_b": ["TypeScript", "Rust", "OSS", "外注", "プロンプト改善", "ルールベース"],
    "goal": ["読みやすく", "高速化", "保守", "監査", "本番運用", "再利用"],
}


def fill_template(template: str, rng: random.Random) -> str:
    fields = re.findall(r"{([a-z_]+)}", template)
    values = {field: rng.choice(VALUE_BANK[field]) for field in fields}
    return template.format(**values)


def build_synthetic_dataset(per_label: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    labels = list(LABELS)
    for label, templates in SEED_TEMPLATES.items():
        for index in range(per_label):
            template_id, template = templates[index % len(templates)]
            text = fill_template(template, rng)
            variant = stable_int(f"{seed}:{label}:{index}:{text}") % 100_000
            records.append(
                {
                    "question_id": f"poc-{label}-{index + 1:06d}",
                    "text": text,
                    "gold_label": label,
                    "label_id": labels.index(label),
                    "target_model": LABELS[label]["target_model"],
                    "tags": [label, rng.choice(EXTRA_TAGS[label])],
                    "importance": rng.choice(["normal", "normal", "high", "critical"]),
                    "source": "synthetic_poc",
                    "template_id": template_id,
                    "split_group": f"{template_id}:{variant % 20}",
                    "split": "",
                    "rubric": {
                        "minimum_quality": 0.75 if label != "general" else 0.65,
                        "requires_human_review": False,
                    },
                }
            )
    rng.shuffle(records)
    return records
