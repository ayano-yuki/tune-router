from __future__ import annotations

import random
import re

from config import EXTRA_TAGS, LABELS
from utils import stable_int


SEED_TEMPLATES = {
    "Storage": [
        ("Storage.ceph", "Cephの{storage_component}で{storage_symptom}が起きています。切り分け手順を整理してください。"),
        ("Storage.zfs", "ZFSの{storage_operation}が遅いです。確認すべきメトリクスと原因候補を教えて。"),
        ("Storage.nas", "{storage_protocol}の共有で{storage_symptom}が出ます。設定とネットワーク以外で見る点は?"),
        ("Storage.backup", "{storage_product}のバックアップ失敗時に確認するログと復旧手順をまとめてください。"),
    ],
    "Network": [
        ("Network.bgp", "BGPピアが{network_state}になりません。確認すべきコマンドと原因候補を教えて。"),
        ("Network.dns", "社内DNSの名前解決が{network_symptom}です。切り分け順序を整理してください。"),
        ("Network.vlan", "VLAN間通信で{network_symptom}が発生します。L2/L3の確認観点を教えて。"),
        ("Network.vpn", "{vpn_product}のVPN接続が不安定です。MTUや経路を含めて調査したいです。"),
    ],
    "Coding": [
        ("Coding.debug", "{lang}で{feature}を実装したら{error}になります。原因と修正案を教えて。"),
        ("Coding.test", "{framework}で{feature}の単体テストを書く観点を整理してください。"),
        ("Coding.sql", "{db}で{metric}を集計するSQLを作ってください。"),
        ("Coding.review", "この{lang}コードのバグになりそうな箇所をレビューしてください。"),
    ],
    "Security": [
        ("Security.cve", "{product}の{cve}について、影響範囲と一次対応を整理してください。"),
        ("Security.alert", "{source}ログに{indicator}が大量に出ています。攻撃か誤検知かを見たいです。"),
        ("Security.siem", "SIEMで{alert}が発火しました。優先度とトリアージ観点を整理してください。"),
        ("Security.rule", "{tool}向けに{attack}を検知するルールの考え方を作ってください。"),
    ],
    "Database": [
        ("Database.slow_query", "{db}のスロークエリが増えています。実行計画とインデックスの見方を教えて。"),
        ("Database.replication", "{db}のレプリケーション遅延が発生しています。確認手順を整理してください。"),
        ("Database.schema", "{db}で{db_object}を設計するときの注意点を教えて。"),
        ("Database.cache", "{cache}のメモリ使用量が急増しています。原因調査の進め方を知りたいです。"),
    ],
    "General": [
        ("General.explain", "{topic}とは何かを、初心者にも分かるように説明してください。"),
        ("General.compare", "{option_a}と{option_b}の違いを比較して、選び方を教えて。"),
        ("General.summary", "次の文章を短く要約してください。テーマは{topic}です。"),
        ("General.plan", "{goal}を進めるための現実的な段取りを作ってください。"),
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
    "storage_component": ["OSD", "MON", "MDS", "pool", "RBD", "journal"],
    "storage_symptom": ["高レイテンシ", "頻繁なdown", "容量逼迫", "I/Oエラー", "スループット低下"],
    "storage_operation": ["scrub", "snapshot", "replication", "resilver", "restore"],
    "storage_protocol": ["NFS", "SMB", "iSCSI", "S3互換API"],
    "storage_product": ["Ceph", "ZFS", "NAS", "SAN", "Object Storage"],
    "network_state": ["Established", "Idle", "Active", "Connect"],
    "network_symptom": ["タイムアウトする", "遅い", "片方向だけ疎通しない", "断続的に切れる"],
    "vpn_product": ["WireGuard", "IPsec", "OpenVPN", "AnyConnect"],
    "cloud": ["AWS", "Azure", "GCP"],
    "resource": ["VPC", "S3 bucket", "IAM role", "ALB", "Secret"],
    "operation": ["ログ収集", "バックアップ", "権限設定", "デプロイ", "ロールバック"],
    "service": ["API", "バッチ", "DB", "認証基盤", "Webフロント"],
    "db_object": ["テーブル", "インデックス", "パーティション", "マイグレーション", "ビュー"],
    "cache": ["Redis", "Memcached", "ElastiCache"],
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
                        "minimum_quality": 0.75 if label != "General" else 0.65,
                        "requires_human_review": False,
                    },
                }
            )
    rng.shuffle(records)
    return records
