from __future__ import annotations

import random
import re

from config import EXTRA_TAGS, LABELS
from utils import stable_int


SEED_TEMPLATES = {
    "Storage": [
        ("storage.ceph", "Cephの{ceph_component}が頻繁に{storage_failure}する原因を切り分けたい。"),
        ("storage.zfs", "ZFSの{zfs_task}が{storage_symptom}です。確認すべきコマンドと順序を教えて。"),
        ("storage.nas", "NASの{nas_feature}で{storage_symptom}が出ています。運用上の注意点を整理してください。"),
        ("storage.iscsi", "iSCSIのLUNが{storage_failure}します。multipathとネットワークの切り分け方を知りたい。"),
        ("storage.raid", "RAID構成で{disk_issue}が起きたときの復旧手順と避けるべき操作を教えて。"),
    ],
    "Network": [
        ("network.bgp", "BGPピアが{bgp_state}になりません。確認すべき設定と疎通試験を整理してください。"),
        ("network.dns", "社内DNSの名前解決が{network_symptom}です。キャッシュ、権威DNS、再帰問い合わせを切り分けたい。"),
        ("network.vlan", "VLAN間通信で{network_symptom}が出ています。L2/L3どちらの問題か確認したい。"),
        ("network.vpn", "VPN接続後に{network_target}へアクセスできません。MTUと経路の見方を教えて。"),
        ("network.ospf", "OSPFネイバーが{ospf_state}のままです。原因候補を優先度順に挙げてください。"),
    ],
    "Coding": [
        ("coding.debug", "{lang}で{feature}を実装したら{error}になります。原因と修正案を教えて。"),
        ("coding.test", "{framework}で{feature}の単体テストを書く観点を整理してください。"),
        ("coding.refactor", "この{lang}コードを保守しやすくリファクタリングする方針を提案してください。"),
        ("coding.build", "{runtime}のビルドが{error}で落ちます。依存関係と設定の確認順序を教えて。"),
        ("coding.api", "{lang}でREST APIの{api_feature}を実装するサンプルを書いてください。"),
    ],
    "Security": [
        ("security.cve", "{product}の{cve}について、影響範囲と一次対応を整理してください。"),
        ("security.incident", "{source}ログに{indicator}が大量に出ています。攻撃か誤検知かを見たいです。"),
        ("security.tls", "TLS証明書更新後に{tls_symptom}が起きています。確認すべき点を教えて。"),
        ("security.waf", "WAFで{attack}を防ぐルールを作るときの注意点を整理してください。"),
        ("security.auth", "{auth_system}の認証ログから不審なログインを調査する観点を教えて。"),
    ],
    "Database": [
        ("database.postgres", "PostgreSQLの{db_problem}を改善したいです。EXPLAINの見方と対処を教えて。"),
        ("database.mysql", "MySQLで{db_problem}が起きています。インデックスとロックの切り分け方を知りたい。"),
        ("database.redis", "Redisの{redis_problem}が発生しています。メモリと永続化設定を確認したい。"),
        ("database.replication", "{db_product}のレプリケーション遅延が大きいです。原因候補を整理してください。"),
        ("database.backup", "{db_product}のバックアップ復元手順を本番運用向けに点検してください。"),
    ],
    "General": [
        ("general.weather", "明日の天気に合わせて持ち物を考えたいです。"),
        ("general.travel", "{place}へ旅行するときの大まかな段取りを作ってください。"),
        ("general.summary", "次の文章を短く要約してください。テーマは{topic}です。"),
        ("general.compare", "{option_a}と{option_b}の違いを初心者向けに比較してください。"),
        ("general.recipe", "{dish}を作るときの材料と手順を簡単に教えてください。"),
    ],
}


VALUE_BANK = {
    "ceph_component": ["osd", "mon", "mgr", "mds", "placement group"],
    "storage_failure": ["down", "timeout", "read-only", "遅延", "認識されない状態"],
    "zfs_task": ["スクラブ", "resilver", "snapshot削除", "pool import", "send/receive"],
    "storage_symptom": ["途中で止まる", "非常に遅い", "エラーになる", "容量不足になる", "片系だけ失敗する"],
    "nas_feature": ["NFS共有", "SMB共有", "スナップショット", "クォータ", "レプリケーション"],
    "disk_issue": ["ディスク故障", "リビルド失敗", "ライトエラー", "不良セクタ増加", "コントローラ交換"],
    "bgp_state": ["Established", "Idle", "Active", "OpenConfirm", "Connect"],
    "network_symptom": ["遅い", "不安定", "片方向だけ通らない", "タイムアウトする", "断続的に失敗する"],
    "network_target": ["社内DNS", "Gitサーバー", "監視サーバー", "管理画面", "データベース"],
    "ospf_state": ["Init", "2-Way", "ExStart", "Exchange", "Fullにならない状態"],
    "lang": ["Python", "TypeScript", "Go", "Rust", "Java", "React", "Node.js"],
    "feature": ["認証処理", "CSVインポート", "非同期ジョブ", "検索機能", "キャッシュ"],
    "error": ["NullReferenceException", "型エラー", "timeout", "500エラー", "メモリリーク"],
    "framework": ["pytest", "Jest", "Vitest", "Go test", "JUnit", "Playwright"],
    "runtime": ["Node.js", "Go", "Rust", "Docker", "Vite", "TypeScript"],
    "api_feature": ["認証", "ページネーション", "エラーハンドリング", "入力バリデーション", "リトライ"],
    "product": ["OpenSSL", "Apache Struts", "nginx", "GitLab", "Windows Server", "Kubernetes"],
    "cve": ["CVE-2024-3094", "CVE-2023-34362", "CVE-2021-44228", "CVE-2022-22965"],
    "source": ["nginx access", "CloudTrail", "Windows Event", "VPC Flow", "IDS"],
    "indicator": ["不審なUser-Agent", "401", "大量のPOST", "海外IP", "短時間のスキャン"],
    "tls_symptom": ["証明書エラー", "ハンドシェイク失敗", "古い暗号スイート警告", "一部端末だけ接続失敗"],
    "attack": ["SQLインジェクション", "XSS", "権限昇格", "横展開", "credential stuffing"],
    "auth_system": ["VPN", "SSO", "Active Directory", "OIDC", "管理画面"],
    "db_problem": ["スロークエリ", "デッドロック", "VACUUM遅延", "接続数枯渇", "ロック待ち"],
    "redis_problem": ["evicted_keys増加", "永続化失敗", "メモリ逼迫", "レイテンシ悪化", "レプリカ遅延"],
    "db_product": ["PostgreSQL", "MySQL", "Oracle", "MongoDB", "Redis"],
    "place": ["京都", "札幌", "沖縄", "金沢", "福岡"],
    "topic": ["生成AI", "在庫管理", "プロジェクト管理", "データ分析", "読書メモ"],
    "option_a": ["Python", "Go", "SaaS", "内製", "RAG", "Fine-Tuning"],
    "option_b": ["TypeScript", "Rust", "OSS", "外注", "プロンプト改善", "ルールベース"],
    "dish": ["カレー", "味噌汁", "パスタ", "オムライス", "サラダ"],
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
                    "question_id": f"poc-{label.lower()}-{index + 1:06d}",
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
                        "minimum_quality": 0.65 if label == "General" else 0.75,
                        "requires_human_review": False,
                    },
                }
            )
    rng.shuffle(records)
    return records
