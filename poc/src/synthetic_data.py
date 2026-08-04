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


SEED_TEMPLATES["Storage"].extend(
    [
        ("storage.ceph_osd", "Cephの{ceph_component}が頻繁に{storage_failure}します。原因を切り分けたいです。"),
        ("storage.ceph_pg", "CephでPGが{pg_state}のままです。確認すべきコマンドと復旧手順を教えて。"),
        ("storage.zfs_task", "ZFSの{zfs_task}が{storage_symptom}です。確認順序を整理してください。"),
        ("storage.zfs_pool", "ZFS poolで{zfs_pool_issue}が出ています。データ保護を優先した対応を知りたい。"),
        ("storage.nas_share", "NASの{nas_feature}で{storage_symptom}が出ています。運用上の注意点を整理してください。"),
        ("storage.iscsi", "iSCSIのLUNが{storage_failure}します。multipathとパス冗長化の切り分け方を知りたい。"),
        ("storage.raid", "RAID構成で{disk_issue}が起きたときの復旧手順と避けるべき操作を教えて。"),
        ("storage.backup", "{storage_product}のバックアップが{backup_issue}です。ログ確認と復旧の流れをまとめて。"),
        ("storage.capacity", "{storage_product}の容量逼迫が近いです。短期対応と恒久対応を分けて提案してください。"),
        ("storage.performance", "{storage_product}で{io_metric}が悪化しています。ボトルネック調査の順番を教えて。"),
        ("storage.snapshot", "{storage_product}のスナップショット運用で{snapshot_issue}が起きています。見直し観点は?"),
        ("storage.migration", "{storage_product}から別ストレージへ移行します。停止時間を短くする計画を作ってください。"),
    ]
)
SEED_TEMPLATES["Network"].extend(
    [
        ("network.bgp", "BGPピアが{bgp_state}になりません。{environment}で、{impact_scope}を意識して確認すべき設定と疎通試験を整理してください。"),
        ("network.ospf", "OSPFネイバーが{ospf_state}のままです。{urgency}に動けるよう、原因候補を{answer_style}で挙げてください。"),
        ("network.dns", "社内DNSの名前解決が{network_symptom}です。{environment}でキャッシュ、権威DNS、再帰問い合わせを切り分けたい。"),
        ("network.vlan", "VLAN間通信で{network_symptom}が出ています。{impact_scope}を確認しながらL2/L3どちらの問題か見たい。"),
        ("network.vpn", "VPN接続後に{network_target}へアクセスできません。{environment}でMTUと経路の見方を{answer_style}で教えて。"),
        ("network.mtu", "{network_service}で大きなレスポンスだけ失敗します。{constraint}を守りつつMTU/MSSの確認手順を知りたい。"),
        ("network.firewall", "Firewall変更後に{network_target}への通信が失敗します。{urgency}に切り戻す前に見るべき点は?"),
        ("network.loadbalancer", "ロードバランサ配下で{network_symptom}が起きています。{impact_scope}を踏まえてヘルスチェックと経路を確認したい。"),
        ("network.packet_loss", "{network_segment}でパケットロスが出ています。{environment}で物理層から順に調査したい。"),
        ("network.dhcp", "DHCPでIPが配布されません。{impact_scope}を切り分けながらスコープ、リレー、VLANの確認順序を教えて。"),
        ("network.route", "特定サブネットだけ経路が{route_issue}です。{answer_style}でルーティングテーブルの見方を整理してください。"),
        ("network.proxy", "プロキシ経由の通信が{network_symptom}です。{constraint}を守りながらDNSとTLSを含めて切り分けたい。"),
    ]
)
SEED_TEMPLATES["Coding"].extend(
    [
        ("coding.debug", "{lang}で{feature}を実装したら{error}になります。原因と修正案を教えて。"),
        ("coding.test", "{framework}で{feature}の単体テストを書く観点を整理してください。"),
        ("coding.refactor", "この{lang}コードを保守しやすくリファクタリングする方針を提案してください。"),
        ("coding.build", "{runtime}のビルドが{error}で落ちます。依存関係と設定の確認順序を教えて。"),
        ("coding.api", "{lang}でREST APIの{api_feature}を実装するサンプルを書いてください。"),
        ("coding.sql", "{db}で{metric}を集計するSQLを作ってください。"),
        ("coding.performance", "{lang}アプリの{perf_issue}を改善したいです。プロファイルの取り方を教えて。"),
        ("coding.concurrency", "{lang}で並行処理を書いたら{concurrency_issue}が起きます。修正方針を知りたい。"),
        ("coding.frontend", "{frontend_framework}で{ui_feature}を作る実装方針を整理してください。"),
        ("coding.ci", "CIで{ci_issue}が発生しています。再現方法と調査手順を教えて。"),
        ("coding.migration", "{lang}のライブラリを{migration_target}へ移行します。互換性の注意点は?"),
        ("coding.review", "{lang}のコードレビューで{review_focus}を重点的に見る観点を教えて。"),
    ]
)
SEED_TEMPLATES["Security"].extend(
    [
        ("security.cve", "{product}の{cve}について、{impact_scope}を想定して影響範囲と一次対応を整理してください。"),
        ("security.incident", "{source}ログに{indicator}が大量に出ています。{environment}で攻撃か誤検知かを見たいです。"),
        ("security.tls", "TLS証明書更新後に{tls_symptom}が起きています。{audience}向けに確認すべき点を教えて。"),
        ("security.waf", "WAFで{attack}を防ぐルールを作るときの注意点を{answer_style}で整理してください。"),
        ("security.auth", "{auth_system}の認証ログから不審なログインを調査する観点を{output_format}で教えて。"),
        ("security.iam", "{cloud}のIAM権限が広すぎるか確認したいです。{constraint}を守る棚卸し手順を整理してください。"),
        ("security.audit", "{audit_target}の監査ログから不正操作を探す観点を{urgency}に確認できる順で教えて。"),
        ("security.malware", "{endpoint_product}でマルウェア疑いの検知が出ました。{impact_scope}を抑える初動対応をまとめて。"),
        ("security.vulnerability", "{service}に脆弱性診断の指摘が出ました。{environment}で優先度付けの方法を知りたい。"),
        ("security.network", "{network_device}で不審な通信を遮断したいです。{constraint}を前提に影響確認とルール設計を教えて。"),
        ("security.secrets", "{secret_location}に秘密情報が混入した疑いがあります。{urgency}に確認とローテーション手順を進めるには?"),
        ("security.hardening", "{os_product}のハードニングを進めます。{audience}が最初に見る設定を整理してください。"),
    ]
)
SEED_TEMPLATES["Database"].extend(
    [
        ("database.postgres", "PostgreSQLの{db_problem}を改善したいです。{environment}でEXPLAINの見方と対処を教えて。"),
        ("database.mysql", "MySQLで{db_problem}が起きています。{impact_scope}を意識してインデックスとロックの切り分け方を知りたい。"),
        ("database.redis", "Redisの{redis_problem}が発生しています。{constraint}を守りながらメモリと永続化設定を確認したい。"),
        ("database.replication", "{db_product}のレプリケーション遅延が大きいです。{answer_style}で原因候補を整理してください。"),
        ("database.backup", "{db_product}のバックアップ復元手順を{environment}向けに点検してください。"),
        ("database.connection", "{db_product}で接続数が上限に近いです。{audience}が見るべきアプリ側とDB側の確認観点は?"),
        ("database.lock", "{db_product}でロック待ちが増えています。{urgency}にブロッキングセッションを調べる方法を教えて。"),
        ("database.schema", "{db_product}で{db_object}を変更します。{constraint}を守ってダウンタイムを避ける手順を考えたい。"),
        ("database.partition", "{db_product}のパーティション設計を見直したいです。{impact_scope}を踏まえた判断基準を教えて。"),
        ("database.vacuum", "PostgreSQLでVACUUMが追いついていません。{answer_style}で原因と対策を整理してください。"),
        ("database.migration", "{db_product}のバージョンアップを計画しています。{environment}での検証項目を作ってください。"),
        ("database.observability", "{db_product}の監視で{db_metric}が悪化しています。{output_format}で見るべきメトリクスは?"),
    ]
)
SEED_TEMPLATES["General"].extend(
    [
        ("general.weather", "{place}の天気に合わせて持ち物を考えたいです。{audience}向けに{output_format}でお願いします。"),
        ("general.travel", "{place}へ旅行するときの大まかな段取りを作ってください。{constraint}も考慮してください。"),
        ("general.summary", "次の文章を短く要約してください。テーマは{topic}です。{audience}向けにしてください。"),
        ("general.compare", "{option_a}と{option_b}の違いを初心者向けに比較してください。{output_format}でお願いします。"),
        ("general.recipe", "{dish}を作るときの材料と手順を簡単に教えてください。{constraint}に合わせたいです。"),
        ("general.email", "{email_purpose}のメール文面を丁寧な日本語で作ってください。{audience}に送る想定です。"),
        ("general.schedule", "{personal_event}までにやることを週ごとの計画にしてください。{urgency}に進めたいです。"),
        ("general.study", "{study_topic}を学ぶための入門ロードマップを作ってください。{output_format}で見たいです。"),
        ("general.shopping", "{shopping_item}を選ぶときの比較ポイントを教えてください。{constraint}を重視します。"),
        ("general.health", "{habit_goal}を続けるための無理のない習慣化プランを考えて。{audience}向けにしてください。"),
        ("general.writing", "{writing_theme}について短い説明文を書いてください。{output_format}にしてください。"),
        ("general.brainstorm", "{creative_goal}のアイデアを10個出してください。{constraint}も踏まえてください。"),
    ]
)


SEED_TEMPLATES["Storage"] = [
    (
        template_id,
        template
        if any(marker in template for marker in ("{environment}", "{answer_style}", "{urgency}", "{impact_scope}", "{constraint}", "{audience}", "{output_format}"))
        else f"{template} {{environment}}で、{{answer_style}}にしてください。",
    )
    for template_id, template in SEED_TEMPLATES["Storage"]
]
SEED_TEMPLATES["Network"] = [
    (
        template_id,
        template
        if any(marker in template for marker in ("{environment}", "{answer_style}", "{urgency}", "{impact_scope}", "{constraint}", "{audience}", "{output_format}"))
        else f"{template} {{environment}}で、{{answer_style}}にしてください。",
    )
    for template_id, template in SEED_TEMPLATES["Network"]
]
SEED_TEMPLATES["Coding"] = [
    (
        template_id,
        template
        if any(marker in template for marker in ("{environment}", "{answer_style}", "{urgency}", "{impact_scope}", "{constraint}", "{audience}", "{output_format}"))
        else f"{template} {{environment}}で、{{answer_style}}にしてください。",
    )
    for template_id, template in SEED_TEMPLATES["Coding"]
]
SEED_TEMPLATES["Security"] = [
    (
        template_id,
        template
        if any(marker in template for marker in ("{environment}", "{answer_style}", "{urgency}", "{impact_scope}", "{constraint}", "{audience}", "{output_format}"))
        else f"{template} {{environment}}で、{{answer_style}}にしてください。",
    )
    for template_id, template in SEED_TEMPLATES["Security"]
]
SEED_TEMPLATES["Database"] = [
    (
        template_id,
        template
        if any(marker in template for marker in ("{environment}", "{answer_style}", "{urgency}", "{impact_scope}", "{constraint}", "{audience}", "{output_format}"))
        else f"{template} {{environment}}で、{{answer_style}}にしてください。",
    )
    for template_id, template in SEED_TEMPLATES["Database"]
]
SEED_TEMPLATES["General"] = [
    (
        template_id,
        template
        if any(marker in template for marker in ("{environment}", "{answer_style}", "{urgency}", "{impact_scope}", "{constraint}", "{audience}", "{output_format}"))
        else f"{template} {{environment}}で、{{answer_style}}にしてください。",
    )
    for template_id, template in SEED_TEMPLATES["General"]
]

SEED_TEMPLATES["Network"] = [
    (template_id, template if "{output_format}" in template else f"{template} 結果は{{output_format}}でまとめてください。")
    for template_id, template in SEED_TEMPLATES["Network"]
]
SEED_TEMPLATES["Security"] = [
    (template_id, template if "{output_format}" in template else f"{template} 結果は{{output_format}}でまとめてください。")
    for template_id, template in SEED_TEMPLATES["Security"]
]
SEED_TEMPLATES["Database"] = [
    (template_id, template if "{output_format}" in template else f"{template} 結果は{{output_format}}でまとめてください。")
    for template_id, template in SEED_TEMPLATES["Database"]
]
SEED_TEMPLATES["General"] = [
    (template_id, template if "{output_format}" in template else f"{template} 結果は{{output_format}}でまとめてください。")
    for template_id, template in SEED_TEMPLATES["General"]
]


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


VALUE_BANK["lang"].extend(["C#", "Kotlin"])
VALUE_BANK["feature"].append("通知処理")
VALUE_BANK["error"].append("依存解決エラー")
VALUE_BANK["framework"].append("RSpec")
VALUE_BANK["db"].append("DuckDB")
VALUE_BANK["product"].append("OpenSSH")
VALUE_BANK["cve"].append("CVE-2023-4966")
VALUE_BANK["source"].append("EDR")
VALUE_BANK["indicator"].append("異常な失敗ログイン")
VALUE_BANK["attack"].append("パストラバーサル")
VALUE_BANK["storage_symptom"].extend(["途中で止まる", "非常に遅い", "エラーになる", "容量不足になる", "片系だけ失敗する"])
VALUE_BANK["storage_product"].extend(["MinIO", "TrueNAS"])
VALUE_BANK["network_symptom"].extend(["不安定", "片方向だけ通らない", "断続的に失敗する"])
VALUE_BANK["service"].extend(["Webアプリ", "管理画面", "ファイル共有"])
VALUE_BANK["topic"].extend(["読書メモ", "英語学習"])
VALUE_BANK["option_a"].append("朝型生活")
VALUE_BANK["option_b"].append("夜型生活")
VALUE_BANK["ceph_component"] = ["osd", "mon", "mgr", "mds", "placement group", "crush map", "rbd"]
VALUE_BANK["storage_failure"] = ["down", "timeout", "read-only", "遅延", "認識されない状態", "I/Oエラー"]
VALUE_BANK["pg_state"] = ["stuck inactive", "degraded", "undersized", "peering", "backfill_wait"]
VALUE_BANK["zfs_task"] = ["スクラブ", "resilver", "snapshot削除", "pool import", "send/receive", "圧縮設定変更"]
VALUE_BANK["zfs_pool_issue"] = ["checksum error", "容量逼迫", "import失敗", "断片化", "デバイス欠落"]
VALUE_BANK["nas_feature"] = ["NFS共有", "SMB共有", "スナップショット", "クォータ", "レプリケーション", "ACL設定"]
VALUE_BANK["disk_issue"] = ["ディスク故障", "リビルド失敗", "ライトエラー", "不良セクタ増加", "コントローラ交換"]
VALUE_BANK["backup_issue"] = ["失敗します", "長時間化しています", "世代管理が崩れています", "復元検証でエラーになります"]
VALUE_BANK["io_metric"] = ["read latency", "write latency", "IOPS", "throughput", "queue depth"]
VALUE_BANK["snapshot_issue"] = ["削除が遅い", "容量を圧迫している", "保持世代が多すぎる", "復元に失敗する"]
VALUE_BANK["bgp_state"] = ["Established", "Idle", "Active", "OpenConfirm", "Connect"]
VALUE_BANK["ospf_state"] = ["Init", "2-Way", "ExStart", "Exchange", "Fullにならない状態"]
VALUE_BANK["network_target"] = ["社内DNS", "Gitサーバー", "監視サーバー", "管理画面", "データベース", "SaaS"]
VALUE_BANK["network_service"] = ["API通信", "ファイル転送", "VPN通信", "Webアクセス", "DB接続"]
VALUE_BANK["network_segment"] = ["拠点間VPN", "サーバーセグメント", "DMZ", "無線LAN", "管理ネットワーク"]
VALUE_BANK["route_issue"] = ["消えています", "非対称になっています", "意図しない経路へ流れます", "冗長経路へ切り替わりません"]
VALUE_BANK["runtime"] = ["Node.js", "Go", "Rust", "Docker", "Vite", "TypeScript", "JVM"]
VALUE_BANK["api_feature"] = ["認証", "ページネーション", "エラーハンドリング", "入力バリデーション", "リトライ"]
VALUE_BANK["perf_issue"] = ["CPU使用率", "メモリ使用量", "p95レイテンシ", "起動時間", "DB待ち"]
VALUE_BANK["concurrency_issue"] = ["デッドロック", "競合", "goroutineリーク", "race condition", "順序不定の失敗"]
VALUE_BANK["frontend_framework"] = ["React", "Vue", "Svelte", "Next.js", "Vite"]
VALUE_BANK["ui_feature"] = ["検索UI", "フォーム", "チャート", "ダッシュボード", "ファイルアップロード"]
VALUE_BANK["ci_issue"] = ["テスト失敗", "キャッシュ不整合", "lintエラー", "Docker build失敗", "環境変数不足"]
VALUE_BANK["migration_target"] = ["新メジャーバージョン", "別SDK", "ESM", "新API", "別ORM"]
VALUE_BANK["review_focus"] = ["例外処理", "境界値", "可読性", "性能", "セキュリティ"]
VALUE_BANK["tls_symptom"] = ["証明書エラー", "ハンドシェイク失敗", "古い暗号スイート警告", "一部端末だけ接続失敗"]
VALUE_BANK["auth_system"] = ["VPN", "SSO", "Active Directory", "OIDC", "管理画面", "LDAP"]
VALUE_BANK["audit_target"] = ["CloudTrail", "監査ログ", "管理画面", "IAM操作", "DB監査ログ"]
VALUE_BANK["endpoint_product"] = ["EDR", "Defender", "CrowdStrike", "アンチウイルス", "端末管理ツール"]
VALUE_BANK["network_device"] = ["Firewall", "WAF", "IDS", "Proxy", "VPN Gateway"]
VALUE_BANK["secret_location"] = ["Gitリポジトリ", "CIログ", "コンテナイメージ", "設定ファイル", "チケット"]
VALUE_BANK["os_product"] = ["Linuxサーバー", "Windows Server", "Ubuntu", "RHEL", "コンテナホスト"]
VALUE_BANK["db_problem"] = ["スロークエリ", "デッドロック", "VACUUM遅延", "接続数枯渇", "ロック待ち"]
VALUE_BANK["redis_problem"] = ["evicted_keys増加", "永続化失敗", "メモリ逼迫", "レイテンシ悪化", "レプリカ遅延"]
VALUE_BANK["db_product"] = ["PostgreSQL", "MySQL", "Oracle", "MongoDB", "Redis", "MariaDB"]
VALUE_BANK["db_metric"] = ["接続数", "ロック待ち", "キャッシュヒット率", "WAL量", "レプリケーション遅延"]
VALUE_BANK["place"] = ["京都", "札幌", "沖縄", "金沢", "福岡", "東京", "大阪"]
VALUE_BANK["dish"] = ["カレー", "味噌汁", "パスタ", "オムライス", "サラダ", "親子丼"]
VALUE_BANK["email_purpose"] = ["日程調整", "お礼", "依頼", "謝罪", "問い合わせ"]
VALUE_BANK["personal_event"] = ["引っ越し", "資格試験", "旅行", "発表", "部屋の片付け"]
VALUE_BANK["study_topic"] = ["統計", "英会話", "簿記", "機械学習", "デザイン"]
VALUE_BANK["shopping_item"] = ["ノートPC", "椅子", "イヤホン", "炊飯器", "モニター"]
VALUE_BANK["habit_goal"] = ["早寝", "散歩", "読書", "筋トレ", "家計管理"]
VALUE_BANK["writing_theme"] = ["自己紹介", "イベント案内", "サービス説明", "読書感想", "議事メモ"]
VALUE_BANK["creative_goal"] = ["新しいアプリ", "ブログ記事", "勉強会テーマ", "チーム名", "プレゼント"]
VALUE_BANK["environment"] = ["小規模環境", "本番環境", "検証環境", "複数拠点環境", "クラウド併用環境", "オンプレ環境", "リモート勤務環境", "社内向け環境"]
VALUE_BANK["answer_style"] = ["確認コマンド中心", "チェックリスト形式", "原因候補の優先度順", "初心者にも分かる形", "運用手順として使える形", "短期対応と恒久対応を分ける形"]
VALUE_BANK["urgency"] = ["今日中", "今週中", "次回メンテナンスまで", "影響範囲確認後", "段階的", "まず30分で", "翌営業日まで"]
VALUE_BANK["impact_scope"] = ["一部ユーザーへの影響", "全社影響", "特定拠点だけの影響", "夜間バッチへの影響", "管理者操作への影響", "外部公開サービスへの影響"]
VALUE_BANK["constraint"] = ["停止時間を最小にして", "安全にロールバックできる形で", "追加コストを抑えて", "利用者影響を抑えて", "ログだけで確認できる範囲で", "権限変更を最小にして"]
VALUE_BANK["audience"] = ["新人担当者", "運用チーム", "非エンジニア", "管理者", "レビュー担当者", "意思決定者"]
VALUE_BANK["output_format"] = ["箇条書き", "表形式", "手順書形式", "短い説明", "比較表", "優先順位付きリスト"]


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
