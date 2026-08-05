ROUTER_BASE_MODEL = "Qwen/Qwen2.5-0.5B"

LABELS = {
    "Storage": {
        "target_model": "storage-specialist",
        "description": "storage platforms, Ceph, ZFS, RAID, NAS, SAN, iSCSI, NFS",
    },
    "Network": {
        "target_model": "network-specialist",
        "description": "networking, BGP, OSPF, VLAN, DNS, DHCP, VPN, L2/L3 reachability",
    },
    "Coding": {
        "target_model": "coding-specialist",
        "description": "software implementation, debugging, refactoring, builds, tests",
    },
    "Security": {
        "target_model": "security-specialist",
        "description": "vulnerabilities, incidents, authentication, TLS, WAF, hardening",
    },
    "Database": {
        "target_model": "database-specialist",
        "description": "PostgreSQL, MySQL, Oracle, Redis, MongoDB, query tuning, replication",
    },
    "General": {
        "target_model": "general-fallback",
        "description": "fallback for questions outside the configured specialist categories",
    },
}

OSS_SOURCES = {
    "Storage": [
        {
            "name": "stackexchange-storage",
            "dataset": "Stack Exchange API",
            "loader": "stackexchange_api",
            "split": "train",
            "license": "cc-by-sa-4.0",
            "url": "https://api.stackexchange.com/docs/questions",
            "stackexchange_sites": [
                {"site": "serverfault", "tags": ["storage", "nfs", "raid", "zfs", "ceph", "iscsi", "nas", "backup"]},
                {"site": "unix", "tags": ["storage", "nfs", "raid", "zfs", "ceph"]},
                {"site": "superuser", "tags": ["storage", "backup", "raid", "nas"]},
            ],
            "combine_fields": ["title", "body"],
            "prompt_fields": ["title", "body"],
            "question_template": "このストレージ関連の質問について、原因切り分けや設計上の注意点を教えて:\n{text}",
        },
        {
            "name": "dolly-storage-fallback",
            "dataset": "databricks/databricks-dolly-15k",
            "split": "train",
            "license": "cc-by-sa-3.0",
            "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
            "prompt_fields": ["instruction", "prompt", "question", "context", "text"],
        },
    ],
    "Network": [
        {
            "name": "stackexchange-network",
            "dataset": "Stack Exchange API",
            "loader": "stackexchange_api",
            "split": "train",
            "license": "cc-by-sa-4.0",
            "url": "https://api.stackexchange.com/docs/questions",
            "stackexchange_sites": [
                {"site": "networkengineering", "tags": ["bgp", "ospf", "vlan", "vpn", "dns", "routing", "switching"]},
                {"site": "serverfault", "tags": ["networking", "dns", "vpn", "dhcp", "vlan", "routing", "firewall"]},
                {"site": "superuser", "tags": ["networking", "dns", "vpn", "router"]},
            ],
            "combine_fields": ["title", "body"],
            "prompt_fields": ["title", "body"],
            "question_template": "このネットワーク関連の質問について、設定確認や原因切り分けの観点を教えて:\n{text}",
        },
        {
            "name": "netconfeval-config-generation",
            "dataset": "NetConfEval/NetConfEval",
            "config": "Configuration Generation",
            "split": "train",
            "license": "mit",
            "url": "https://huggingface.co/datasets/NetConfEval/NetConfEval",
            "prompt_fields": ["prompt"],
            "question_template": "このネットワーク構成生成タスクで、要件を満たす設定方針を教えて:\n{text}",
        },
    ],
    "Coding": [
        {
            "name": "magicoder-oss-instruct",
            "dataset": "ise-uiuc/Magicoder-OSS-Instruct-75K",
            "split": "train",
            "license": "mit",
            "url": "https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K",
            "prompt_fields": ["problem"],
            "question_template": "この実装タスクを解いてください:\n{text}",
        },
        {
            "name": "codeinstruct-20k",
            "dataset": "SoyMaycol/CodeInstruct-20K",
            "split": "train",
            "license": "cc-by-4.0",
            "url": "https://huggingface.co/datasets/SoyMaycol/CodeInstruct-20K",
            "prompt_fields": ["question", "prompt", "instruction", "text"],
        },
    ],
    "Security": [
        {
            "name": "trendyol-cybersecurity-instruct",
            "dataset": "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset",
            "split": "train",
            "license": "apache-2.0",
            "url": "https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset",
            "prompt_fields": ["user", "instruction", "prompt", "question", "input", "text"],
        },
    ],
    "Database": [
        {
            "name": "gretel-synthetic-text-to-sql",
            "dataset": "gretelai/synthetic_text_to_sql",
            "split": "train",
            "license": "apache-2.0",
            "url": "https://huggingface.co/datasets/gretelai/synthetic_text_to_sql",
            "prompt_fields": ["sql_prompt"],
            "question_template": "このデータベース/SQLの依頼に答えてください:\n{text}",
        },
        {
            "name": "sql-create-context",
            "dataset": "b-mc2/sql-create-context",
            "split": "train",
            "license": "cc-by-4.0",
            "url": "https://huggingface.co/datasets/b-mc2/sql-create-context",
            "prompt_fields": ["question"],
        },
        {
            "name": "stackexchange-database",
            "dataset": "Stack Exchange API",
            "loader": "stackexchange_api",
            "split": "train",
            "license": "cc-by-sa-4.0",
            "url": "https://api.stackexchange.com/docs/questions",
            "stackexchange_sites": [
                {"site": "dba", "tags": ["postgresql", "mysql", "index", "performance", "replication", "deadlock"]},
                {"site": "serverfault", "tags": ["postgresql", "mysql", "database", "sql-server", "mongodb"]},
            ],
            "combine_fields": ["title", "body"],
            "prompt_fields": ["title", "body"],
            "question_template": "このデータベース運用の質問について、調査やチューニングの観点を教えて:\n{text}",
        },
    ],
    "General": [
        {
            "name": "dolly-general",
            "dataset": "databricks/databricks-dolly-15k",
            "split": "train",
            "license": "cc-by-sa-3.0",
            "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
            "prompt_fields": ["instruction", "prompt", "question", "context", "text"],
        },
    ],
}

EXTRA_TAGS = {
    "Storage": ["ceph", "zfs", "raid", "nas", "nfs", "iscsi"],
    "Network": ["bgp", "ospf", "vlan", "dns", "dhcp", "vpn"],
    "Coding": ["implementation", "debug", "test", "review", "build"],
    "Security": ["cve", "incident", "tls", "waf", "auth", "hardening"],
    "Database": ["postgresql", "mysql", "index", "query", "replication", "redis"],
    "General": ["fallback", "explain", "summary", "planning", "compare"],
}
