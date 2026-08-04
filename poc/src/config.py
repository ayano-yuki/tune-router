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
    "Storage": {
        "dataset": "databricks/databricks-dolly-15k",
        "split": "train",
        "license": "cc-by-sa-3.0",
        "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "prompt_fields": ["instruction", "prompt", "question", "context", "text"],
    },
    "Network": {
        "dataset": "databricks/databricks-dolly-15k",
        "split": "train",
        "license": "cc-by-sa-3.0",
        "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "prompt_fields": ["instruction", "prompt", "question", "context", "text"],
    },
    "Coding": {
        "dataset": "SoyMaycol/CodeInstruct-20K",
        "split": "train",
        "license": "cc-by-4.0",
        "url": "https://huggingface.co/datasets/SoyMaycol/CodeInstruct-20K",
        "prompt_fields": ["question", "prompt", "instruction", "text"],
    },
    "Security": {
        "dataset": "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset",
        "split": "train",
        "license": "apache-2.0",
        "url": "https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset",
        "prompt_fields": ["instruction", "prompt", "question", "input", "text"],
    },
    "Database": {
        "dataset": "databricks/databricks-dolly-15k",
        "split": "train",
        "license": "cc-by-sa-3.0",
        "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "prompt_fields": ["instruction", "prompt", "question", "context", "text"],
    },
    "General": {
        "dataset": "databricks/databricks-dolly-15k",
        "split": "train",
        "license": "cc-by-sa-3.0",
        "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "prompt_fields": ["instruction", "prompt", "question", "context", "text"],
    },
}

EXTRA_TAGS = {
    "Storage": ["ceph", "zfs", "raid", "nas", "nfs", "iscsi"],
    "Network": ["bgp", "ospf", "vlan", "dns", "dhcp", "vpn"],
    "Coding": ["implementation", "debug", "test", "review", "build"],
    "Security": ["cve", "incident", "tls", "waf", "auth", "hardening"],
    "Database": ["postgresql", "mysql", "index", "query", "replication", "redis"],
    "General": ["fallback", "explain", "summary", "planning", "compare"],
}
