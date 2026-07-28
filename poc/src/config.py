ROUTER_BASE_MODEL = "Qwen/Qwen2.5-0.5B"

LABELS = {
    "code": {
        "target_model": "qwen2.5-coder-7b",
        "description": "code generation, debugging, SQL, API implementation",
    },
    "security_log": {
        "target_model": "llama-3.1-8b-security-log",
        "description": "CVE, security logs, detections, incident triage",
    },
    "iac_text": {
        "target_model": "qwen2.5-14b-iac-text",
        "description": "IaC, cloud config, Kubernetes, technical writing",
    },
    "general": {
        "target_model": "general-small-or-default",
        "description": "general questions, explanations, summaries, casual tasks",
    },
}

OSS_SOURCES = {
    "code": {
        "dataset": "SoyMaycol/CodeInstruct-20K",
        "split": "train",
        "license": "cc-by-4.0",
        "url": "https://huggingface.co/datasets/SoyMaycol/CodeInstruct-20K",
        "prompt_fields": ["question", "prompt", "instruction", "text"],
    },
    "security_log": {
        "dataset": "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset",
        "split": "train",
        "license": "apache-2.0",
        "url": "https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset",
        "prompt_fields": ["instruction", "prompt", "question", "input", "text"],
    },
    "iac_text": {
        "dataset": "galcan/terraform_sec",
        "split": "train",
        "license": "apache-2.0",
        "url": "https://huggingface.co/datasets/galcan/terraform_sec",
        "prompt_fields": ["instruction", "prompt", "question", "input", "text", "code", "terraform_code"],
    },
    "general": {
        "dataset": "databricks/databricks-dolly-15k",
        "split": "train",
        "license": "cc-by-sa-3.0",
        "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "prompt_fields": ["instruction", "prompt", "question", "context", "text"],
    },
}

EXTRA_TAGS = {
    "code": ["implementation", "debug", "test", "review", "sql"],
    "security_log": ["cve", "log", "incident", "detection", "triage"],
    "iac_text": ["iac", "cloud", "kubernetes", "docs", "runbook"],
    "general": ["explain", "summary", "writing", "planning", "compare"],
}
