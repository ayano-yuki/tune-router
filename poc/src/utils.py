from __future__ import annotations

import hashlib


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16)


def enable_system_cert_store() -> None:
    try:
        import truststore
    except ImportError:
        return
    truststore.inject_into_ssl()
