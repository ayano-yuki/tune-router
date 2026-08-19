from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import time
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


class ArtifactError(RuntimeError):
    """Base error for artifact integrity and persistence failures."""


class ArtifactLockTimeout(ArtifactError):
    """Raised when an artifact lock cannot be acquired before its deadline."""


class ArtifactConflict(ArtifactError):
    """Raised when a compare-and-swap revision does not match."""


class ArtifactValidationError(ArtifactError):
    """Raised when an artifact envelope or digest is invalid."""


class ArtifactSignatureError(ArtifactError):
    """Raised when signing or signature verification fails."""


class ArtifactLock(AbstractContextManager["ArtifactLock"]):
    def __init__(self, target: Path, timeout: float = 10.0, poll_interval: float = 0.05) -> None:
        if timeout < 0 or poll_interval <= 0:
            raise ValueError("lock timeout must be non-negative and poll interval must be positive")
        self.target = Path(target)
        self.lock_path = self.target.with_name(f"{self.target.name}.lock")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self._handle = None

    def __enter__(self) -> "ArtifactLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                _lock_file(handle)
                self._handle = handle
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise ArtifactLockTimeout("artifact lock acquisition timed out") from None
                time.sleep(min(self.poll_interval, max(0.0, deadline - time.monotonic())))

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        del exc_type, exc_value, traceback
        if self._handle is not None:
            try:
                _unlock_file(self._handle)
            finally:
                self._handle.close()
                self._handle = None


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError("artifact is not canonical JSON serializable") from exc


def artifact_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_write_bytes(path: Path, data: bytes, *, lock_timeout: float = 10.0) -> None:
    target = Path(path)
    with ArtifactLock(target, timeout=lock_timeout):
        _atomic_replace_bytes(target, data)


def atomic_write_text(path: Path, text: str, *, lock_timeout: float = 10.0) -> None:
    atomic_write_bytes(path, text.encode("utf-8"), lock_timeout=lock_timeout)


def atomic_write_json(path: Path, value: Any, *, lock_timeout: float = 10.0) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value), lock_timeout=lock_timeout)


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]], *, lock_timeout: float = 10.0) -> None:
    data = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    atomic_write_bytes(path, data, lock_timeout=lock_timeout)


def append_jsonl(path: Path, value: Any, *, lock_timeout: float = 10.0) -> None:
    target = Path(path)
    data = canonical_json_bytes(value) + b"\n"
    with ArtifactLock(target, timeout=lock_timeout):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("ab") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())


def load_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("artifact JSON could not be read") from exc


def registry_digest(registry: dict[str, Any]) -> str:
    payload = {key: value for key, value in registry.items() if key != "digest"}
    return artifact_digest(payload)


def read_registry(path: Path, *, expected_format: str) -> dict[str, Any]:
    value = load_json(path)
    if not isinstance(value, dict):
        raise ArtifactValidationError("registry must be a JSON object")
    if value.get("format") != expected_format:
        raise ArtifactValidationError("registry format is invalid")
    revision = value.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
        raise ArtifactValidationError("registry revision is invalid")
    digest = value.get("digest")
    if not isinstance(digest, str) or digest != registry_digest(value):
        raise ArtifactValidationError("registry digest is invalid")
    return value


def update_registry(
    path: Path,
    *,
    expected_format: str,
    update: Callable[[dict[str, Any] | None], dict[str, Any]],
    expected_revision: int | None = None,
    lock_timeout: float = 10.0,
) -> dict[str, Any]:
    target = Path(path)
    with ArtifactLock(target, timeout=lock_timeout):
        current = read_registry(target, expected_format=expected_format) if target.exists() else None
        current_revision = int(current["revision"]) if current else 0
        if expected_revision is not None and expected_revision != current_revision:
            raise ArtifactConflict("artifact registry revision conflict")
        proposed = update(dict(current) if current else None)
        if not isinstance(proposed, dict):
            raise ArtifactValidationError("registry update must return a JSON object")
        proposed = dict(proposed)
        proposed["format"] = expected_format
        proposed["revision"] = current_revision + 1
        proposed.pop("digest", None)
        proposed["digest"] = registry_digest(proposed)
        _atomic_replace_bytes(target, canonical_json_bytes(proposed))
        return proposed


def append_registry_entry(
    path: Path,
    entry: dict[str, Any],
    *,
    expected_format: str,
    expected_revision: int | None = None,
    dedupe_key: Callable[[dict[str, Any]], Any] | None = None,
    lock_timeout: float = 10.0,
) -> dict[str, Any]:
    def append(current: dict[str, Any] | None) -> dict[str, Any]:
        registry = dict(current or {})
        entries = registry.get("entries", [])
        entries = [dict(item) for item in entries if isinstance(item, dict)] if isinstance(entries, list) else []
        if dedupe_key is not None:
            key = dedupe_key(entry)
            entries = [item for item in entries if dedupe_key(item) != key]
        entries.append(dict(entry))
        registry["entries"] = entries
        return registry

    return update_registry(
        path,
        expected_format=expected_format,
        update=append,
        expected_revision=expected_revision,
        lock_timeout=lock_timeout,
    )


def sign_artifact(
    artifact: Path,
    private_key: Path | bytes,
    *,
    key_id: str,
    signed_at: str | None = None,
) -> dict[str, Any]:
    if not key_id.strip():
        raise ArtifactSignatureError("key id is required")
    try:
        Ed25519PrivateKey, _, load_pem_private_key, _ = _cryptography_types()
        key_data = private_key if isinstance(private_key, bytes) else Path(private_key).read_bytes()
        key = load_pem_private_key(key_data, password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise TypeError("not an Ed25519 private key")
        digest = file_digest(artifact)
        signature = key.sign(digest.encode("ascii"))
    except Exception:
        raise ArtifactSignatureError("artifact signing failed") from None
    return {
        "format": "tune-artifact-signature-v1",
        "artifact_digest": digest,
        "key_id": key_id,
        "algorithm": "Ed25519",
        "signature": base64.b64encode(signature).decode("ascii"),
        "signed_at": signed_at or _utc_now(),
    }


def verify_artifact_signature(artifact: Path, signature: dict[str, Any], public_key: Path | bytes) -> bool:
    try:
        _, Ed25519PublicKey, _, load_pem_public_key = _cryptography_types()
        if signature.get("format") != "tune-artifact-signature-v1" or signature.get("algorithm") != "Ed25519":
            return False
        digest = file_digest(artifact)
        if signature.get("artifact_digest") != digest:
            return False
        key_data = public_key if isinstance(public_key, bytes) else Path(public_key).read_bytes()
        key = load_pem_public_key(key_data)
        if not isinstance(key, Ed25519PublicKey):
            return False
        encoded = signature.get("signature")
        if not isinstance(encoded, str):
            return False
        key.verify(base64.b64decode(encoded, validate=True), digest.encode("ascii"))
        return True
    except Exception:
        return False


def _atomic_replace_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _lock_file(handle) -> None:  # noqa: ANN001
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle) -> None:  # noqa: ANN001
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cryptography_types():
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
    except ImportError:
        raise ArtifactSignatureError("security dependencies are not installed") from None
    return Ed25519PrivateKey, Ed25519PublicKey, load_pem_private_key, load_pem_public_key


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
