from __future__ import annotations

import json
import io
import multiprocessing
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tune_artifacts import (
    ArtifactConflict,
    ArtifactLock,
    ArtifactLockTimeout,
    ArtifactSignatureError,
    ArtifactValidationError,
    append_registry_entry,
    artifact_digest,
    atomic_write_json,
    canonical_json_bytes,
    read_registry,
    sign_artifact,
    update_registry,
    verify_artifact_signature,
)
from tune_cli import _validate_output_signature_args, build_parser


REGISTRY_FORMAT = "tune-test-registry-v1"


def _append_registry_worker(path: str, index: int) -> None:
    append_registry_entry(
        Path(path),
        {"id": index},
        expected_format=REGISTRY_FORMAT,
    )


class ArtifactTests(unittest.TestCase):
    def test_canonical_json_and_digest_are_stable(self) -> None:
        left = {"z": "日本語", "a": [2, 1]}
        right = {"a": [2, 1], "z": "日本語"}
        self.assertEqual(b'{"a":[2,1],"z":"\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e"}', canonical_json_bytes(left))
        self.assertEqual(artifact_digest(left), artifact_digest(right))
        with self.assertRaises(ArtifactValidationError):
            canonical_json_bytes({"invalid": float("nan")})

    def test_atomic_replace_preserves_previous_artifact_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "artifact.json"
            atomic_write_json(path, {"revision": 1})
            original = path.read_bytes()
            with patch("tune_artifacts.os.replace", side_effect=OSError("injected replace failure")):
                with self.assertRaises(OSError):
                    atomic_write_json(path, {"revision": 2})
            self.assertEqual(original, path.read_bytes())
            self.assertEqual([], list(path.parent.glob(f".{path.name}.*.tmp")))

    def test_lock_times_out_without_exposing_target_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "secret-name.json"
            with ArtifactLock(path):
                with self.assertRaises(ArtifactLockTimeout) as raised:
                    with ArtifactLock(path, timeout=0.05, poll_interval=0.01):
                        pass
            self.assertNotIn(str(path), str(raised.exception))

    def test_registry_compare_and_swap_and_digest_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "registry.json"
            first = update_registry(
                path,
                expected_format=REGISTRY_FORMAT,
                expected_revision=0,
                update=lambda current: {"entries": [{"id": "one"}]},
            )
            self.assertEqual(1, first["revision"])
            self.assertEqual(first, read_registry(path, expected_format=REGISTRY_FORMAT))
            with self.assertRaises(ArtifactConflict):
                update_registry(
                    path,
                    expected_format=REGISTRY_FORMAT,
                    expected_revision=0,
                    update=lambda current: {"entries": []},
                )

            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["entries"].append({"id": "tampered"})
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(ArtifactValidationError):
                read_registry(path, expected_format=REGISTRY_FORMAT)

    def test_concurrent_registry_append_has_no_entry_loss(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "registry.json"
            context = multiprocessing.get_context("spawn")
            processes = [context.Process(target=_append_registry_worker, args=(str(path), index)) for index in range(6)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=15)
                self.assertEqual(0, process.exitcode)
            registry = read_registry(path, expected_format=REGISTRY_FORMAT)
            self.assertEqual(6, registry["revision"])
            self.assertEqual(set(range(6)), {entry["id"] for entry in registry["entries"]})

    def test_ed25519_signature_detects_single_byte_tamper(self) -> None:
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        except ImportError:
            self.skipTest("security dependencies are not installed")

        private_key = Ed25519PrivateKey.generate()
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "artifact.json"
            artifact.write_bytes(b'{"status":"pass"}')
            signature = sign_artifact(artifact, private_pem, key_id="test-key", signed_at="2026-01-01T00:00:00Z")
            self.assertTrue(verify_artifact_signature(artifact, signature, public_pem))
            artifact.write_bytes(b'{"status":"fail"}')
            self.assertFalse(verify_artifact_signature(artifact, signature, public_pem))

    def test_signing_error_does_not_disclose_private_key_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "artifact.json"
            private_key = Path(temp_dir) / "operator-private-key.pem"
            artifact.write_text("{}", encoding="utf-8")
            private_key.write_text("invalid", encoding="utf-8")
            with self.assertRaises(ArtifactSignatureError) as raised:
                sign_artifact(artifact, private_key, key_id="test-key")
            self.assertNotIn(str(private_key), str(raised.exception))

    def test_sign_and_verify_cli(self) -> None:
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        except ImportError:
            self.skipTest("security dependencies are not installed")

        key = Ed25519PrivateKey.generate()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = root / "artifact.json"
            private_key = root / "private.pem"
            public_key = root / "public.pem"
            signature = root / "artifact.sig.json"
            artifact.write_text('{"status":"pass"}', encoding="utf-8")
            private_key.write_bytes(
                key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
            public_key.write_bytes(
                key.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo,
                )
            )
            parser = build_parser()
            sign_args = parser.parse_args(
                [
                    "sign-artifact",
                    "--artifact",
                    str(artifact),
                    "--private-key",
                    str(private_key),
                    "--key-id",
                    "test-key",
                    "--out",
                    str(signature),
                ]
            )
            with redirect_stdout(io.StringIO()):
                sign_args.func(sign_args)
            verify_args = parser.parse_args(
                [
                    "verify-artifact",
                    "--artifact",
                    str(artifact),
                    "--signature",
                    str(signature),
                    "--public-key",
                    str(public_key),
                ]
            )
            with redirect_stdout(io.StringIO()):
                verify_args.func(verify_args)
            artifact.write_text('{"status":"fail"}', encoding="utf-8")
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
                verify_args.func(verify_args)
            self.assertEqual(1, raised.exception.code)

    def test_production_output_requires_all_signature_arguments(self) -> None:
        missing = SimpleNamespace(private_key=None, key_id=None, signature_out=None)
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised:
            _validate_output_signature_args(missing, required=True)
        self.assertEqual(1, raised.exception.code)
        complete = SimpleNamespace(private_key="private.pem", key_id="production", signature_out="artifact.sig.json")
        _validate_output_signature_args(complete, required=True)


if __name__ == "__main__":
    unittest.main()
