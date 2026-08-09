from __future__ import annotations

import unittest
from pathlib import Path

from graphs import load_graphs
from ops import required_model_aliases, validate_local_orchestrator_adapter, validate_model_config


GRAPHS = Path(__file__).resolve().parents[2] / "graphs"


class OpsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graphs = load_graphs(GRAPHS)

    def test_required_model_aliases_cover_runtime_graphs(self) -> None:
        aliases = required_model_aliases(self.graphs)
        self.assertIn("storage-specialist", aliases)
        self.assertIn("general-fallback", aliases)
        self.assertIn("general-synthesizer", aliases)
        self.assertIn("verifier", aliases)

    def test_model_config_validation_reports_missing_aliases(self) -> None:
        checks = validate_model_config(
            {
                "defaults": {"base_url": "http://127.0.0.1:18002/v1"},
                "models": {
                    "storage-specialist": {"model": "storage"},
                    "general-fallback": {"model": "general"},
                },
            },
            self.graphs,
        )
        alias_check = next(check for check in checks if check.name == "model-config.aliases")
        self.assertEqual("fail", alias_check.status)
        self.assertIn("database-specialist", alias_check.metadata["missing"])
        self.assertIn("verifier", alias_check.metadata["missing"])

    def test_model_config_validation_accepts_full_alias_set(self) -> None:
        aliases = required_model_aliases(self.graphs)
        checks = validate_model_config(
            {
                "defaults": {"base_url": "http://127.0.0.1:18002/v1"},
                "models": {alias: {"model": "shared-model"} for alias in aliases},
            },
            self.graphs,
        )
        self.assertTrue(all(check.status == "pass" for check in checks))

    def test_local_orchestrator_adapter_validation(self) -> None:
        out = Path(__file__).parent / "output" / "adapter-check"
        out.mkdir(parents=True, exist_ok=True)
        try:
            (out / "adapter_config.json").write_text("{}", encoding="utf-8")
            (out / "adapter_model.safetensors").write_bytes(b"test")
            (out / "orchestrator_config.json").write_text(
                '{"format":"tune-orchestrator-adapter-v1","base_model":"test-model"}',
                encoding="utf-8",
            )
            check = validate_local_orchestrator_adapter(out)
            self.assertEqual("pass", check.status)
            self.assertEqual("test-model", check.metadata["orchestrator_config"]["base_model"])
        finally:
            for path in out.glob("*"):
                path.unlink()
            out.rmdir()


if __name__ == "__main__":
    unittest.main()
