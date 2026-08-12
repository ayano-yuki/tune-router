from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MODULES = {
    "tune_bandit",
    "tune_cli",
    "tune_clients",
    "tune_composition",
    "tune_constants",
    "tune_evaluation",
    "tune_executor",
    "tune_ft_data",
    "tune_graphs",
    "tune_learned",
    "tune_models",
    "tune_ops",
    "tune_router_learning",
    "tune_selector",
    "tune_shadow",
    "tune_training",
}


def _load_build_backend():
    spec = importlib.util.spec_from_file_location("tune_test_build_backend", PROJECT_ROOT / "build_backend.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load build backend")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackagingTests(unittest.TestCase):
    def test_source_has_only_collision_safe_flat_modules(self) -> None:
        modules = {path.stem for path in (PROJECT_ROOT / "src").glob("*.py")}
        self.assertEqual(EXPECTED_MODULES, modules)
        self.assertFalse((PROJECT_ROOT / "src" / "tune_orchestrator").exists())

    def test_entry_point_targets_tune_cli(self) -> None:
        backend = _load_build_backend()
        self.assertEqual(EXPECTED_MODULES, set(backend.MODULES))
        self.assertEqual("[console_scripts]\ntune-orchestrator = tune_cli:main\n", backend._entry_points())
        self.assertIn('tune-orchestrator = "tune_cli:main"', (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    def test_wheel_contains_internal_modules_and_packaged_graphs(self) -> None:
        backend = _load_build_backend()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            wheel_name = backend.build_wheel(temp)
            wheel_path = temp / wheel_name
            site = temp / "site"

            with zipfile.ZipFile(wheel_path) as wheel:
                names = set(wheel.namelist())
                wheel.extractall(site)

            self.assertEqual({f"{name}.py" for name in EXPECTED_MODULES}, {name for name in names if name.endswith(".py")})
            self.assertTrue(any(name.startswith("graph_definitions/") for name in names))
            self.assertFalse(any("test" in Path(name).parts for name in names))
            self.assertFalse(any(name in {"models.py", "cli.py", "training.py", "__init__.py"} for name in names))

            env = os.environ.copy()
            env["PYTHONPATH"] = str(site)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import pathlib, tune_cli, tune_models; "
                        "from tune_graphs import load_graphs; "
                        "assert pathlib.Path(tune_models.__file__).parent == pathlib.Path(tune_cli.__file__).parent; "
                        "assert load_graphs(tune_cli.PACKAGED_GRAPHS)"
                    ),
                ],
                cwd=temp,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, result.returncode, result.stderr)


if __name__ == "__main__":
    unittest.main()
