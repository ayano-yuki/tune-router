from __future__ import annotations

import base64
import csv
import hashlib
import io
import zipfile
from pathlib import Path


NAME = "tune-orchestrator"
NORMALIZED = "tune_orchestrator"
VERSION = "0.2.0"
DIST_INFO = f"{NORMALIZED}-{VERSION}.dist-info"
ROOT = Path(__file__).resolve().parent
MODULES = (
    "tune_artifacts",
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
)


def get_requires_for_build_wheel(config_settings=None):  # noqa: ANN001
    return []


def get_requires_for_build_editable(config_settings=None):  # noqa: ANN001
    return []


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):  # noqa: ANN001
    del config_settings, metadata_directory
    wheel_name = f"{NORMALIZED}-{VERSION}-py3-none-any.whl"
    wheel_path = Path(wheel_directory) / wheel_name
    records: list[tuple[str, bytes]] = []

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        _write(wheel, records, f"{NORMALIZED}.pth", f"{(ROOT / 'src').as_posix()}\n".encode("utf-8"))
        _write(wheel, records, f"{DIST_INFO}/METADATA", _metadata().encode("utf-8"))
        _write(wheel, records, f"{DIST_INFO}/WHEEL", _wheel_metadata().encode("utf-8"))
        _write(wheel, records, f"{DIST_INFO}/entry_points.txt", _entry_points().encode("utf-8"))
        _write_record(wheel, records)

    return wheel_name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):  # noqa: ANN001
    del config_settings, metadata_directory
    wheel_name = f"{NORMALIZED}-{VERSION}-py3-none-any.whl"
    wheel_path = Path(wheel_directory) / wheel_name
    records: list[tuple[str, bytes]] = []

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for module in MODULES:
            path = ROOT / "src" / f"{module}.py"
            if not path.is_file():
                raise FileNotFoundError(f"wheel module is missing: {path}")
            _write(wheel, records, path.name, path.read_bytes())

        graphs = ROOT / "graphs"
        if graphs.exists():
            for path in sorted(graphs.rglob("*")):
                if path.is_file():
                    rel = "graph_definitions/" + path.relative_to(graphs).as_posix()
                    _write(wheel, records, rel, path.read_bytes())

        _write(wheel, records, f"{DIST_INFO}/METADATA", _metadata().encode("utf-8"))
        _write(wheel, records, f"{DIST_INFO}/WHEEL", _wheel_metadata().encode("utf-8"))
        _write(wheel, records, f"{DIST_INFO}/entry_points.txt", _entry_points().encode("utf-8"))
        _write_record(wheel, records)

    return wheel_name


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):  # noqa: ANN001
    del config_settings
    dist_info = Path(metadata_directory) / DIST_INFO
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_metadata(), encoding="utf-8", newline="\n")
    (dist_info / "WHEEL").write_text(_wheel_metadata(), encoding="utf-8", newline="\n")
    (dist_info / "entry_points.txt").write_text(_entry_points(), encoding="utf-8", newline="\n")
    return DIST_INFO


def _metadata() -> str:
    return "\n".join(
        [
            "Metadata-Version: 2.3",
            f"Name: {NAME}",
            f"Version: {VERSION}",
            "Summary: Learned graph orchestrator and router evaluation harness for TuneRouter",
            "Requires-Python: >=3.11",
            "Requires-Dist: PyYAML>=6.0",
            "Provides-Extra: security",
            "Requires-Dist: cryptography>=43.0.0,<47; extra == 'security'",
            "",
        ]
    )


def _wheel_metadata() -> str:
    return "\n".join(
        [
            "Wheel-Version: 1.0",
            "Generator: tune-orchestrator-local-backend",
            "Root-Is-Purelib: true",
            "Tag: py3-none-any",
            "",
        ]
    )


def _entry_points() -> str:
    return "[console_scripts]\ntune-orchestrator = tune_cli:main\n"


def _write(wheel: zipfile.ZipFile, records: list[tuple[str, bytes]], name: str, data: bytes) -> None:
    wheel.writestr(name, data)
    records.append((name, data))


def _write_record(wheel: zipfile.ZipFile, records: list[tuple[str, bytes]]) -> None:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    for name, data in records:
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
        writer.writerow([name, f"sha256={digest}", str(len(data))])
    writer.writerow([f"{DIST_INFO}/RECORD", "", ""])
    wheel.writestr(f"{DIST_INFO}/RECORD", output.getvalue().encode("utf-8"))
