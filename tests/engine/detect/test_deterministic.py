"""Determinism and the network/LLM-free guarantee (DET-04, D-10)."""

import ast
from pathlib import Path

import numpy as np

from custom_components.powerguesser.engine.detect.online import DeviceModel, OnlineDetector
from custom_components.powerguesser.engine.detect.replay import replay_online
from tests.fixtures.detect_traces import add_run, grid_trace, random_series, switched_artefact

GRID_S = 30
FORBIDDEN = ("homeassistant", "aiohttp", "socket", "urllib", "http", "requests", "ssl")


def _overlap_trace() -> tuple[np.ndarray, list[dict]]:
    r = random_series(99, 1500)
    for start, n, w in ((50, 6, 2000.0), (200, 10, 2000.0), (400, 40, 3000.0), (600, 8, 2200.0)):
        add_run(r, start, n, w)
    artes = [
        switched_artefact("k", rise=(1900.0, 2000.0, 2100.0), plateau=(1900.0, 2000.0, 2100.0)),
        switched_artefact("h", rise=(2900.0, 3000.0, 3100.0), plateau=(2900.0, 3000.0, 3100.0)),
        switched_artefact("o", rise=(2100.0, 2200.0, 2300.0), plateau=(2100.0, 2200.0, 2300.0)),
    ]
    return r, artes


def test_deterministic() -> None:
    """Two runs over the same trace give identical OnlineReplay and state()."""
    r, artes = _overlap_trace()
    rep1 = replay_online(artes, r, float(GRID_S))
    rep2 = replay_online(artes, r, float(GRID_S))
    assert rep1 == rep2

    models = [DeviceModel.from_artefact(a["signature_id"], a) for a in artes]
    det1 = OnlineDetector(models, grid_s=GRID_S, threshold_w=120.0)
    det2 = OnlineDetector(models, grid_s=GRID_S, threshold_w=120.0)
    for i in range(len(r)):
        det1.step(float(i) * GRID_S, float(r[i]))
        det2.step(float(i) * GRID_S, float(r[i]))
    assert det1.state() == det2.state()


def test_detect_package_imports_no_network_or_llm() -> None:
    """engine/detect imports no network, HA or LLM module, one level deep."""
    repo = Path(__file__).parent.parent.parent.parent
    detect_dir = repo / "custom_components" / "powerguesser" / "engine" / "detect"
    engine_dir = detect_dir.parent

    files: set[Path] = set(detect_dir.glob("*.py"))
    for py_file in list(detect_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                base = detect_dir
                for _ in range(node.level - 1):
                    base = base.parent
                module = node.module or ""
                target = base / (module.replace(".", "/") + ".py")
                if target.exists():
                    files.add(target)

    violations: list[str] = []
    for py_file in files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if root in FORBIDDEN or "llm" in root:
                    violations.append(f"{py_file.name}:{node.lineno}: {name}")

    assert not violations, "Forbidden imports in engine/detect graph:\n" + "\n".join(violations)
