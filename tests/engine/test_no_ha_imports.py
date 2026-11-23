"""Ensure the engine package has zero homeassistant or integration imports."""

import ast
from pathlib import Path


def test_no_homeassistant_imports_in_engine() -> None:
    """Validate that custom_components/powerguesser/engine never imports homeassistant."""
    engine_dir = Path(__file__).parent.parent.parent / "custom_components" / "powerguesser" / "engine"
    assert engine_dir.is_dir(), f"Engine dir {engine_dir} not found"

    violations: list[str] = []

    for py_file in engine_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(py_file))
        allowed_level = len(py_file.relative_to(engine_dir).parts)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "homeassistant" or alias.name.startswith("homeassistant."):
                        violations.append(f"{py_file.name}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > allowed_level:
                    violations.append(f"{py_file.name}:{node.lineno}: relative import level {node.level} > {allowed_level}")
                if node.module and (
                    node.module == "homeassistant" or node.module.startswith("homeassistant.")
                ):
                    violations.append(f"{py_file.name}:{node.lineno}: from {node.module} import ...")

    assert not violations, "Forbidden homeassistant imports found in engine:\n" + "\n".join(violations)


def test_engine_does_not_import_integration_package() -> None:
    """Validate that engine files never import custom_components directly."""
    engine_dir = Path(__file__).parent.parent.parent / "custom_components" / "powerguesser" / "engine"
    assert engine_dir.is_dir(), f"Engine dir {engine_dir} not found"

    violations: list[str] = []

    for py_file in engine_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "custom_components" or alias.name.startswith("custom_components."):
                        violations.append(f"{py_file.name}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if not node.level and node.module and (
                    node.module == "custom_components" or node.module.startswith("custom_components.")
                ):
                    violations.append(f"{py_file.name}:{node.lineno}: from {node.module} import ...")

    assert not violations, "Forbidden custom_components imports found in engine:\n" + "\n".join(violations)
