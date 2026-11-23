"""Tests for the README accuracy-table generator (scripts/render_accuracy_table.py).

Plan 06-02 Task 2 (D-20, SC4). The script renders the README accuracy block from
the committed validation report: only classes that pass the D-08 bar show
figures, every other vocabulary row reads "not validated".
"""

from collections.abc import Mapping
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "render_accuracy_table.py"
_ENGINE_PARENT = _REPO_ROOT / "custom_components" / "powerguesser"

START = "<!-- accuracy-table:start -->"
END = "<!-- accuracy-table:end -->"

_README_TEMPLATE = (
    "# PowerGuesser\n"
    "\n"
    "![PowerGuesser](custom_components/powerguesser/brand/icon.png)\n"
    "\n"
    "## Accuracy\n"
    "\n"
    f"{START}\n"
    f"{END}\n"
    "\n"
    "## Known limitations\n"
    "\n"
    "tail text stays byte-identical\n"
)


def _load_module():
    """Load scripts/render_accuracy_table.py as a module via spec_from_file_location."""
    if not _SCRIPT_PATH.exists():
        pytest.fail(f"Script {_SCRIPT_PATH} does not exist yet")
    spec = importlib.util.spec_from_file_location("render_accuracy_table", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec from {_SCRIPT_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _vocabulary():
    """The imported BANDS and REPORT_SCHEMA_VERSION (the script must import the same)."""
    if str(_ENGINE_PARENT) not in sys.path:
        sys.path.insert(0, str(_ENGINE_PARENT))
    from engine.validation.classes import BANDS
    from engine.validation.constants import REPORT_SCHEMA_VERSION

    return BANDS, REPORT_SCHEMA_VERSION


def _data_rows(block):
    """The table's data rows (header and separator excluded)."""
    rows = []
    for line in block.splitlines():
        if not line.startswith("|"):
            continue
        first = line.split("|")[1].strip()
        if first in ("", "Class", "---"):
            continue
        rows.append(line)
    return rows


def _ok_report(schema_version, classes):
    return {
        "schema_version": schema_version,
        "status": "ok",
        "reason": None,
        "generated_at": "2026-11-20T10:00:00Z",
        "holdout": {"start": "2026-11-13", "end": "2026-11-20"},
        "thresholds": {"min_events_pass": 8, "class_min_passing": 2},
        "classes": classes,
    }


def _passing_class():
    return {
        "class": "switched_1.5_3kw_1_level",
        "peers": 2,
        "passing": 2,
        "median_error_pct": 12.3,
        "max_error_pct": 18.0,
        "n_events": 31,
        "passes": True,
    }


def test_row_count_is_the_full_vocabulary():
    """One row per kind x band x level, computed from BANDS and KINDS (never typed).

    Review suggestion L68@8b3be9c: a vocabulary change must fail loudly at both ends.
    """
    mod = _load_module()
    bands, schema = _vocabulary()
    block = mod.render_block(_ok_report(schema, []))
    rows = _data_rows(block)
    assert len(rows) == len(mod.KINDS) * len(bands) * 2
    assert len(rows) == 16


def test_passing_class_shows_figures_and_the_rest_read_not_validated():
    """An ok report with one passing class shows its figures; the other 15 say not validated."""
    mod = _load_module()
    _, schema = _vocabulary()
    block = mod.render_block(_ok_report(schema, [_passing_class()]))

    lines = _data_rows(block)
    validated = [line for line in lines if "| validated |" in line]
    assert len(validated) == 1
    row = validated[0]
    assert "switched_1.5_3kw_1_level" in row
    assert "Switched load, 1.5-3 kW, single level" in row
    assert "2/2" in row
    assert "12.3" in row
    assert "18.0" in row
    assert "31" in row

    not_validated = [line for line in lines if "| not validated |" in line]
    assert len(not_validated) == len(lines) - 1
    # A non-passing row shows no figure at all.
    other = next(line for line in not_validated if "under_0.5kw_1_level" in line)
    assert "12.3" not in other


def test_failing_class_entry_reads_not_validated():
    """passes false -> the row reads not validated and shows no figure."""
    mod = _load_module()
    _, schema = _vocabulary()
    entry = {**_passing_class(), "passes": False}
    block = mod.render_block(_ok_report(schema, [entry]))
    row = next(line for line in _data_rows(block) if "1.5_3kw_1_level" in line)
    assert "| not validated |" in row
    assert "12.3" not in row


@pytest.mark.parametrize(
    "report",
    [
        {"schema_version": 1, "status": "refused", "reason": "insufficient_data"},
        {"schema_version": 1, "status": "none", "reason": "no validation report has been published"},
        {"schema_version": 2, "status": "ok", "reason": None, "classes": []},
        None,
    ],
    ids=["refused", "status-none", "foreign-schema", "missing"],
)
def test_unusable_report_reads_not_validated_with_a_reason(report):
    """Every vocabulary row reads not validated and a reason line names why."""
    mod = _load_module()
    block = mod.render_block(report)
    rows = _data_rows(block)
    assert len(rows) == len(mod.KINDS) * 4 * 2
    assert all("| not validated |" in line for line in rows)
    assert all("| validated |" not in line for line in rows)
    # A reason line exists outside the table.
    reason = mod.reason_line(report)
    assert isinstance(reason, str) and reason
    assert reason in block


def test_redact_keeps_only_class_level_and_header_fields():
    """--redact-from keeps the allowlist only; no entity id, device name or peer list survives."""
    mod = _load_module()
    raw = {
        "schema_version": 1,
        "status": "ok",
        "reason": None,
        "generated_at": "2026-11-20T10:00:00Z",
        "holdout": {"start_ms": 1, "end_ms": 2, "days": [{"day": "x"}]},
        "thresholds": {"min_events_pass": 8},
        "devices": [{"entity_id": "sensor.washer_power", "name": "Washer"}],
        "refusals": [{"entity_id": "sensor.kettle_power"}],
        "not_discovered": [{"entity_id": "sensor.foo_power"}],
        "confirmed_devices": [{"name": "Washer"}],
        "plug_classes": {"sensor.washer_power": "switched_1.5_3kw_1_level"},
        "classes": [
            {
                "class": "switched_1.5_3kw_1_level",
                "peers": 2,
                "passing": 2,
                "median_error_pct": 12.3,
                "max_error_pct": 18.0,
                "median_nde": 0.5,
                "median_mae_w": 90.0,
                "n_events": 31,
                "peer_entities": ["sensor.washer_power", "sensor.dryer_power"],
                "passes": True,
            }
        ],
    }
    out = mod.redact_report(raw)
    text = json.dumps(out, sort_keys=True)

    assert "sensor." not in text
    for forbidden in ("peer_entities", "devices", "refusals", "confirmed_devices", "plug_classes"):
        assert forbidden not in text
    assert "median_nde" not in text and "median_mae_w" not in text

    assert out["schema_version"] == 1
    assert out["status"] == "ok"
    assert out["holdout"] == {"start_ms": 1, "end_ms": 2}
    assert out["thresholds"] == {"min_events_pass": 8}
    assert out["classes"] == [
        {
            "class": "switched_1.5_3kw_1_level",
            "peers": 2,
            "passing": 2,
            "median_error_pct": 12.3,
            "max_error_pct": 18.0,
            "n_events": 31,
            "passes": True,
        }
    ]


def test_write_replaces_only_between_markers_and_check_detects_drift(tmp_path):
    """--check exits 1 when the block differs and 0 after --write; text outside is byte-identical."""
    mod = _load_module()
    _, schema = _vocabulary()

    report_path = tmp_path / "accuracy-report.json"
    report_path.write_text(
        json.dumps(_ok_report(schema, [_passing_class()])), encoding="utf-8"
    )
    readme_path = tmp_path / "README.md"
    readme_path.write_text(_README_TEMPLATE, encoding="utf-8")
    before = readme_path.read_text(encoding="utf-8")

    argv = ["--report", str(report_path), "--readme", str(readme_path)]
    assert mod.main([*argv, "--check"]) == 1

    assert mod.main([*argv, "--write"]) == 0
    after = readme_path.read_text(encoding="utf-8")

    i = before.index(START) + len(START)
    j = before.index(END)
    assert after[:i] == before[:i]
    assert after[after.index(END):] == before[j:]

    assert mod.main([*argv, "--check"]) == 0
    assert "not validated" in after


def test_cli_redact_from_writes_a_redacted_copy(tmp_path):
    """--redact-from RAW --out PATH writes the allowlisted report."""
    mod = _load_module()
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "ok",
                "devices": [{"entity_id": "sensor.washer_power"}],
                "classes": [],
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "redacted.json"
    assert mod.main(["--redact-from", str(raw_path), "--out", str(out_path)]) == 0
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["status"] == "ok"
    assert "devices" not in saved
    assert "sensor." not in out_path.read_text(encoding="utf-8")


def test_runs_under_isolated_python(tmp_path):
    """python3 -I -S runs the script: it imports nothing beyond the stdlib and the HA-free engine.

    Review concern L64@8b3be9c: -I -S keeps site-packages out, so an accidental numpy
    or homeassistant import fails the run.
    """
    report_path = tmp_path / "accuracy-report.json"
    report_path.write_text(
        json.dumps({"schema_version": 1, "status": "none", "reason": "placeholder"}),
        encoding="utf-8",
    )
    readme_path = tmp_path / "README.md"
    readme_path.write_text(_README_TEMPLATE, encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(_SCRIPT_PATH),
            "--report",
            str(report_path),
            "--readme",
            str(readme_path),
            "--check",
        ],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
        check=False,
    )
    assert proc.returncode == 1, proc.stderr  # drift, not an import error
    assert "ModuleNotFoundError" not in proc.stderr
    assert "Traceback" not in proc.stderr
