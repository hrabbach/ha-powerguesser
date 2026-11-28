"""Capture-free compile_score golden (Plan 09-05 Task 1, D-10 gate scope).

This baseline was recorded from the UNCHANGED code (before any production edit)
and committed alone; it is never regenerated afterwards. It pins that a
compile_score job WITHOUT a ``capture_windows`` key and the same job WITH
``capture_windows`` ``[]`` produce byte-identical ``val_{round}.json``,
``library_{round}.json`` and output dicts: the capture-excluded path must not
move when there are no captures (D-10 gate scope, capture-free runs score
exactly as today).

The run reuses the ``test_episode_table_golden.py`` synthetic buffer
(``SynthSpec(days=22, oven=True, washer=True, spikes=True, gaps=True)``, a
14-day preview window) and the ``kettle_proposal`` fixture, so the golden
covers round 1 (no incumbent) and round 2 (an incumbent plus one widened
``actives`` artefact, so ``proposal_gate`` is stamped on the round-2 library).
"""

from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
from syrupy.extensions.amber import AmberSnapshotExtension

from custom_components.powerguesser.engine.learning.worker import run_stage
from tests.fixtures.learning.fake_llm import kettle_proposal
from tests.fixtures.learning.synth import SynthSpec, build_buffer, utc_day_bounds


@pytest.fixture
def snapshot(snapshot):
    return snapshot.use_extension(AmberSnapshotExtension)


def _canon(obj: object) -> object:
    """Round-trip through sorted-key JSON so the snapshot holds plain JSON types."""
    return json.loads(json.dumps(obj, sort_keys=True))


# Output dict keys that carry absolute stage paths: dropped so the golden is
# path-independent (the val/library files themselves are snapshotted separately).
_OUTPUT_PATH_KEYS = ("stage_dir", "library", "scorecard", "val")

# The extract artefacts every compile run reads (the same set refit/compile_score need).
_BASE_FILES = ("episodes.npz", "digest_meta.json", "digest.json", "residual.npz")

_END_DT = datetime(2026, 11, 23, 0, 0, tzinfo=timezone.utc)
_TOTAL_DAYS = 22
_PREVIEW_DAYS = 14
_RUN_ID = "golden-capturefree"


def _out_canon(out: dict) -> dict:
    return _canon({k: v for k, v in out.items() if k not in _OUTPUT_PATH_KEYS})


def _widen_first_rise(artefact: dict) -> dict:
    """A copy of the artefact whose first state's rise_w box is 10 % wider.

    Widening guarantees ``detection_differs(active, matched)`` is True, so
    ``proposal_gate`` is stamped on the round-2 library entry (the D-17 gate
    path the golden must cover).
    """
    widened = copy.deepcopy(artefact)
    box = widened["states"][0]["rise_w"]
    box["lo"] = box["lo"] * 0.9
    box["hi"] = box["hi"] * 1.1
    return widened


def _build_base(tmp_path: Path) -> tuple[Path, dict]:
    """Run extract once on the fixed buffer; return (base stage dir, proposal)."""
    buf_path = tmp_path / "buf_capturefree.sqlite"
    base_dir = tmp_path / "stage_base"
    end_ms = int(_END_DT.timestamp() * 1000)
    all_bounds_ms = utc_day_bounds(end_ms, _TOTAL_DAYS)
    build_buffer(
        buf_path,
        SynthSpec(days=_TOTAL_DAYS, oven=True, washer=True, spikes=True, gaps=True),
        end_ms,
    )

    preview_bounds = all_bounds_ms[: _PREVIEW_DAYS + 1]
    extract_job = {
        "buffer_path": str(buf_path),
        "stage_dir": str(base_dir),
        "grid_s": 30,
        "day_bounds_ms": preview_bounds,
        "first_weekday": datetime.fromtimestamp(
            preview_bounds[0] / 1000.0, tz=timezone.utc
        ).weekday(),
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": ["step", "cycle_shape", "continuous"],
        "metered_channels": [],
        "phase_channels": {},
    }
    extract_res = run_stage("extract", extract_job)
    return base_dir, kettle_proposal(extract_res["digest"])


def _run_variant(base_dir: Path, stage_dir: Path, proposal: dict, *, empty_key: bool) -> dict:
    """Run the round 1 -> round 2 compile sequence; return the six golden pieces."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    for name in _BASE_FILES:
        shutil.copyfile(base_dir / name, stage_dir / name)
    (stage_dir / "proposal_1.json").write_text(json.dumps(proposal), encoding="utf-8")
    (stage_dir / "proposal_2.json").write_text(json.dumps(proposal), encoding="utf-8")

    base_job: dict = {
        "stage_dir": str(stage_dir),
        "run_id": _RUN_ID,
        "model": "golden-model",
        "validated": False,
        "prior": [],
        "decisions": {},
    }
    if empty_key:
        base_job["capture_windows"] = []

    # Round 1: no incumbent.
    job1 = dict(
        base_job,
        round=1,
        proposal_path=str(stage_dir / "proposal_1.json"),
        incumbent=None,
    )
    out1 = run_stage("compile_score", job1)
    lib1 = json.loads((stage_dir / "library_1.json").read_text(encoding="utf-8"))
    val1 = json.loads((stage_dir / "val_1.json").read_text(encoding="utf-8"))

    # Round 2: the same proposal against round 1's own score, plus one widened
    # active under the round-2 signature id so proposal_gate is stamped.
    job2 = dict(
        base_job,
        round=2,
        proposal_path=str(stage_dir / "proposal_2.json"),
        incumbent={
            "round": 1,
            "S": out1["S"],
            "conflict_share": out1["conflict_share"],
        },
    )
    discover = run_stage("compile_score", job2)
    sig_id = json.loads(
        Path(discover["library"]).read_text(encoding="utf-8")
    )[0]["signature_id"]
    out2 = run_stage("compile_score", dict(job2, actives={sig_id: _widen_first_rise(lib1[0])}))
    lib2 = json.loads((stage_dir / "library_2.json").read_text(encoding="utf-8"))
    val2 = json.loads((stage_dir / "val_2.json").read_text(encoding="utf-8"))

    return {
        "val_1": _canon(val1),
        "val_2": _canon(val2),
        "library_1": _canon(lib1),
        "library_2": _canon(lib2),
        "out_1": _out_canon(out1),
        "out_2": _out_canon(out2),
    }


@pytest.fixture(scope="module")
def golden_variants(tmp_path_factory) -> SimpleNamespace:
    tmp = tmp_path_factory.mktemp("capture_free_golden")
    base_dir, proposal = _build_base(tmp)
    plain = _run_variant(base_dir, tmp / "plain", proposal, empty_key=False)
    empty = _run_variant(base_dir, tmp / "empty", proposal, empty_key=True)
    return SimpleNamespace(plain=plain, empty=empty)


def test_capture_free_scoring_matches_golden(golden_variants, snapshot) -> None:
    """Capture-free compile scoring equals the golden recorded from unchanged code.

    Both variants (no ``capture_windows`` key, and ``capture_windows`` ``[]``)
    equal the frozen snapshot, so the capture-excluded path is a no-op without
    captures.
    """
    assert golden_variants.plain == snapshot
    assert golden_variants.empty == golden_variants.plain


def test_golden_covers_round2_gate(golden_variants) -> None:
    """Round 2's golden library carries a proposal_gate (the D-17 path is covered)."""
    entry = golden_variants.plain["library_2"][0]
    assert "proposal_gate" in entry
    assert set(entry["proposal_gate"]) == {"passed", "proposed_S", "active_S", "basis"}
    assert golden_variants.plain["out_1"]["accepted"] is True
    assert golden_variants.plain["out_2"]["accepted"] is False
