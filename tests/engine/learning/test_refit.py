"""Unit tests for the refit stage's keep-or-reject, thin-state and determinism rules (Plan 04-12).

The refit stage itself is implemented by the Task-1 tracer; these tests pin its
rules on synthetic buffers built with tests/fixtures/learning/synth.py, and pin
that refit builds its episode table through worker.build_episode_table and
refits each state through compile.fit_state_boxes (Plan 04-13's shared helpers,
RESEARCH Open Question 5) with no private copy of either inside refit.
"""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pytest

from custom_components.powerguesser.engine.learning import worker
from custom_components.powerguesser.engine.learning.artefact import (
    SignatureArtefact,
    State,
)
from custom_components.powerguesser.engine.learning.score import replay
from custom_components.powerguesser.engine.learning.split import (
    block_of,
    data_days,
    split_days,
)
from custom_components.powerguesser.engine.learning.series import load_residual
from custom_components.powerguesser.engine.learning.worker import run_stage
from tests.fixtures.learning.synth import SynthSpec, build_buffer, utc_day_bounds

_DAY_MS = 86_400_000
_END_MS = int(datetime(2026, 11, 23, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
_ENABLED = ["step", "cycle_shape", "continuous"]


def _box(lo: float, p50: float, hi: float) -> dict[str, float]:
    return {"lo": lo, "p50": p50, "hi": hi}


def _artefact(
    sig_id: str,
    rise_box: dict[str, float],
    dur_box: dict[str, float],
    *,
    name: str = "on",
) -> dict:
    """A single-state switched artefact; fall/plateau reuse the rise box."""
    return {
        "schema_version": 1,
        "signature_id": sig_id,
        "kind": "switched",
        "label": "Kettle",
        "category": "appliance",
        "extraction": {
            "grid_s": 30,
            "edge_threshold_w": 120.0,
            "merge_samples": 2,
            "pair_tol": 0.25,
        },
        "states": [
            {
                "name": name,
                "rise_w": rise_box,
                "fall_w": dict(rise_box),
                "dur_s": dur_box,
                "plateau_w": dict(rise_box),
                "phase_share": None,
            }
        ],
        "session": None,
        "time_prior": {"hour": [1 / 24] * 24, "dow": [1 / 7] * 7},
        "constraints": {"max_concurrent": 1},
        "provenance": {},
        "scores": {},
    }


def _job(
    stage_dir: Path,
    buffer_path: Path,
    bounds: list[int],
    current: dict,
    extra: list[dict],
    *,
    validated: bool = True,
) -> dict:
    return {
        "buffer_path": str(buffer_path),
        "stage_dir": str(stage_dir),
        "grid_s": 30,
        "day_bounds_ms": bounds,
        "first_weekday": datetime.fromtimestamp(
            bounds[0] / 1000.0, tz=timezone.utc
        ).weekday(),
        "holdout_days": 0,
        "train_fraction": 0.7,
        "enabled_event_types": list(_ENABLED),
        "metered_channels": [],
        "phase_channels": {},
        "validated": validated,
        "current": current,
        "extra": extra,
    }


def _two_level_kettle_buffer(tmp_path: Path) -> tuple[Path, list[int]]:
    """22 days: older 11 days kettle 2150 W, recent 11 days kettle 2000 W."""
    buf = tmp_path / "buf_two_level.sqlite"
    build_buffer(
        buf,
        SynthSpec(days=11, kettle_w=2150.0, kettle_s=180),
        _END_MS - 11 * _DAY_MS,
    )
    build_buffer(buf, SynthSpec(days=11, kettle_w=2000.0, kettle_s=180), _END_MS)
    return buf, utc_day_bounds(_END_MS, 22)


def _short_train_buffer(tmp_path: Path) -> tuple[Path, list[int]]:
    """22 days: oldest 15 days short kettle (2150 W, 60 s), recent 7 days long
    kettle (2000 W, 180 s) — the 70/30 split puts only shorts in train and
    only longs in validation."""
    buf = tmp_path / "buf_short_train.sqlite"
    build_buffer(
        buf,
        SynthSpec(days=15, kettle_w=2150.0, kettle_s=60, kettle_per_day=(3, 4)),
        _END_MS - 7 * _DAY_MS,
    )
    build_buffer(
        buf,
        SynthSpec(days=7, kettle_w=2000.0, kettle_s=180, kettle_per_day=(2, 3)),
        _END_MS,
    )
    return buf, utc_day_bounds(_END_MS, 22)


def _plain_kettle_buffer(tmp_path: Path) -> tuple[Path, list[int]]:
    buf = tmp_path / "buf_plain.sqlite"
    build_buffer(buf, SynthSpec(days=22), _END_MS)
    return buf, utc_day_bounds(_END_MS, 22)


def _current_artefact() -> dict:
    """Fitted-style narrow boxes around the 2000 W / 180 s kettle episodes."""
    return _artefact(
        "sig-main",
        _box(1900.0, 2000.0, 2100.0),
        _box(170.0, 180.0, 190.0),
    )


def _extra_artefact() -> dict:
    """The merged duplicate: 2150 W kettle episodes."""
    return _artefact(
        "sig-extra",
        _box(1935.0, 2150.0, 2365.0),
        _box(170.0, 180.0, 190.0),
    )


def test_refit_union_widens_boxes_and_keeps_signature(tmp_path: Path) -> None:
    """The union refit covers both kettle groups; signature, kind and label stay."""
    buf, bounds = _two_level_kettle_buffer(tmp_path)
    res = run_stage(
        "refit",
        _job(tmp_path / "stage", buf, bounds, _current_artefact(), [_extra_artefact()]),
    )
    assert res["accepted"] is True

    refit = json.loads(Path(res["result"]).read_text(encoding="utf-8"))
    art = refit["artefact"]
    assert art["signature_id"] == "sig-main"
    assert art["kind"] == "switched"
    assert art["label"] == "Kettle"
    assert art["provenance"]["source"] == "union"
    assert art["provenance"]["aliases"] == ["sig-extra"]

    rise = art["states"][0]["rise_w"]
    # Widened past the current box on both sides: the 2150 W group is covered.
    assert rise["lo"] < 1900.0
    assert rise["hi"] > 2100.0


def test_refit_rejects_worse_union(tmp_path: Path) -> None:
    """A union that degrades the fit is rejected; refit.json still holds both scores."""
    buf, bounds = _short_train_buffer(tmp_path)
    extra_short = _artefact(
        "sig-extra",
        _box(1935.0, 2150.0, 2365.0),
        _box(30.0, 60.0, 90.0),
    )
    res = run_stage(
        "refit",
        _job(tmp_path / "stage", buf, bounds, _current_artefact(), [extra_short]),
    )
    assert res["accepted"] is False
    assert res["new_S"] < res["current_S"]

    refit = json.loads(Path(res["result"]).read_text(encoding="utf-8"))
    assert refit["accepted"] is False
    assert refit["current_S"] == res["current_S"]
    assert refit["new_S"] == res["new_S"]
    # The refitted artefact exists for diagnosis even when rejected.
    assert refit["artefact"]["signature_id"] == "sig-main"


def test_refit_state_with_too_few_episodes_keeps_current_boxes(
    tmp_path: Path,
) -> None:
    """A state whose union has < MIN_TRAIN_EPISODES keeps its current boxes."""
    buf, bounds = _plain_kettle_buffer(tmp_path)
    current = _artefact(
        "sig-main",
        _box(1900.0, 2000.0, 2100.0),
        _box(170.0, 180.0, 190.0),
        name="tank",
    )
    # Second state at 3000 W with no episodes anywhere in this buffer.
    state_b = {
        "name": "heater",
        "rise_w": _box(2850.0, 3000.0, 3150.0),
        "fall_w": _box(2850.0, 3000.0, 3150.0),
        "dur_s": _box(500.0, 600.0, 700.0),
        "plateau_w": _box(2850.0, 3000.0, 3150.0),
        "phase_share": None,
    }
    current["states"].append(state_b)
    # An extra artefact claiming nothing (box around a load that never runs).
    extra_far = _artefact(
        "sig-extra",
        _box(3600.0, 4000.0, 4400.0),
        _box(170.0, 180.0, 190.0),
    )

    res = run_stage(
        "refit",
        _job(tmp_path / "stage", buf, bounds, current, [extra_far]),
    )
    art = json.loads(Path(res["result"]).read_text(encoding="utf-8"))["artefact"]
    # The thin state keeps its current boxes exactly; the fed state was refit.
    assert art["states"][1] == state_b
    assert art["states"][0] != current["states"][0]


def test_refit_is_deterministic(tmp_path: Path) -> None:
    """The same job twice yields a byte-identical refit.json."""
    buf, bounds = _two_level_kettle_buffer(tmp_path)
    res1 = run_stage(
        "refit",
        _job(tmp_path / "stage1", buf, bounds, _current_artefact(), [_extra_artefact()]),
    )
    res2 = run_stage(
        "refit",
        _job(tmp_path / "stage2", buf, bounds, _current_artefact(), [_extra_artefact()]),
    )
    assert Path(res1["result"]).read_bytes() == Path(res2["result"]).read_bytes()


def test_refit_preview_has_no_val_scores(tmp_path: Path) -> None:
    """validated false -> scores.val null, validation_error_pct and basis null."""
    buf, bounds = _two_level_kettle_buffer(tmp_path)
    res = run_stage(
        "refit",
        _job(
            tmp_path / "stage",
            buf,
            bounds,
            _current_artefact(),
            [_extra_artefact()],
            validated=False,
        ),
    )
    art = json.loads(Path(res["result"]).read_text(encoding="utf-8"))["artefact"]
    scores = art["scores"]
    assert scores["validated"] is False
    assert scores["val"] is None
    assert scores["validation_error_pct"] is None
    assert scores["validation_basis"] is None
    assert scores["train"]["n"] > 0


def test_refit_uses_shared_helpers(tmp_path: Path) -> None:
    """Each refitted state equals compile.fit_state_boxes on that state's union
    indices of the table worker.build_episode_table returns for the same
    buffer and bounds — refit holds no private copy of either helper."""
    from custom_components.powerguesser.engine.learning import compile as compile_mod

    buf, bounds = _two_level_kettle_buffer(tmp_path)
    current = _current_artefact()
    extra = _extra_artefact()
    res = run_stage(
        "refit",
        _job(tmp_path / "stage", buf, bounds, current, [extra]),
    )
    new_art = json.loads(Path(res["result"]).read_text(encoding="utf-8"))["artefact"]

    # Independently rebuild the table through the published helper.
    series = load_residual(str(buf), bounds[0], bounds[-1], 30_000, [], {})
    r = series.residual
    valid_days = data_days(r, bounds[0], 30_000, bounds)
    split = split_days(
        valid_days,
        train_fraction=0.7,
        holdout_days=0,
        n_dropped=len(bounds) - 1 - len(valid_days),
    )
    table, _all_rle, threshold_w = worker.build_episode_table(
        r,
        {},
        bounds[0],
        30_000,
        bounds,
        datetime.fromtimestamp(bounds[0] / 1000.0, tz=timezone.utc).weekday(),
        list(_ENABLED),
    )
    table.pop("peak_w", None)
    table.pop("n_segments", None)
    table["block"] = block_of(
        bounds[0] + table["start_i"] * 30_000, bounds, split
    )

    # Union of kind-0 train episodes claimed by either artefact (single state:
    # every union episode is assigned to state 0 by the documented rule).
    arts = [SignatureArtefact.from_dict(current), SignatureArtefact.from_dict(extra)]
    claims = replay(arts, table, block=0)
    union = sorted(
        int(i)
        for art in arts
        for i in claims.claims[art.signature_id]
        if int(table["kind"][int(i)]) == 0
    )
    expected = compile_mod.fit_state_boxes(
        table,
        np.array(union, dtype=np.int64),
        threshold_w,
        name="on",
    )
    assert State.from_dict(new_art["states"][0]) == expected
