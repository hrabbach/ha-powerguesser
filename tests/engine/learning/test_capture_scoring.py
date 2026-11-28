"""Tests for the capture-excluded scoring path of compile_score (Plan 09-05 Task 1, D-10).

The one shared ``_capture_gate_table`` helper builds the episode table in which
every row overlapping a padded capture window has block ``-1``; compile_score's
round acceptance and every D-17 gate call read that table whenever the job
carries a non-empty ``capture_windows``. An emptied validation block is never a
pass and never accepts a round (``insufficient_validation``), while a capture-free
run (no key or ``[]``) scores exactly as before.
"""

from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from custom_components.powerguesser.engine.capture.constants import HOLDOUT_PAD_STEPS
from custom_components.powerguesser.engine.learning import worker
from custom_components.powerguesser.engine.learning.artefact import SignatureArtefact
from custom_components.powerguesser.engine.learning.score import (
    GATE_BASIS,
    block_score,
    gate_proposal,
    replay,
)
from custom_components.powerguesser.engine.learning.worker import (
    _capture_gate_table,
    run_stage,
)
from tests.fixtures.learning.fake_llm import kettle_proposal
from tests.fixtures.learning.synth import SynthSpec, build_buffer, utc_day_bounds

_END_MS = int(datetime(2026, 11, 23, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
_ENABLED = ["step", "cycle_shape", "continuous"]
_EVIDENCE_NOW = "2026-11-25T07:00:00+00:00"
_RUN_ID = "capture-scoring"
_UNSET = object()


def _canon(obj: object) -> object:
    return json.loads(json.dumps(obj, sort_keys=True))


def _widened(artefact: dict) -> dict:
    """A copy of the artefact whose every box is 10 % wider (so detection_differs)."""
    out = copy.deepcopy(artefact)
    for st in out["states"]:
        for key in ("rise_w", "fall_w", "dur_s", "plateau_w"):
            box = st[key]
            st[key] = {"lo": box["lo"] * 0.9, "p50": box["p50"], "hi": box["hi"] * 1.1}
    return out


def _overlaps(row_start: int, row_end: int, windows, pad_ms: int) -> bool:
    """Independent overlap predicate: does [row_start, row_end] overlap a padded window?"""
    for window in windows:
        win_start = int(window[0]) - pad_ms
        win_end = int(window[1]) + pad_ms
        if row_start < win_end and row_end >= win_start:
            return True
    return False


# --------------------------------------------------------------------------- #
# Fixture: one extract on a fixed synthetic buffer (kettle + oven sessions).
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def run_dir(tmp_path_factory) -> SimpleNamespace:
    tmp = tmp_path_factory.mktemp("capture_scoring")
    buf = tmp / "buf.sqlite"
    stage_dir = tmp / "stage"
    build_buffer(buf, SynthSpec(days=22, oven=True, spikes=True, gaps=True), _END_MS)
    bounds = utc_day_bounds(_END_MS, 22)
    extract_res = run_stage(
        "extract",
        {
            "buffer_path": str(buf),
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
        },
    )
    proposal = kettle_proposal(extract_res["digest"])
    (stage_dir / "proposal_1.json").write_text(json.dumps(proposal), encoding="utf-8")
    return SimpleNamespace(stage_dir=stage_dir, proposal=proposal, bounds=bounds, buf=buf)


def _episodes_and_grid(stage_dir: Path):
    episodes = worker.load_episodes(stage_dir / "episodes.npz")
    with np.load(str(stage_dir / "residual.npz"), allow_pickle=False) as res:
        start_ms = int(res["start_ms"][0])
        step_ms = int(res["step_ms"][0])
    return episodes, start_ms, step_ms


def _val_residual(stage_dir: Path) -> float:
    meta = json.loads((stage_dir / "digest_meta.json").read_text(encoding="utf-8"))
    return float(meta["block_residual_wh"]["val"])


def _block1_rows(episodes) -> list[int]:
    return [int(i) for i in np.where(episodes["block"] == 1)[0]]


def _claimed_block1(artefact: SignatureArtefact, episodes) -> list[int]:
    res = replay([artefact], episodes, block=1)
    return sorted(res.claims[artefact.signature_id] & set(_block1_rows(episodes)))


def _window_for_row(episodes, start_ms: int, step_ms: int, row: int) -> list[int]:
    return [
        int(start_ms + episodes["start_i"][row] * step_ms),
        int(start_ms + episodes["end_i"][row] * step_ms),
    ]


def _job(
    run_dir: SimpleNamespace,
    round_no: int,
    *,
    incumbent=None,
    actives=None,
    capture_windows=_UNSET,
    with_evidence: bool = False,
    stage_dir: Path | None = None,
    validated: bool = True,
) -> dict:
    job = {
        "stage_dir": str(stage_dir if stage_dir is not None else run_dir.stage_dir),
        "run_id": _RUN_ID,
        "round": round_no,
        "model": "fake-model",
        "proposal_path": str(run_dir.stage_dir / "proposal_1.json"),
        "incumbent": incumbent,
        "validated": validated,
        "prior": [],
        "decisions": {},
    }
    if actives is not None:
        job["actives"] = actives
    if capture_windows is not _UNSET:
        job["capture_windows"] = capture_windows
    if with_evidence:
        job["with_evidence"] = True
        job["now_iso"] = _EVIDENCE_NOW
    return job


def _library(out: dict) -> list[dict]:
    return json.loads(Path(out["library"]).read_text(encoding="utf-8"))


def _val(stage_dir: Path, round_no: int) -> dict:
    return json.loads((stage_dir / f"val_{round_no}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# The shared helper.
# --------------------------------------------------------------------------- #
def test_gate_table_excludes_capture_windows() -> None:
    """``_capture_gate_table`` sets block -1 on every row overlapping a padded
    window (endpoint inside, or spanning a whole window) and leaves the input
    untouched; with no windows it returns the original object."""
    step_ms = 30_000
    episodes = {
        "start_i": np.array([0, 18, 14, 10, 100], dtype=np.int64),
        "end_i": np.array([1, 19, 22, 40, 101], dtype=np.int64),
        "block": np.array([1, 1, 1, 1, 1], dtype=np.int32),
    }
    # window = grid steps 20..21; padded by HOLDOUT_PAD_STEPS to steps 18..23.
    windows = [[20 * step_ms, 21 * step_ms]]
    pad_ms = HOLDOUT_PAD_STEPS * step_ms

    table = _capture_gate_table(episodes, windows, 0, step_ms)

    assert table is not episodes
    assert list(table["block"]) == [1, -1, -1, -1, 1]
    # Every other array is shared, not copied; the input table is untouched.
    assert table["start_i"] is episodes["start_i"]
    assert list(episodes["block"]) == [1, 1, 1, 1, 1]

    # The exact invariant: block is -1 iff the row overlaps a padded window, and
    # no block-1 row overlaps one (over all windows).
    row_start = 0 + episodes["start_i"] * step_ms
    row_end = 0 + episodes["end_i"] * step_ms
    for i in range(len(episodes["start_i"])):
        overlaps = _overlaps(int(row_start[i]), int(row_end[i]), windows, pad_ms)
        assert (int(table["block"][i]) == -1) is overlaps
        if int(table["block"][i]) == 1:
            assert not overlaps

    # A row spanning a whole window (row 3: [10, 40] encloses [20, 21]) is excluded.
    assert int(table["block"][3]) == -1

    # No capture windows -> the original object, and [] -> also the original.
    assert _capture_gate_table(episodes, [], 0, step_ms) is episodes
    assert _capture_gate_table(episodes, [], None, None) is episodes


# --------------------------------------------------------------------------- #
# compile_score's D-17 gate reads the capture-excluded table.
# --------------------------------------------------------------------------- #
def test_compile_score_gate_excludes_capture_windows(run_dir, monkeypatch) -> None:
    """With actives and capture_windows, every table the gate receives has no
    block-1 row overlapping a padded capture window, and the stamps equal the
    gate result."""
    episodes, start_ms, step_ms = _episodes_and_grid(run_dir.stage_dir)
    plain = run_stage("compile_score", _job(run_dir, 1))
    lib1 = _library(plain)
    artefact = SignatureArtefact.from_dict(lib1[0])
    claimed = _claimed_block1(artefact, episodes)
    assert claimed, "fixture must have validation rows the kettle claims"
    windows = [_window_for_row(episodes, start_ms, step_ms, r) for r in claimed[:3]]
    actives = {lib1[0]["signature_id"]: _widened(lib1[0])}

    tables: list = []
    results: list = []
    original = worker.gate_proposal

    def spy(proposed, active, table, residual_wh):
        out = original(proposed, active, table, residual_wh)
        tables.append(table)
        results.append(out)
        return out

    monkeypatch.setattr(worker, "gate_proposal", spy)

    out = run_stage(
        "compile_score",
        _job(run_dir, 1, actives=actives, capture_windows=windows),
    )
    entry = _library(out)[0]

    assert tables, "the D-17 gate must have been called"
    pad_ms = HOLDOUT_PAD_STEPS * step_ms
    for table in tables:
        for i in np.where(table["block"] == 1)[0]:
            assert not _overlaps(
                int(start_ms + table["start_i"][i] * step_ms),
                int(start_ms + table["end_i"][i] * step_ms),
                windows,
                pad_ms,
            )
    # The artefact's proposal_gate is exactly the gate's return value.
    assert entry["proposal_gate"] == _canon(results[0])
    assert set(entry["proposal_gate"]) == {"passed", "proposed_S", "active_S", "basis"}


# --------------------------------------------------------------------------- #
# compile_score's round acceptance reads the capture-excluded table.
# --------------------------------------------------------------------------- #
def test_round_acceptance_excludes_capture_windows(run_dir, monkeypatch) -> None:
    """A round-2 acceptance reads the capture-excluded table: the accept spy and
    val_2.json equal a block_score of a replay over ``_capture_gate_table`` (and
    differ from the capture-free round), and a round whose only gain is the
    captured rows is accepted without capture_windows and refused with them."""
    stage_dir = run_dir.stage_dir
    episodes, start_ms, step_ms = _episodes_and_grid(stage_dir)
    val_energy = _val_residual(stage_dir)
    plain1 = run_stage("compile_score", _job(run_dir, 1))
    lib1 = _library(plain1)
    artefact = SignatureArtefact.from_dict(lib1[0])
    claimed = _claimed_block1(artefact, episodes)
    block1 = set(_block1_rows(episodes))
    unclaimed = sorted(block1 - set(claimed))
    assert claimed and unclaimed, "fixture needs claimed and unclaimed validation rows"

    # Exclude as many claimed rows as possible while keeping one unclaimed
    # validation row alive (the exclusion must not empty the block here).
    keep = max(
        unclaimed,
        key=lambda r: min(
            abs(int(episodes["start_i"][r]) - int(episodes["start_i"][c])) for c in claimed
        ),
    )
    covered = list(claimed)
    windows: list = []
    while covered:
        windows = [_window_for_row(episodes, start_ms, step_ms, c) for c in covered]
        acc_table = _capture_gate_table(episodes, windows, start_ms, step_ms)
        if int(acc_table["block"][keep]) == 1:
            break
        covered.remove(
            min(covered, key=lambda c: abs(int(episodes["start_i"][c]) - int(episodes["start_i"][keep])))
        )
    assert windows, "no captured-row window survived"
    acc_table = _capture_gate_table(episodes, windows, start_ms, step_ms)
    assert int(acc_table["block"][keep]) == 1

    accept_args: list = []
    original_accept = worker.accept

    def accept_spy(new_score, best_score):
        accept_args.append((new_score, best_score))
        return original_accept(new_score, best_score)

    monkeypatch.setattr(worker, "accept", accept_spy)

    # Capture-free round 2, to learn the capture-free S for the incumbent.
    incumbent = {"round": 1, "S": 0.0, "conflict_share": 1.0}
    out_cf = run_stage("compile_score", _job(run_dir, 2, incumbent=incumbent))
    val_cf = _val(stage_dir, 2)
    lib_cf = _library(out_cf)
    assert out_cf["accepted"] is True
    expected_cf = block_score(replay([SignatureArtefact.from_dict(d) for d in lib_cf], episodes, block=1, residual_wh=val_energy))
    assert val_cf["S"] == pytest.approx(round(expected_cf[0], 5))

    # Capture-excluded round 2 with an incumbent the excluded replay cannot beat.
    arts_win = [SignatureArtefact.from_dict(d) for d in lib_cf]
    expected_S, expected_conf = block_score(replay(arts_win, acc_table, block=1, residual_wh=val_energy))
    excluded_incumbent = {
        "round": 1,
        "S": expected_S,
        "conflict_share": max(expected_conf, float(val_cf["conflict_share"])),
    }
    assert expected_cf[0] >= expected_S + 0.005, "fixture: exclusion must move S by a margin"

    out_win = run_stage(
        "compile_score",
        _job(run_dir, 2, incumbent=excluded_incumbent, capture_windows=windows),
    )
    val_win = _val(stage_dir, 2)

    assert out_win["accepted"] is False
    assert val_win["accepted"] is False
    assert val_win["S"] == pytest.approx(round(expected_S, 5))
    assert val_win["conflict_share"] == pytest.approx(round(expected_conf, 5))
    assert val_win["S"] != val_cf["S"]

    # The accept spy got exactly block_score of the capture-excluded replay.
    assert accept_args, "accept must have been called for the capture-excluded round"
    seen_new = accept_args[-1][0]
    assert seen_new[0] == pytest.approx(expected_S, abs=1e-9)
    assert seen_new[1] == pytest.approx(expected_conf, abs=1e-9)

    # val_2.json's per_signature equals that replay's.
    acc_replay = replay(arts_win, acc_table, block=1, residual_wh=val_energy)
    for sig, stats in acc_replay.per_signature.items():
        assert val_win["per_signature"][sig]["n"] == stats["n"]
        assert val_win["per_signature"][sig]["claimed_share"] == pytest.approx(
            round(float(stats["claimed_wh"]) / acc_replay.denominator_wh, 4)
        )


# --------------------------------------------------------------------------- #
# insufficient_validation: the exclusion emptied the block.
# --------------------------------------------------------------------------- #
_INSUFFICIENT_GATE = {
    "passed": False,
    "proposed_S": None,
    "active_S": None,
    "basis": GATE_BASIS,
    "reason": "insufficient_validation",
}


def test_insufficient_validation_when_all_val_rows_captured(run_dir, monkeypatch) -> None:
    """Every validation row inside a padded window: neither round 1 nor round 2 is
    accepted, val_{round}.json carries the reason and no score, the gate verdict
    is the insufficiency, and neither accept nor gate is called."""
    stage_dir = run_dir.stage_dir
    episodes, start_ms, step_ms = _episodes_and_grid(stage_dir)
    block1 = _block1_rows(episodes)
    assert block1
    lo = int(start_ms + episodes["start_i"][block1].min() * step_ms)
    hi = int(start_ms + episodes["end_i"][block1].max() * step_ms)
    windows = [[lo, hi]]
    # The whole block must be inside the padded window.
    acc_table = _capture_gate_table(episodes, windows, start_ms, step_ms)
    assert not np.any(acc_table["block"] == 1)

    # Learn each round's matched signature id from a plain run (no capture windows).
    actives_by_round = {}
    for round_no in (1, 2):
        lib = _library(run_stage("compile_score", _job(run_dir, round_no)))
        actives_by_round[round_no] = {lib[0]["signature_id"]: _widened(lib[0])}

    accept_args: list = []
    gate_args: list = []
    original_accept = worker.accept
    original_gate = worker.gate_proposal
    monkeypatch.setattr(worker, "accept", lambda a, b: accept_args.append((a, b)) or original_accept(a, b))
    monkeypatch.setattr(
        worker, "gate_proposal", lambda *a, **k: gate_args.append(a) or original_gate(*a, **k)
    )

    for round_no, incumbent in ((1, None), (2, {"round": 1, "S": 0.5, "conflict_share": 0.0})):
        out = run_stage(
            "compile_score",
            _job(
                run_dir,
                round_no,
                incumbent=incumbent,
                actives=actives_by_round[round_no],
                capture_windows=windows,
            ),
        )
        assert out["accepted"] is False
        assert out["S"] == 0.0
        assert out["conflict_share"] == 0.0
        assert out["reason"] == "insufficient_validation"
        assert _val(stage_dir, round_no) == {
            "round": round_no,
            "S": 0.0,
            "conflict_share": 0.0,
            "accepted": False,
            "reason": "insufficient_validation",
            "per_signature": {},
        }
        entry = _library(out)[0]
        assert entry["proposal_gate"] == _INSUFFICIENT_GATE

    assert accept_args == [], "accept must not be called on an emptied block"
    assert gate_args == [], "the gate must not be called on an emptied block"


def test_insufficient_validation_without_residual_grid(run_dir, monkeypatch) -> None:
    """capture_windows non-empty but residual.npz absent: the exclusion cannot be
    computed, so the round is not accepted and the gate is the insufficiency."""
    stage_dir = run_dir.stage_dir
    # A full copy of the stage dir without residual.npz.
    copied = stage_dir.parent / "stage_no_residual"
    copied.mkdir(exist_ok=True)
    for path in stage_dir.iterdir():
        if path.name == "residual.npz":
            continue
        shutil.copyfile(path, copied / path.name)

    accept_args: list = []
    gate_args: list = []
    original_accept = worker.accept
    original_gate = worker.gate_proposal
    monkeypatch.setattr(worker, "accept", lambda a, b: accept_args.append((a, b)) or original_accept(a, b))
    monkeypatch.setattr(
        worker, "gate_proposal", lambda *a, **k: gate_args.append(a) or original_gate(*a, **k)
    )

    # Learn the signature id from a plain (no capture windows) run in the copy.
    lib = _library(run_stage("compile_score", _job(run_dir, 1, stage_dir=copied, validated=False)))
    actives = {lib[0]["signature_id"]: _widened(lib[0])}

    out = run_stage(
        "compile_score",
        _job(
            run_dir,
            1,
            actives=actives,
            capture_windows=[[0, 1]],
            stage_dir=copied,
            validated=False,
        ),
    )
    assert out["accepted"] is False
    assert out["reason"] == "insufficient_validation"
    assert _val(copied, 1)["reason"] == "insufficient_validation"
    assert _library(out)[0]["proposal_gate"] == _INSUFFICIENT_GATE
    assert accept_args == []
    assert gate_args == []


# --------------------------------------------------------------------------- #
# Capture-free behaviour is unchanged.
# --------------------------------------------------------------------------- #
def test_capture_free_empty_val_block_unchanged(run_dir, monkeypatch) -> None:
    """Without capture_windows an empty validation block keeps today's verdict and
    acceptance: the gate verdict equals gate_proposal's return, round 1 is
    accepted, and neither output carries a reason key."""
    stage_dir = run_dir.stage_dir
    # Learn the signature id from a normal run.
    lib = _library(run_stage("compile_score", _job(run_dir, 1)))
    actives = {lib[0]["signature_id"]: _widened(lib[0])}
    val_energy = _val_residual(stage_dir)

    real_load = worker.load_episodes

    def empty_block1(path):
        table = dict(real_load(path))
        block = np.array(table["block"], copy=True)
        block[block == 1] = -1
        table["block"] = block
        return table

    monkeypatch.setattr(worker, "load_episodes", empty_block1)

    out = run_stage("compile_score", _job(run_dir, 1, actives=actives))
    assert out["accepted"] is True
    assert "reason" not in out
    val = _val(stage_dir, 1)
    assert "reason" not in val
    assert val["S"] == 0.0

    entry = _library(out)[0]
    patched_episodes = empty_block1(stage_dir / "episodes.npz")
    expected = gate_proposal(
        SignatureArtefact.from_dict({k: v for k, v in entry.items() if k != "proposal_gate"}),
        SignatureArtefact.from_dict(actives[lib[0]["signature_id"]]),
        patched_episodes,
        val_energy,
    )
    assert entry["proposal_gate"] == _canon(expected)
    assert set(entry["proposal_gate"]) == {"passed", "proposed_S", "active_S", "basis"}


def test_scores_and_evidence_keep_full_blocks(run_dir) -> None:
    """Per-artefact scores and evidence describe the candidate: they stay on the
    full blocks, identical between a job with capture_windows and the same job
    without (only the acceptance and the gate read the capture-excluded table)."""
    stage_dir = run_dir.stage_dir
    episodes, start_ms, step_ms = _episodes_and_grid(stage_dir)

    out_cf = run_stage("compile_score", _job(run_dir, 1, with_evidence=True))
    lib_cf = _library(out_cf)
    artefact = SignatureArtefact.from_dict(lib_cf[0])
    claimed = _claimed_block1(artefact, episodes)
    assert claimed
    windows = [_window_for_row(episodes, start_ms, step_ms, r) for r in claimed]

    out_win = run_stage(
        "compile_score",
        _job(run_dir, 1, with_evidence=True, capture_windows=windows),
    )
    lib_win = _library(out_win)

    assert _canon(lib_win) == _canon(lib_cf)
    assert all("evidence" in entry for entry in lib_win)
    for entry in lib_win:
        assert set(entry["scores"]) == {
            "train",
            "val",
            "validated",
            "conflicts",
            "validation_error_pct",
            "validation_basis",
        }
