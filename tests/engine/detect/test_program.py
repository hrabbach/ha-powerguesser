"""Program devices: stage-template compile (Gate B), replay tracer (Task 1) and
honest divergence/restart/refusal behavior cases (Task 3, D-01..D-04, D-42)."""

import numpy as np
import pytest

from custom_components.powerguesser.engine.detect.online import (  # noqa: E402
    GAP_HOLD_STEPS,
    LOW_RESIDUAL_STEPS,
    STUCK_FACTOR,
    DeviceModel,
    OnlineDetector,
    ProgramModel,
    compile_model,
)
from custom_components.powerguesser.engine.detect.program import (
    PROGRAM_ADVANCE_STEPS,
    PROGRAM_DIVERGE_STEPS,
)
from custom_components.powerguesser.engine.learning.compile import compile_proposal
from tests.fixtures.detect_traces import add_run, grid_trace, switched_artefact
from tests.fixtures.programs import (
    dishwasher_stages,
    kettle_overlay,
    program_artefact,
    program_episode_table,
    program_run,
    program_rle,
)

GRID_S = 30
THRESHOLD = 120.0


def _compile_dishwasher(n_runs: int = 12, *, seed: int = 42, skip=()):
    """Compile a program artefact from n synthetic dishwasher session runs."""
    rng = np.random.default_rng(seed)
    runs = [
        program_rle(
            dishwasher_stages(),
            rng=rng,
            level_jitter=0.03,
            dur_jitter=0.10,
            skip=set(skip),
        )
        for _ in range(n_runs)
    ]
    table = program_episode_table(runs)
    proposal = {
        "devices": [
            {
                "ref": "dev1",
                "member_clusters": ["c1"],
                "tracker_kind": "program",
                "label": "Dishwasher",
                "category": "dishwasher",
            }
        ]
    }
    artefacts, unresolved = compile_proposal(
        proposal,
        table,
        {"c1": list(range(n_runs))},
        run_id="test-run",
        round_no=1,
        model="fake-model",
        input_digest="digest",
        extraction={"grid_s": GRID_S, "edge_threshold_w": THRESHOLD},
        threshold_w=THRESHOLD,
    )
    assert unresolved == []
    assert len(artefacts) == 1
    return artefacts[0].to_dict()


def _dishwasher_model(**kwargs):
    return compile_model(
        "dishwasher", program_artefact("dishwasher", dishwasher_stages(), **kwargs)
    )


def _detector(*models):
    return OnlineDetector(
        list(models), grid_s=GRID_S, threshold_w=THRESHOLD, suppress_alignment=True
    )


def _drive_from(det, r, start_i=0, stop_i=None, ts0=0.0):
    stop = len(r) if stop_i is None else min(stop_i, len(r))
    return [det.step(ts0 + float(i) * GRID_S, float(r[i])) for i in range(start_i, stop)]


def _full_dishwasher_trace(n=400, *, start=100, overlay=None):
    r = grid_trace(n)
    program_run(r, start, dishwasher_stages())
    if overlay is not None:
        overlay(r)
    return r


# ---------------------------------------------------------------------------
# Gate B: the compile half of the tracer
# ---------------------------------------------------------------------------


def test_compile_program_emits_stages() -> None:
    """compile_proposal fits a local stage template from program session members."""
    artefact = _compile_dishwasher()
    session = artefact["session"]
    stages = session["stages"]
    assert len(stages) == 4
    expected = [2000.0, 150.0, 2000.0, 200.0]
    for st, want in zip(stages, expected):
        assert abs(st["level_w"]["p50"] - want) <= 0.05 * want
    terminal = session["terminal"]["fall_w"]
    assert terminal["lo"] <= -200.0 <= terminal["hi"]


# ---------------------------------------------------------------------------
# Task 1 tracer: replay follows a learned dishwasher through its stages
# ---------------------------------------------------------------------------


def test_replay_tracks_dishwasher_runs() -> None:
    """Tracer: a learned dishwasher is followed through its stages by replay_online."""
    arte = program_artefact("dishwasher", dishwasher_stages())
    model = compile_model("dishwasher", arte)
    assert isinstance(model, ProgramModel)

    r = grid_trace(2880, base_w=250.0)
    e1 = program_run(r, 100, dishwasher_stages())
    e2 = program_run(r, 1500, dishwasher_stages())

    det = _detector(model)
    results = _drive_from(det, r)
    det.flush()

    key = "dishwasher"
    on_idx = [i for i, res in enumerate(results) if key in res.on]
    assert on_idx, "the program device never turned ON"
    # D-02/D-33: ON only after promotion, never at the first matched edge
    assert on_idx[0] > 104

    # D-01: reported power inside a stage box; on the one-sample transition
    # lag the D-06 sum clamp may cut the report to the residual instead
    boxes = [(st["level_w"]["lo"], st["level_w"]["hi"]) for st in arte["session"]["stages"]]
    for i in on_idx:
        w = results[i].device_w[key]
        residual = max(0.0, float(r[i]))
        in_a_box = any(lo <= w <= hi for lo, hi in boxes)
        assert in_a_box or w <= residual + 0.5, (i, w, residual)

    # pending phase reads OFF at 0 W (D-33)
    assert all(results[i].device_w[key] == 0.0 for i in range(101, on_idx[0]))

    eps = det.drain_episodes()[key]
    assert len(eps) == 2
    # D-02: committed run energy within 10 % of the generator's true energy
    assert abs(eps[0].energy_wh - e1) <= 0.10 * e1
    assert abs(eps[1].energy_wh - e2) <= 0.10 * e2


def test_compile_model_rejects_corrupt_program_artefacts() -> None:
    """ProgramModel.from_artefact rejects empty stages, bad bounds and bad kinds."""
    arte = program_artefact("prog", dishwasher_stages())

    no_stages = program_artefact("prog", dishwasher_stages())
    del no_stages["session"]["stages"]
    with pytest.raises(ValueError):
        compile_model("prog", no_stages)

    empty_stages = program_artefact("prog", dishwasher_stages())
    empty_stages["session"]["stages"] = []
    with pytest.raises(ValueError):
        compile_model("prog", empty_stages)

    nan_bound = program_artefact("prog", dishwasher_stages())
    nan_bound["session"]["stages"][0]["level_w"]["lo"] = float("nan")
    with pytest.raises(ValueError):
        compile_model("prog", nan_bound)

    reversed_box = program_artefact("prog", dishwasher_stages())
    reversed_box["session"]["stages"][0]["level_w"]["lo"] = 3000.0
    with pytest.raises(ValueError):
        compile_model("prog", reversed_box)

    bad_dur = program_artefact("prog", dishwasher_stages())
    bad_dur["session"]["stages"][0]["dur_s"]["hi"] = 0.0
    with pytest.raises(ValueError):
        compile_model("prog", bad_dur)

    sw = compile_model("sw", switched_artefact("sw"))
    assert isinstance(sw, DeviceModel)
    with pytest.raises(ValueError):
        compile_model("prog", {"kind": "session"})


# ---------------------------------------------------------------------------
# Task 3: honest divergence, restarts, refusals (D-04/D-42)
# ---------------------------------------------------------------------------


def test_abandoned_run_goes_unknown_then_proven_off() -> None:
    """D-04/D-42: abandonment -> unknown (energy kept) -> 0 W only via D-30a."""
    det = _detector(_dishwasher_model())
    r = grid_trace(400)
    program_run(r, 100, [(2000.0, 20), (150.0, 10)])  # abandoned during stage 2
    results = _drive_from(det, r)
    key = "dishwasher"

    drop = 100 + 20 + 10 + 1  # first baseline sample after the partial run
    unk_idx = [i for i in range(drop, len(results)) if key in results[i].unknown]
    assert unk_idx, "the abandoned run never went unknown"
    assert unk_idx[0] - drop <= PROGRAM_DIVERGE_STEPS + 1
    first_unknown = results[unk_idx[0]]
    assert key not in first_unknown.on
    assert first_unknown.device_w[key] == 0.0
    energy_at_unknown = det.energy_wh[key]
    assert energy_at_unknown > 0.0  # committed energy is kept

    closed = det.drain_closed()
    div = [c for c in closed if not c.verified]
    assert len(div) == 1 and div[0].key == key
    assert 0.0 <= div[0].match_score <= 1.0

    # D-42: the D-30a exit needs LOW_RESIDUAL_STEPS low samples from the
    # unknown entry; no tick between running and 0 W shows it running
    on_idx = [i for i, res in enumerate(results) if key in res.on]
    assert max(on_idx) < unk_idx[0]
    off_idx = [i for i in range(unk_idx[0], len(results)) if key not in results[i].unknown]
    assert off_idx, "the unknown program never returned to 0 W"
    first_off = off_idx[0]
    assert first_off - unk_idx[0] >= LOW_RESIDUAL_STEPS - 1
    for i in range(unk_idx[0], min(first_off + 1, len(results))):
        assert results[i].device_w[key] == 0.0
        assert key not in results[i].on
    assert det.energy_wh[key] == energy_at_unknown  # nothing accrues while unknown


def test_slow_decay_ends_unknown_first() -> None:
    """A decaying last stage leaves the box slowly: unknown, then the D-30a off."""
    det = _detector(_dishwasher_model())
    r = _full_dishwasher_trace()
    drying0 = 100 + 20 + 30 + 16 + 1
    for k in range(12):  # linear decay to baseline over 12 samples
        i = drying0 + k
        r[i] = 250.0 + 200.0 * (1.0 - k / 12.0) + (r[i] - 250.0 - 200.0)
    results = _drive_from(det, r)
    key = "dishwasher"
    unk_idx = [i for i, res in enumerate(results) if key in res.unknown]
    assert unk_idx, "the decaying run never went unknown"
    on_idx = [i for i, res in enumerate(results) if key in res.on]
    assert max(on_idx) < unk_idx[0]  # never 0 W straight from running
    assert key not in det.drain_episodes()  # the unverified end reports no episode


def test_slow_decay_with_unmodelled_load_stays_unknown() -> None:
    """Remainder >= half the step: no proven-off evidence, unknown to the end."""
    det = _detector(_dishwasher_model())
    r = grid_trace(400)
    add_run(r, 0, 398, 1200.0)  # unmodelled load keeps the remainder high
    program_run(r, 100, dishwasher_stages())
    drying0 = 100 + 20 + 30 + 16 + 1
    for k in range(12):
        r[drying0 + k] = 250.0 + 1200.0 + 200.0 * (1.0 - k / 12.0)
    # the load stays off for the rest of the original run window
    r[drying0 + 12 : 100 + 86 + 1] = 250.0 + 1200.0
    results = _drive_from(det, r)
    key = "dishwasher"
    unk_idx = [i for i, res in enumerate(results) if key in res.unknown]
    assert unk_idx
    assert all(key in res.unknown for res in results[unk_idx[0] :])
    assert results[-1].device_w[key] == 0.0
    assert key not in results[-1].on


def test_pending_divergence_drops_pending_energy() -> None:
    """D-08: a pending run that diverges restores idle with no committed energy."""
    det = _detector(_dishwasher_model())
    r = grid_trace(300)
    program_run(r, 100, [(2000.0, 6)])  # stage 1 cut short: never promoted
    results = _drive_from(det, r)
    key = "dishwasher"
    assert det.energy_wh[key] == 0.0  # nothing was committed
    rt = det._runtimes[key]
    assert rt.program_phase == "idle"
    assert not rt.on and not rt.unknown
    # the detector still recognises a later full run (continuous timestamps)
    r2 = grid_trace(300)
    program_run(r2, 100, dishwasher_stages())
    results2 = _drive_from(det, r2, ts0=float(len(r)) * GRID_S)
    assert any(key in res.on for res in results2)


def test_overdue_stage_enters_unknown() -> None:
    """A stage held beyond STUCK_FACTOR x dur_s.hi diverges honestly."""
    det = _detector(_dishwasher_model())
    r = grid_trace(500)
    program_run(r, 100, [(2000.0, 20), (150.0, 60), (2000.0, 16), (200.0, 20)])
    results = _drive_from(det, r)
    key = "dishwasher"
    unk_idx = [i for i, res in enumerate(results) if key in res.unknown]
    assert unk_idx, "the overstayed stage never went unknown"
    # stage 2 begins ~122; its overdue bound is 1.5 x 1080 s = 54 samples
    assert unk_idx[0] <= 122 + int(STUCK_FACTOR * 1080.0 / GRID_S) + 2
    closed = [c for c in det.drain_closed() if not c.verified]
    assert len(closed) == 1 and closed[0].key == key


def test_max_run_time_enters_unknown() -> None:
    """A run longer than max_run_s (= STUCK_FACTOR x session dur_s.hi) diverges."""
    det = _detector(_dishwasher_model(session_dur_s=(300.0, 400.0, 1000.0)))
    r = grid_trace(300)
    program_run(r, 100, [(2000.0, 20), (150.0, 30), (2000.0, 20), (200.0, 20)])
    results = _drive_from(det, r)
    key = "dishwasher"
    unk_idx = [i for i, res in enumerate(results) if key in res.unknown]
    assert unk_idx, "the over-long run never went unknown"
    # max_run_s = 1.5 x 1000 s = 50 samples after the run start (~101)
    assert unk_idx[0] <= 101 + 50 + 2


def test_ramp_into_next_stage_advances_without_edge() -> None:
    """A ramp without an edge above the threshold advances after 2 in-box samples."""
    det = _detector(_dishwasher_model())
    r = _full_dishwasher_trace()
    # replace the sharp stage 3 -> 4 fall (at 167) with a 3-sample ramp
    stage4_0 = 100 + 20 + 30 + 16 + 1
    r[stage4_0] = 250.0 + 1400.0
    r[stage4_0 + 1] = 250.0 + 700.0
    results = _drive_from(det, r)
    key = "dishwasher"
    s4 = program_artefact("dishwasher", dishwasher_stages())["session"]["stages"][3]["level_w"]
    in_stage4 = [
        i
        for i, res in enumerate(results)
        if key in res.on and s4["lo"] <= res.device_w[key] <= s4["hi"]
    ]
    assert in_stage4, "the program never reached stage 4"
    # the ramp enters the stage-4 box at stage4_0 + 2; advance needs 2 samples
    assert in_stage4[0] <= stage4_0 + 2 + PROGRAM_ADVANCE_STEPS


def test_promotion_within_d03_bound() -> None:
    """Promotion no later than stage-1 dur_s.lo + one grid step (D-03)."""
    det = _detector(_dishwasher_model())
    r = grid_trace(300)
    program_run(r, 100, dishwasher_stages())
    results = _drive_from(det, r)
    key = "dishwasher"
    on_idx = [i for i, res in enumerate(results) if key in res.on]
    assert on_idx
    edge_ts = 100 * GRID_S
    stage1 = program_artefact("dishwasher", dishwasher_stages())["session"]["stages"][0]
    bound_s = stage1["dur_s"]["lo"] + GRID_S
    assert results[on_idx[0]].grid_ts - edge_ts <= bound_s


def test_kettle_rise_fitting_due_next_stage_entry_refused() -> None:
    """A kettle rise that also fits the program's due next-stage entry is refused."""
    model = _dishwasher_model()
    kettle = compile_model(
        "kettle", switched_artefact("kettle", rise=(1850.0, 2000.0, 2150.0))
    )
    det = _detector(model)
    # stage 1 plus a long stage 2: the program stays due-in-stage-2 while the
    # kettle boils; the overlay sits in BOTH the stage-3 entry box and the
    # kettle rise box
    r = grid_trace(230)
    program_run(r, 100, [(2000.0, 20), (150.0, 80)])
    kettle_overlay(r, 148, 3, w=1900.0)
    # stage 1 would double-fit with the kettle model present (D-41 production
    # path), so the kettle joins only once the program is running
    _drive_from(det, r, 0, 148)
    det.set_models([model, kettle])
    results = _drive_from(det, r, 148)
    key = "dishwasher"
    refused = [
        res
        for prev, res in zip(results, results[1:])
        if res.unattributed_edges.get(key, 0) > prev.unattributed_edges.get(key, 0)
    ]
    assert len(refused) == 1, "the ambiguous kettle rise was not refused once"
    res = refused[0]
    assert res.unattributed_edges["kettle"] > 0
    # the refusal leaves the edge unattributed: the kettle never turns ON and
    # the program's later stage-3 report comes from the measured level (D-01)
    assert key in res.on
    assert "kettle" not in res.on
    assert not det._runtimes["kettle"].on  # the kettle stayed OFF


def test_idle_program_and_switched_double_fit_refused() -> None:
    """One rise fitting an idle program's stage-1 entry AND an OFF switched box."""
    kettle = compile_model("kettle", switched_artefact("kettle"))
    det = _detector(_dishwasher_model(), kettle)
    r = grid_trace(100)
    add_run(r, 10, 20, 2000.0)
    results = _drive_from(det, r)
    refused = [
        res
        for prev, res in zip(results, results[1:])
        if res.unattributed_edges["dishwasher"] > prev.unattributed_edges["dishwasher"]
    ]
    assert len(refused) == 1
    res = refused[0]
    assert res.unattributed_edges["kettle"] == 1
    rt = det._runtimes["dishwasher"]
    assert rt.program_phase == "idle"  # no pending run, no base_w
    assert rt.base_w == 0.0
    assert not det._runtimes["kettle"].on


def test_program_containing_with_switched_runner_up_refused() -> None:
    """One containing program explanation plus a close switched runner-up refuses."""
    near = compile_model("near", switched_artefact("near", rise=(1700.0, 1800.0, 1900.0)))
    det = _detector(_dishwasher_model(), near)
    r = grid_trace(100)
    add_run(r, 10, 20, 1910.0)  # inside the program entry box, outside 'near'
    results = _drive_from(det, r)
    refused = [
        res
        for prev, res in zip(results, results[1:])
        if res.unattributed_edges["dishwasher"] > prev.unattributed_edges["dishwasher"]
    ]
    assert len(refused) == 1
    res = refused[0]
    assert det._runtimes["dishwasher"].program_phase == "idle"
    near_rt = det._runtimes.get("near")
    assert near_rt is None or not near_rt.on


def test_state_restore_round_trip() -> None:
    """Running restores at its stage; gap/grid mismatch restores unknown; pending idles."""
    model = _dishwasher_model()

    # --- running, no gap, same grid: restores running at the stored stage
    det = _detector(model)
    r = grid_trace(190)
    program_run(r, 100, dishwasher_stages())
    results = _drive_from(det, r, 0, 160)  # snapshot mid-run (stage 3)
    assert [i for i, res in enumerate(results) if "dishwasher" in res.on]
    snap = det.state()
    info = snap["devices"]["dishwasher"]["program"]
    assert info["phase"] == "running"
    stored_stage = info["stage_index"]

    det2 = _detector(model)
    last_ts = float(snap["last_grid_ts"])
    det2.restore(snap, last_ts + GRID_S)
    res = det2.step(last_ts + 2 * GRID_S, 250.0 + 2000.0)
    assert "dishwasher" in res.on
    assert det2._runtimes["dishwasher"].stage_index == stored_stage

    # --- gap beyond GAP_HOLD_STEPS: restores unknown
    det3 = _detector(model)
    det3.restore(snap, last_ts + (GAP_HOLD_STEPS + 2) * GRID_S)
    res = det3.step(last_ts + (GAP_HOLD_STEPS + 3) * GRID_S, 250.0)
    assert "dishwasher" in res.unknown and "dishwasher" not in res.on

    # --- changed grid: restores unknown
    snap60 = dict(snap)
    snap60["grid_s"] = 60
    det4 = _detector(model)
    det4.restore(snap60, last_ts + GRID_S)
    res = det4.step(last_ts + 2 * GRID_S, 250.0)
    assert "dishwasher" in res.unknown and "dishwasher" not in res.on

    # --- pending: never persisted, restores idle with no committed energy
    det5 = _detector(model)
    r5 = grid_trace(200)
    program_run(r5, 100, dishwasher_stages())
    _drive_from(det5, r5[:113])  # promotion happens at ~117: still pending
    assert det5._runtimes["dishwasher"].program_phase == "pending"
    snap_pending = det5.state()
    assert "program" not in snap_pending["devices"]["dishwasher"]
    det6 = _detector(model)
    det6.restore(snap_pending, float(snap_pending["last_grid_ts"]) + GRID_S)
    assert det6._runtimes["dishwasher"].program_phase == "idle"
    # the pending energy stayed in the remainder (D-08), never added later
    assert det6.energy_wh["dishwasher"] == 0.0

    # --- switched device dicts carry no program key
    kettle = compile_model("kettle", switched_artefact("kettle"))
    det7 = _detector(model, kettle)
    _drive_from(det7, grid_trace(20))
    snap7 = det7.state()
    assert "program" not in snap7["devices"]["kettle"]
