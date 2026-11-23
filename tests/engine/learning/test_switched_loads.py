"""Plan 04-06: extraction + compile emit a switched artefact for every SC2 load.

For each Phase 4 switching load — kettle, oven (preheat plus thermostat
cycling), water heater and resistive dryer element (cycling on top of a
drum-motor step) — a 22-day synthetic buffer yields a switched cluster at the
load's power, and compile_score turns a switched_proposal for it into an
artefact of kind "switched" with at least MIN_TRAIN_EPISODES train episodes
(DET-01, D-22). The oven is additionally replayed end to end: the two-state
artefact reports every ON period verified, and the cycle-only artefact (the
grouping a real LLM may choose) is pinned with its D-28/D-29 unknown state:
the stuck-ON guard forces the preheat OFF at max_on_s, the rest of the
preheat stays in the remainder counted once and claimed by no device (D-08,
OUT-03), the energy committed up to max_on_s is kept (D-09, D-28), and the
preheat's own fall ends the unknown period at 0 W with no unattributed edge
(D-29c).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pytest

from custom_components.powerguesser.engine.detect.online import DeviceModel, OnlineDetector
from custom_components.powerguesser.engine.detect.replay import replay_online
from custom_components.powerguesser.engine.learning.compile import MIN_TRAIN_EPISODES
from custom_components.powerguesser.engine.learning.digest import MIN_SAMPLES
from custom_components.powerguesser.engine.learning.episodes import MIN_EDGE_W, detect_edges
from custom_components.powerguesser.engine.learning.worker import run_stage
from tests.fixtures.detect_traces import add_run, grid_trace
from tests.fixtures.learning.fake_llm import switched_proposal
from tests.fixtures.learning.synth import SynthSpec, build_buffer, utc_day_bounds

GRID_S = 30
OVEN_PREHEAT_STEPS = 25  # the 12.5 min merged preheat of the oven trace

_CASES: dict[str, dict[str, Any]] = {
    "kettle": {
        "spec": {"days": 22},
        "label": "Kettle",
        "category": "kettle",
        "target_w": 2000.0,
        "states": (("on", None),),
    },
    "oven": {
        "spec": {"days": 22, "kettle_per_day": (0, 0), "oven": True},
        "label": "Oven",
        "category": "oven",
        "target_w": 2200.0,
        "states": (("preheat", (8.0, 20.0)), ("cycle", (1.0, 5.0))),
    },
    "water_heater": {
        "spec": {"days": 22, "kettle_per_day": (0, 0), "water_heater": True},
        "label": "Water heater",
        "category": "water_heater",
        "target_w": 3000.0,
        "states": (("on", (15.0, 50.0)),),
    },
    "dryer": {
        "spec": {"days": 22, "kettle_per_day": (0, 0), "dryer": True},
        "label": "Dryer element",
        "category": "dryer",
        "target_w": 2400.0,
        "states": (("on", (1.0, 6.0)),),
    },
}


def _extract(tmp_path: Path, name: str, case: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    """Build the case buffer and run the extract stage; return (digest, stage_dir)."""
    dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    end_ms = int(dt.timestamp() * 1000)
    days = case["spec"]["days"]
    day_bounds_ms = utc_day_bounds(end_ms, days)
    buf_path = tmp_path / f"buf_{name}.sqlite"
    stage_dir = tmp_path / f"stage_{name}"
    build_buffer(buf_path, SynthSpec(**case["spec"]), end_ms)

    extract_res = run_stage(
        "extract",
        {
            "schema": 1,
            "buffer_path": str(buf_path),
            "stage_dir": str(stage_dir),
            "grid_s": GRID_S,
            "day_bounds_ms": day_bounds_ms,
            "first_weekday": datetime.fromtimestamp(
                day_bounds_ms[0] / 1000.0, tz=timezone.utc
            ).weekday(),
            "holdout_days": 0,
            "train_fraction": 0.7,
            "enabled_event_types": ["step", "cycle_shape", "continuous"],
            "metered_channels": [],
            "phase_channels": {},
        },
    )
    return extract_res["digest"], stage_dir


def _switched_cluster_for(digest: dict[str, Any], case: dict[str, Any], name: str) -> dict[str, Any]:
    """The switched cluster near the load power; fail with the digest's cluster list."""
    switched = [c for c in digest["clusters"] if c.get("kind") == "switched"]
    listing = [(c["id"], c["rise_w"]["p50"], c["dur_min"]["p50"], c["n"]) for c in switched]
    near = [
        c
        for c in switched
        if abs(c["rise_w"]["p50"] - case["target_w"]) <= 0.05 * case["target_w"]
    ]
    assert near, (
        f"{name}: no switched cluster within 5% of {case['target_w']} W; "
        f"switched clusters (id, rise p50, dur p50, n): {listing}"
    )
    assert all(c["n"] >= MIN_SAMPLES for c in near), (
        f"{name}: cluster below MIN_SAMPLES={MIN_SAMPLES}: {listing}"
    )
    return near[0]


def _compile(
    tmp_path: Path,
    name: str,
    digest: dict[str, Any],
    case: dict[str, Any],
    stage_dir: Path,
    *,
    states: Any = None,
    round_no: int = 1,
) -> dict[str, Any]:
    """Compile switched_proposal over the digest; return the single artefact."""
    prop = switched_proposal(
        digest,
        label=case["label"],
        category=case["category"],
        target_w=case["target_w"],
        states=case["states"] if states is None else states,
    )
    prop_path = stage_dir / "proposal_1.json"
    prop_path.parent.mkdir(parents=True, exist_ok=True)
    prop_path.write_text(json.dumps(prop), encoding="utf-8")

    compile_res = run_stage(
        "compile_score",
        {
            "stage_dir": str(stage_dir),
            "run_id": f"run-{name}",
            "round": round_no,
            "model": "fake-model",
            "proposal_path": str(prop_path),
            "incumbent": None,
            "validated": True,
            "prior": [],
            "decisions": {},
        },
    )
    assert compile_res["unresolved"] == [], f"{name}: unresolved clusters {compile_res['unresolved']}"
    library = json.loads(Path(compile_res["library"]).read_text(encoding="utf-8"))
    assert len(library) == 1, f"{name}: expected exactly one artefact, got {len(library)}"
    return library[0]


@pytest.mark.parametrize("name", sorted(_CASES))
def test_learning_emits_switched_artefact(tmp_path: Path, name: str) -> None:
    """Each SC2 load: switched cluster at the load power, artefact kind switched."""
    case = _CASES[name]
    digest, stage_dir = _extract(tmp_path, name, case)
    _switched_cluster_for(digest, case, name)
    artefact = _compile(tmp_path, name, digest, case, stage_dir)
    assert artefact["kind"] == "switched"
    assert artefact["scores"]["train"]["n"] >= MIN_TRAIN_EPISODES, (
        f"{name}: train n {artefact['scores']['train']['n']} < {MIN_TRAIN_EPISODES}"
    )
    assert [s["name"] for s in artefact["states"]] == [s[0] for s in case["states"]]


def test_dryer_element_not_confused_with_motor(tmp_path: Path) -> None:
    """The element's cluster is the 2400 W one, never the 250 W drum-motor step."""
    case = _CASES["dryer"]
    digest, stage_dir = _extract(tmp_path, "dryer", case)
    switched = [c for c in digest["clusters"] if c.get("kind") == "switched"]
    motor = [c for c in switched if abs(c["rise_w"]["p50"] - 250.0) <= 50.0]
    assert motor, (
        "digest has no drum-motor cluster near 250 W; the test cannot prove "
        f"the element was not confused with it: {[(c['id'], c['rise_w']['p50']) for c in switched]}"
    )
    prop = switched_proposal(
        digest,
        label=case["label"],
        category=case["category"],
        target_w=case["target_w"],
        states=case["states"],
    )
    chosen = prop["devices"][0]["member_clusters"]
    for c in switched:
        if c["id"] in chosen:
            assert abs(c["rise_w"]["p50"] - 250.0) > 100.0, (
                f"the drum-motor cluster {c['id']} ({c['rise_w']['p50']} W) was chosen for the element"
            )
    artefact = _compile(tmp_path, "dryer", digest, case, stage_dir)
    assert any(abs(s["rise_w"]["p50"] - 2400.0) < 100.0 for s in artefact["states"])


def _oven_artefact(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Extract the oven buffer and compile the two-state artefact."""
    case = _CASES["oven"]
    digest, stage_dir = _extract(tmp_path, "oven", case)
    artefact = _compile(tmp_path, "oven", digest, case, stage_dir)
    by_name = {s["name"]: s for s in artefact["states"]}
    return digest, artefact, by_name


def _oven_trace(by_name: dict[str, Any]) -> tuple[Any, int, int]:
    """Gridded trace: preheat, 3 min off, two cycles of n_cyc steps, 3 min off."""
    n_cyc = round(by_name["cycle"]["dur_s"]["p50"] / GRID_S)
    assert by_name["cycle"]["dur_s"]["lo"] <= n_cyc * GRID_S <= by_name["cycle"]["dur_s"]["hi"], (
        f"live cycle {n_cyc * GRID_S}s outside dur box {by_name['cycle']['dur_s']}"
    )
    r = grid_trace(120, base_w=250.0, noise_w=2.0)
    i = 20
    add_run(r, i, OVEN_PREHEAT_STEPS, 2200.0)
    i += OVEN_PREHEAT_STEPS + 6  # 3 min off
    add_run(r, i, n_cyc, 2200.0)
    i += n_cyc + 6
    add_run(r, i, n_cyc, 2200.0)
    return r, n_cyc, i + n_cyc


def test_oven_two_state_episodes_verified(tmp_path: Path) -> None:
    """The two-state oven artefact: 3 verified episodes, nothing forced or unverified."""
    _digest, artefact, by_name = _oven_artefact(tmp_path)
    n_pre = round(by_name["preheat"]["dur_s"]["p50"] / GRID_S)
    assert n_pre == OVEN_PREHEAT_STEPS, (
        f"preheat dur p50 {by_name['preheat']['dur_s']['p50']}s is not the {OVEN_PREHEAT_STEPS}-step merged preheat"
    )
    assert by_name["preheat"]["dur_s"]["lo"] <= n_pre * GRID_S <= by_name["preheat"]["dur_s"]["hi"], (
        f"live preheat {n_pre * GRID_S}s outside dur box {by_name['preheat']['dur_s']}"
    )
    r, _n_cyc, _end = _oven_trace(by_name)
    res = replay_online([artefact], r, float(GRID_S))
    episodes = res.episodes.get(artefact["signature_id"], [])
    assert len(episodes) == 3, f"expected 3 verified episodes, got {len(episodes)}: {episodes}"
    assert res.unverified == 0, "SC2 evidence: every live ON period is reported and verified"
    assert res.forced_off == 0, "the preheat must not be forced OFF by the stuck-ON guard"


def test_oven_cycle_only_artefact_leaves_rest_of_preheat_in_remainder(tmp_path: Path) -> None:
    """Cycle-only oven: forced OFF at max_on_s, the rest of the preheat stays in the remainder.

    Disclosed known limitation (D-28/D-29): the preheat beyond max_on_s is shown
    as unknown, never followed and never claimed as 0 W; its fall ends the
    unknown period (D-29c) and the thermostat cycles are tracked and verified.
    """
    digest, _two_state, two_by_name = _oven_artefact(tmp_path)
    case = _CASES["oven"]
    stage_dir = tmp_path / "stage_oven"
    artefact = _compile(
        tmp_path,
        "oven_cycle",
        digest,
        case,
        stage_dir,
        states=(("cycle", (1.0, 5.0)),),
        round_no=2,
    )
    cyc = artefact["states"][0]
    assert [s["name"] for s in artefact["states"]] == ["cycle"]
    model = DeviceModel.from_artefact("oven", artefact)
    max_on_s = model.max_on_s
    assert max_on_s + 60.0 < OVEN_PREHEAT_STEPS * GRID_S, (
        f"case does not exist: max_on_s {max_on_s}s is not shorter than the "
        f"{OVEN_PREHEAT_STEPS * GRID_S}s preheat (cycle dur box {cyc['dur_s']})"
    )
    # The two-state grouping would follow the whole preheat; the one-state
    # grouping's cap must be the short cycle-only one.
    assert max_on_s < 1.5 * two_by_name["preheat"]["dur_s"]["hi"]

    n_cyc = round(cyc["dur_s"]["p50"] / GRID_S)
    assert cyc["dur_s"]["lo"] <= n_cyc * GRID_S <= cyc["dur_s"]["hi"], (
        f"live cycle {n_cyc * GRID_S}s outside dur box {cyc['dur_s']}"
    )
    r, n_cyc, _end = _oven_trace(two_by_name)
    assert n_cyc == round(cyc["dur_s"]["p50"] / GRID_S)

    threshold = max(MIN_EDGE_W, model.threshold_w)
    idx, delta = detect_edges(r, threshold)
    falls = {int(ix): float(d) for ix, d in zip(idx, delta) if d < 0}
    preheat_fall = falls[20 + OVEN_PREHEAT_STEPS]
    assert cyc["fall_w"]["lo"] <= preheat_fall <= cyc["fall_w"]["hi"], (
        f"preheat fall {preheat_fall} W outside the cycle fall box {cyc['fall_w']} (D-29c precondition)"
    )

    det = OnlineDetector([model], grid_s=GRID_S, threshold_w=threshold)
    results = [det.step(float(k) * GRID_S, float(r[k])) for k in range(len(r))]

    # D-09: the preheat turns the device ON; the reported power is inside the
    # cycle rise box one sample after the rise.
    on_idx = next(k for k, res in enumerate(results) if "oven" in res.on)
    rise_box = (cyc["rise_w"]["lo"], cyc["rise_w"]["hi"])
    assert rise_box[0] <= results[on_idx].device_w["oven"] <= rise_box[1], (
        f"device_w {results[on_idx].device_w['oven']} outside the cycle rise box {rise_box}"
    )
    pre_edge_idx = on_idx - 2  # engine replay semantics: rise pre-edge sample

    # The stuck-ON guard forces the still-heating oven OFF at max_on_s
    # (between max_on_s and max_on_s + 60 s after the pre-edge sample).
    forced = next(k for k, res in enumerate(results) if res.forced_off == 1)
    held_s = forced * GRID_S - pre_edge_idx * GRID_S
    assert max_on_s <= held_s <= max_on_s + 60.0, (
        f"forced OFF after {held_s}s, expected max_on_s {max_on_s}s + at most 60s"
    )

    # The preheat's fall is finalised two samples after its pre-edge sample.
    fall_pre_idx = 20 + OVEN_PREHEAT_STEPS
    fall_done = fall_pre_idx + 2

    # D-28: from the forcing step until the fall is finalised the oven is
    # unknown — never 0 W, not ON, no energy; the remainder keeps its power
    # (estimated_w 0.0), counted once and claimed by no device (D-08, OUT-03).
    for k in range(forced, fall_done):
        res = results[k]
        assert "oven" in res.unknown, f"step {k}: oven not in the unknown state (D-28)"
        assert "oven" not in res.on, f"step {k}: oven still ON after the guard"
        assert res.energy_wh_delta.get("oven", 0.0) == 0.0, f"step {k}: energy accrual while unknown"
        assert res.estimated_w == 0.0, f"step {k}: the remainder lost the oven power (OUT-03)"

    # D-29c: the preheat's own fall (inside the cycle fall box) ends the
    # unknown period: plain OFF at 0.0 with no unattributed edge.
    end_res = results[fall_done]
    assert "oven" not in end_res.unknown and "oven" not in end_res.on
    assert end_res.device_w["oven"] == 0.0, "0 W is not claimed while the oven is still heating"
    unattributed_before = results[fall_pre_idx].unattributed_total
    for k in range(fall_done, fall_done + 4):
        assert results[k].unattributed_total == unattributed_before, (
            f"step {k}: the preheat's fall was counted as unattributed"
        )

    # D-09/D-28: the energy committed up to max_on_s is the oven's own and is
    # kept; the total stays flat from the fall until the first cycle's rise.
    energy_at_fall = sum(
        res.energy_wh_delta.get("oven", 0.0) for res in results[: fall_done + 1]
    )
    expected_wh = 2200.0 * (forced - pre_edge_idx) * GRID_S / 3600.0
    assert abs(energy_at_fall - expected_wh) <= 2200.0 * GRID_S / 3600.0 * 1.5, (
        f"committed {energy_at_fall:.2f} Wh, expected ~{expected_wh:.2f} Wh "
        f"(2200 W x forced ON time within one grid interval)"
    )
    first_cycle_rise_pre = 20 + OVEN_PREHEAT_STEPS + 6
    for k in range(fall_done + 1, first_cycle_rise_pre + 2):
        accrued = sum(
            results[j].energy_wh_delta.get("oven", 0.0) for j in range(fall_done + 1, k + 1)
        )
        assert accrued == 0.0, f"step {k}: energy changed before the first cycle's rise"

    # Each cycle turns the device ON one sample after its rise and both are
    # reported and verified; the forcing is counted exactly once.
    replay = replay_online([artefact], r, float(GRID_S))
    episodes = replay.episodes.get(artefact["signature_id"], [])
    assert len(episodes) == 2, f"expected the two cycles as verified episodes, got {episodes}"
    assert replay.forced_off == 1
