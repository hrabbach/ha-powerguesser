"""Plan 04-06: every Phase 4 switching load from a learning run to live sensors.

Proves ROADMAP success criterion 2 from the real entry point (DET-01, D-22,
D-24): a 22-day learning run over buffered synthetic data yields a switched
candidate for the load, Review offers Confirm for it, and after confirming,
the device's power sensor follows the load at the pinned hub grid, its energy
grows and the remainder drops. The deterministic responder stands in for the
LLM (fake_llm.switched_proposal over the run's digest): what is proven is
that extraction offers a switched cluster for each load and that the switched
path from that cluster to the sensors works.
"""

from datetime import timedelta
import json
import math
import re
from typing import Any

import pytest

from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import entity_registry as er
from homeassistant.util import dt as dt_util
from pytest_homeassistant.common import async_fire_time_changed

from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate

from tests.conftest import (
    async_fill_synth_buffer,
    async_open_options_step,
    async_seed_consent,
    async_seed_llm,
    async_set_power,
    async_setup_via_flow,
    mock_llm,
)
from tests.fixtures.learning.fake_llm import chat_completion, switched_proposal
from tests.fixtures.learning.synth import SynthSpec

GRID_S = 30
BASE_W = 250.0

_CLUSTER_RE = re.compile(
    r"id=(?P<id>c\d+)\s+kind=(?P<kind>\w+)\s+n=(?P<n>\d+)\s+"
    r"rise=(?P<r10>-?[\d.]+)/(?P<r50>-?[\d.]+)/(?P<r90>-?[\d.]+)\s+"
    r"fall=(?P<f50>-?[\d.]+)\s+dur_min=(?P<d10>[\d.]+)-(?P<d90>[\d.]+)"
)


def _digest_from_request(request_data: Any) -> dict[str, Any]:
    """Rebuild a digest-shaped dict from the rendered digest in an LLM request.

    The rendered cluster lines carry dur_min as p10-p90 only; the midpoint
    stands in for p50 in the dur-range selection (rise p50 is exact in the
    text). The full digest dict with exact dur p50 is used in the engine
    tests, which call switched_proposal directly on the extract output.
    """
    if isinstance(request_data, (bytes, bytearray)):
        request_data = request_data.decode("utf-8")
    req = json.loads(request_data) if isinstance(request_data, str) else request_data
    clusters: list[dict[str, Any]] = []
    for m in req.get("messages", []):
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        for cm in _CLUSTER_RE.finditer(content):
            d10 = float(cm.group("d10"))
            d90 = float(cm.group("d90"))
            clusters.append(
                {
                    "id": cm.group("id"),
                    "kind": cm.group("kind"),
                    "n": int(cm.group("n")),
                    "rise_w": {
                        "p10": float(cm.group("r10")),
                        "p50": float(cm.group("r50")),
                        "p90": float(cm.group("r90")),
                    },
                    "fall_w": {"p50": float(cm.group("f50"))},
                    "dur_min": {"p10": d10, "p50": (d10 + d90) / 2.0, "p90": d90},
                }
            )
    return {"clusters": clusters}


def _aligned_move(freezer: Any) -> None:
    """Move the freezer to the next epoch multiple of the 30 s grid."""
    now = dt_util.utcnow().timestamp()
    aligned = math.ceil(now / GRID_S) * GRID_S
    freezer.move_to(dt_util.utc_from_timestamp(aligned))


def _pin_grid(entry: ConfigEntry) -> None:
    """Publish a 30 s profile snapshot so the D-26 grid is pinned at 30."""
    enabled, disabled = gate(float(GRID_S), False)
    entry.runtime_data.profile_coordinator.async_set_updated_data(
        ProfileSnapshot(
            sources={},
            effective_s=float(GRID_S),
            assumed=False,
            enabled=enabled,
            disabled=disabled,
        )
    )


async def _tick(hass: HomeAssistant, freezer: Any, seconds: int = GRID_S) -> None:
    """Advance the freezer and fire the time-changed event."""
    freezer.tick(timedelta(seconds=seconds))
    async_fire_time_changed(hass, dt_util.utcnow())
    await hass.async_block_till_done()


async def _drive_ticks(hass: HomeAssistant, freezer: Any, n: int) -> None:
    """Advance n full grid steps via the 10 s tick path."""
    for _ in range(n * 3):
        await _tick(hass, freezer, 10)


def _state(hass: HomeAssistant, entity_id: str) -> str:
    """Return the state string of an entity."""
    state = hass.states.get(entity_id)
    assert state is not None
    return state.state


async def _run_learning_to_candidate(
    hass: HomeAssistant,
    entry: ConfigEntry,
    spec: SynthSpec,
    aioclient_mock: Any,
    responder_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Fill the buffer, run learning with a switched responder, return the candidate."""
    await async_fill_synth_buffer(hass, entry, spec=spec)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)

    def _responder(data: Any) -> dict[str, Any]:
        try:
            prop = switched_proposal(_digest_from_request(data), **responder_kwargs)
        except AssertionError as exc:
            pytest.fail(f"responder found no switched cluster: {exc}")
        return {"status": 200, "json": chat_completion(json.dumps(prop))}

    mock_llm(aioclient_mock, "http://llm.test/v1", _responder)
    sup = entry.runtime_data.learning
    await sup.async_request_run("service")
    assert sup.run_task is not None
    await sup.run_task

    visible = sup.visible_candidates()
    assert len(visible) == 1, f"expected exactly one candidate, got {visible!r}"
    cand = visible[0]
    assert cand["kind"] == "switched"
    return cand


async def _confirm_and_entities(
    hass: HomeAssistant,
    entry: ConfigEntry,
    sig_id: str,
    name: str,
) -> tuple[str, str, str]:
    """Review -> candidate menu offers confirm -> confirm; return (uuid, power, energy)."""
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": sig_id}
    )
    assert res_cand["type"] is FlowResultType.MENU
    assert res_cand["step_id"] == "candidate"
    menu_options = list(res_cand.get("menu_options") or [])
    assert "confirm" in menu_options, (
        f"Confirm not offered for the switched candidate (D-22): {menu_options}"
    )
    res_form = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "confirm"}
    )
    assert res_form["type"] is FlowResultType.FORM
    assert res_form["step_id"] == "confirm"
    await hass.config_entries.options.async_configure(
        res_form["flow_id"], {"name": name}
    )

    sub = next(s for s in entry.get_subentries_of_type("device") if s.title == name)
    dev_uuid = sub.data["device_uuid"]
    ent_reg = er.async_get(hass)
    power_eid = ent_reg.async_get_entity_id(
        "sensor", "powerguesser", f"{entry.entry_id}_{dev_uuid}_power"
    )
    energy_eid = ent_reg.async_get_entity_id(
        "sensor", "powerguesser", f"{entry.entry_id}_{dev_uuid}_energy"
    )
    assert power_eid is not None and energy_eid is not None
    return dev_uuid, power_eid, energy_eid


async def _setup_live_base(hass: HomeAssistant, freezer: Any) -> float:
    """Pin the clock to the grid, set a clean 250 W base and settle; return remainder."""
    _aligned_move(freezer)
    async_set_power(hass, "sensor.grid_power", int(BASE_W))
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 5)
    return float(_state(hass, "sensor.powerguesser_remainder_power"))


async def _drive_on_period(
    hass: HomeAssistant,
    freezer: Any,
    power_eid: str,
    remainder_eid: str,
    remainder_before: float,
    devices: Any,
    dev_uuid: str,
    artefact_states: list[dict[str, Any]],
    w: float,
    n_steps: int,
    remainder_extra: float = 0.0,
    grid_base: float = BASE_W,
) -> None:
    """Drive one ON period of n_steps grid steps at +w W and assert the follow."""
    boxes = [(s["rise_w"]["lo"], s["rise_w"]["hi"]) for s in artefact_states]
    async_set_power(hass, "sensor.grid_power", int(round(grid_base + w)))
    await hass.async_block_till_done()
    for step in range(n_steps):
        await _drive_ticks(hass, freezer, 1)
        reason = devices.unavailable_reason(dev_uuid)
        assert reason is None, f"device became unavailable mid-run: {reason}"
        power = float(_state(hass, power_eid))
        if step == 0:
            # The hub's stream hold (seal/probe chain) confirms the edge at
            # the second high sample, so the first step is still transitional.
            continue
        assert power > 0.0, (
            f"device dropped to {power} W at ON step {step} "
            f"(reason={devices.unavailable_reason(dev_uuid)}, "
            f"remainder={hass.states.get(remainder_eid).state})"
        )
        if step == 1:
            # One grid sample after the on-edge the power is inside the rise
            # box of one of the device's states.
            assert any(lo - 1.0 <= power <= hi + 1.0 for lo, hi in boxes), (
                f"power {power} W outside every rise box {boxes}"
            )
        remainder = hass.states.get(remainder_eid)
        target = remainder_before + remainder_extra
        assert abs(float(remainder.state) - target) <= 150.0, (
            f"remainder moved by {float(remainder.state) - target:.1f} W while ON"
        )
        assert float(remainder.attributes["estimated_w"]) == pytest.approx(power, abs=1.0)


async def _drive_off_and_check_energy(
    hass: HomeAssistant,
    freezer: Any,
    power_eid: str,
    energy_eid: str,
    base_w: float,
    w: float,
    n_steps: int,
    energy_before: float,
) -> None:
    """Drop the load, assert 0.0 and energy growth of about w x ON time."""
    async_set_power(hass, "sensor.grid_power", int(round(base_w)))
    await hass.async_block_till_done()
    await _drive_ticks(hass, freezer, 2)
    assert _state(hass, power_eid) == "0.0"
    energy_kwh = float(_state(hass, energy_eid))
    expected_kwh = w * n_steps * GRID_S / 3_600_000.0
    # Back-dating (D-09) credits at most one extra grid interval.
    assert abs(energy_kwh - energy_before - expected_kwh) <= w * GRID_S / 3_600_000.0 * 1.5, (
        f"energy grew by {energy_kwh - energy_before:.5f} kWh, "
        f"expected ~{expected_kwh:.5f} kWh for {n_steps} steps at {w} W"
    )


async def test_oven_from_learning_run_drives_power_sensor(
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The oven learns as a two-state switched device and follows the live run."""
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()

    cand = await _run_learning_to_candidate(
        hass,
        entry,
        SynthSpec(days=22, kettle_per_day=(0, 0), oven=True),
        aioclient_mock,
        {
            "label": "Oven",
            "category": "oven",
            "target_w": 2200.0,
            "states": (("preheat", (8.0, 20.0)), ("cycle", (1.0, 5.0))),
        },
    )
    state_names = {s["name"] for s in cand["states"]}
    assert state_names == {"preheat", "cycle"}, f"unexpected states: {state_names}"

    dev_uuid, power_eid, energy_eid = await _confirm_and_entities(
        hass, entry, cand["signature_id"], "Oven"
    )
    devices = entry.runtime_data.devices
    reason = devices.unavailable_reason(dev_uuid)
    assert reason is None, f"device unavailable right after confirm: {reason!r}"

    # Live durations come from the confirmed active artefact; each must fall
    # inside its state's learned dur box (a failing precondition names the box).
    rec = devices.record(dev_uuid)
    artefact = rec["versions"][rec["active_v"] - 1]["artefact"]
    by_name = {s["name"]: s for s in artefact["states"]}
    n_pre = round(by_name["preheat"]["dur_s"]["p50"] / GRID_S)
    n_cyc = round(by_name["cycle"]["dur_s"]["p50"] / GRID_S)
    for name, n in (("preheat", n_pre), ("cycle", n_cyc)):
        box = by_name[name]["dur_s"]
        assert box["lo"] <= n * GRID_S <= box["hi"], (
            f"{name}: live {n * GRID_S}s outside dur box {box}"
        )

    remainder_eid = "sensor.powerguesser_remainder_power"
    remainder_before = await _setup_live_base(hass, freezer)
    assert _state(hass, power_eid) == "0.0"

    # Preheat: followed throughout, never forced OFF.
    energy_before = float(_state(hass, energy_eid))
    await _drive_on_period(
        hass, freezer, power_eid, remainder_eid, remainder_before,
        devices, dev_uuid, artefact["states"], 2200.0, n_pre,
    )
    await _drive_off_and_check_energy(
        hass, freezer, power_eid, energy_eid, BASE_W, 2200.0, n_pre, energy_before
    )

    # Two thermostat cycles, each reported as its own ON period.
    for cycle in range(2):
        energy_before = float(_state(hass, energy_eid))
        await _drive_on_period(
            hass, freezer, power_eid, remainder_eid, remainder_before,
            devices, dev_uuid, artefact["states"], 2200.0, n_cyc,
        )
        await _drive_off_and_check_energy(
            hass, freezer, power_eid, energy_eid, BASE_W, 2200.0, n_cyc, energy_before
        )

    reason = devices.unavailable_reason(dev_uuid)
    assert reason is None, f"device unavailable at the end of the live run: {reason!r}"


_HA_CASES: dict[str, dict[str, Any]] = {
    "kettle": {
        "spec": {"days": 22},
        "label": "Kettle",
        "category": "kettle",
        "target_w": 2000.0,
        "states": (("on", None),),
        "w": 2000.0,
    },
    "water_heater": {
        "spec": {"days": 22, "kettle_per_day": (0, 0), "water_heater": True},
        "label": "Water heater",
        "category": "water_heater",
        "target_w": 3000.0,
        "states": (("on", (15.0, 50.0)),),
        "w": 3000.0,
    },
    "dryer": {
        "spec": {"days": 22, "kettle_per_day": (0, 0), "dryer": True},
        "label": "Dryer element",
        "category": "dryer",
        "target_w": 2400.0,
        "states": (("on", (1.0, 6.0)),),
        "w": 2400.0,
    },
}


@pytest.mark.parametrize("name", sorted(_HA_CASES))
async def test_switched_load_from_learning_run_drives_power_sensor(
    name: str,
    recorder_mock: Any,
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    aioclient_mock: Any,
    inprocess_worker: Any,
    freezer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kettle, water heater and dryer element follow their live runs like the oven."""
    case = _HA_CASES[name]
    monkeypatch.setattr("custom_components.powerguesser.learning.MAX_REFINE_ROUNDS", 0)
    energy_manager.data = std_prefs
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()

    responder_kwargs = {
        "label": case["label"],
        "category": case["category"],
        "target_w": case["target_w"],
        "states": case["states"],
    }
    cand = await _run_learning_to_candidate(
        hass, entry, SynthSpec(**case["spec"]), aioclient_mock, responder_kwargs
    )
    dev_uuid, power_eid, energy_eid = await _confirm_and_entities(
        hass, entry, cand["signature_id"], case["label"]
    )
    devices = entry.runtime_data.devices
    reason = devices.unavailable_reason(dev_uuid)
    assert reason is None, f"device unavailable right after confirm: {reason!r}"

    rec = devices.record(dev_uuid)
    artefact = rec["versions"][rec["active_v"] - 1]["artefact"]
    states = artefact["states"]
    dur_p50 = states[0]["dur_s"]["p50"]

    remainder_eid = "sensor.powerguesser_remainder_power"
    remainder_before = await _setup_live_base(hass, freezer)
    assert _state(hass, power_eid) == "0.0"
    w = case["w"]

    if name == "water_heater":
        # One 25 min run: stays ON for the whole run, never forced OFF.
        n_on = 50
        max_on_s = 1.5 * max(s["dur_s"]["hi"] for s in states)
        assert max_on_s >= n_on * GRID_S, (
            f"stuck-ON cap {max_on_s}s shorter than the {n_on * GRID_S}s live run"
        )
        assert states[0]["dur_s"]["lo"] <= n_on * GRID_S <= states[0]["dur_s"]["hi"], (
            f"live run {n_on * GRID_S}s outside dur box {states[0]['dur_s']}"
        )
        energy_before = float(_state(hass, energy_eid))
        await _drive_on_period(
            hass, freezer, power_eid, remainder_eid, remainder_before,
            devices, dev_uuid, states, w, n_on,
        )
        await _drive_off_and_check_energy(
            hass, freezer, power_eid, energy_eid, BASE_W, w, n_on, energy_before
        )
    elif name == "dryer":
        # The drum-motor step alone never attributes to the element: the
        # device stays at 0.0 and the motor appears in the remainder (D-08).
        async_set_power(hass, "sensor.grid_power", int(BASE_W + 250.0))
        await hass.async_block_till_done()
        await _drive_ticks(hass, freezer, 3)
        assert _state(hass, power_eid) == "0.0", "the motor step attributed to the element"
        remainder = hass.states.get(remainder_eid)
        assert abs(float(remainder.state) - (remainder_before + 250.0)) <= 150.0, (
            f"motor step missing from the remainder: {float(remainder.state)} W"
        )
        # Two element cycles of 3 min at +2400 W on top of the motor, 2 min off.
        n_on = 6
        assert states[0]["dur_s"]["lo"] <= n_on * GRID_S <= states[0]["dur_s"]["hi"], (
            f"live cycle {n_on * GRID_S}s outside dur box {states[0]['dur_s']}"
        )
        for _cycle in range(2):
            energy_before = float(_state(hass, energy_eid))
            await _drive_on_period(
                hass, freezer, power_eid, remainder_eid, remainder_before,
                devices, dev_uuid, states, w, n_on, remainder_extra=250.0,
                grid_base=BASE_W + 250.0,
            )
            async_set_power(hass, "sensor.grid_power", int(BASE_W + 250.0))
            await hass.async_block_till_done()
            await _drive_ticks(hass, freezer, 4)  # 2 min off, motor still on
            assert _state(hass, power_eid) == "0.0"
            energy_kwh = float(_state(hass, energy_eid))
            expected_kwh = w * n_on * GRID_S / 3_600_000.0
            assert abs(energy_kwh - energy_before - expected_kwh) <= w * GRID_S / 3_600_000.0 * 1.5, (
                f"energy grew by {energy_kwh - energy_before:.5f} kWh, expected ~{expected_kwh:.5f} kWh"
            )
        # Motor off: everything back at base.
        async_set_power(hass, "sensor.grid_power", int(BASE_W))
        await hass.async_block_till_done()
        await _drive_ticks(hass, freezer, 2)
        assert _state(hass, power_eid) == "0.0"
        remainder = hass.states.get(remainder_eid)
        assert abs(float(remainder.state) - remainder_before) <= 150.0
    else:
        # Kettle: one run whose length comes from the learned dur p50.
        n_on = round(dur_p50 / GRID_S)
        assert states[0]["dur_s"]["lo"] <= n_on * GRID_S <= states[0]["dur_s"]["hi"], (
            f"live run {n_on * GRID_S}s outside dur box {states[0]['dur_s']}"
        )
        energy_before = float(_state(hass, energy_eid))
        await _drive_on_period(
            hass, freezer, power_eid, remainder_eid, remainder_before,
            devices, dev_uuid, states, w, n_on,
        )
        await _drive_off_and_check_energy(
            hass, freezer, power_eid, energy_eid, BASE_W, w, n_on, energy_before
        )

    reason = devices.unavailable_reason(dev_uuid)
    assert reason is None, f"device unavailable at the end of the live run: {reason!r}"
