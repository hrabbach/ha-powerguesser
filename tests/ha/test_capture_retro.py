"""End-to-end tracer tests for retroactive capture (CAP-01, CAP-04, CAP-06, CAP-09).

Plan 08-01 Task 1 proves the capture path end to end: service -> snap -> Store ->
Review preview. Task 2 adds capture_list, the tolerant loader and the admin /
not-loaded guards to this same file.
"""

import copy
import json
import unicodedata
from typing import Any

import pytest

from homeassistant.auth.models import User
from homeassistant.components.energy.data import EnergyManager
from homeassistant.core import Context, HomeAssistant
from homeassistant.exceptions import ServiceValidationError, Unauthorized
from homeassistant.util import dt as dt_util

from custom_components.powerguesser.const import (
    CANDIDATES_STORE_KEY,
    CAPTURES_STORE_KEY,
    CAPTURES_STORE_VERSION,
    DOMAIN,
)
from custom_components.powerguesser.engine.capture.preview import fit_preview
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate
from tests.conftest import (
    async_fill_synth_buffer,
    async_set_power,
    async_setup_via_flow,
)
from tests.fixtures.learning.synth import SynthSpec

GRID_S = 30

CAPTURE_WINDOW_KEYS = {
    "capture_id",
    "mode",
    "state",
    "target_kind",
    "target_id",
    "new_device_name",
    "label",
    "press_start_ts",
    "press_stop_ts",
    "window_start_ms",
    "window_end_ms",
    "created_at",
    "resolved_at",
    "reassigned_at",
    "reason_code",
    "gap_flagged",
    "result",
    "features",
    "trace",
    "holdout_excluded",
    "holdout_window_ms",
    "replay",
}


def _pin_grid(entry: Any) -> None:
    """Publish a 30 s profile snapshot so the capture snap uses a 30 s grid."""
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


async def _setup(
    hass: HomeAssistant, energy_manager: EnergyManager, std_prefs: dict[str, Any]
) -> tuple[Any, Any]:
    """Set up a PowerGuesser entry with a 3-day synthetic buffer and a pinned grid."""
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    data = await async_fill_synth_buffer(hass, entry, SynthSpec(days=3))
    return entry, data


def _iso_utc(ms: int) -> str:
    return dt_util.utc_from_timestamp(ms / 1000.0).isoformat()


def _stored_captures(hass_storage: dict[str, Any]) -> list[dict[str, Any]]:
    return hass_storage.get(CAPTURES_STORE_KEY, {}).get("data", {}).get("captures", [])


async def _capture(
    hass: HomeAssistant,
    *,
    start_ms: int,
    end_ms: int,
    label: str = "kettle full",
    target_type: str = "new_device",
    target: str = "Kettle",
) -> dict[str, Any]:
    return await hass.services.async_call(
        DOMAIN,
        "capture_window",
        {
            "start": _iso_utc(start_ms),
            "end": _iso_utc(end_ms),
            "label": label,
            "target_type": target_type,
            "target": target,
        },
        blocking=True,
        return_response=True,
    )


async def _capture_list(
    hass: HomeAssistant, capture_id: str | None = None
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if capture_id is not None:
        data["capture_id"] = capture_id
    return await hass.services.async_call(
        DOMAIN, "capture_list", data, blocking=True, return_response=True
    )


async def test_tracer_capture_window_creates_preview(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A retroactive new-device capture snaps the run and shows a preview in Review."""
    entry, data = await _setup(hass, energy_manager, std_prefs)
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]

    res = await _capture(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)

    result = res["result"]
    assert result["verdict"] == "clean"
    assert abs(result["rise_ms"] - on_ms) <= 33_000
    assert abs(result["fall_ms"] - off_ms) <= 33_000
    assert abs(result["step_w"] - 2000.0) <= 200.0
    assert abs(result["dur_s"] - 180.0) <= 60.0
    assert result["energy_wh"] > 0
    # The window echo and the replay key ride on the response.
    assert res["window"]["tz"]
    assert res["window"]["start_utc"]
    assert res["window"]["start_local"]
    assert "replay" in res and res["replay"] is None
    assert res["state"] == "done"
    assert res["target"]["kind"] == "candidate"
    assert res["target"]["name"] == "Kettle"

    visible = hub.learning.visible_candidates()
    assert len(visible) == 1
    cand = visible[0]
    assert cand["signature_id"].startswith("cap_")
    assert cand["origin"] == "capture"
    assert cand["scores"]["validated"] is False
    assert cand["scores"]["n_examples"] == 1

    label = hub.learning.candidate_label(cand)
    assert label.startswith("Kettle")
    assert "from 1 captured example" in label
    assert "preview, not yet validated" in label

    # The preview is never written into the candidates Store.
    stored = (
        hass_storage.get(CANDIDATES_STORE_KEY, {})
        .get("data", {})
        .get("candidates", [])
    )
    assert all(not str(c.get("signature_id", "")).startswith("cap_") for c in stored)
    # A capture creates no device record.
    assert hub.devices.owned_signature_ids() == set()


async def test_preview_boxes_follow_d02_tolerance() -> None:
    """The n = 1 preview boxes are the captured values plus/minus the D-02 tolerance."""
    example = {
        "capture_id": "c1",
        "result": {
            "verdict": "clean",
            "step_w": 2000.0,
            "fall_w": -2000.0,
            "plateau_w": 1980.0,
            "dur_s": 180.0,
            "threshold_w": 120.0,
            "grid_s": 30.0,
        },
    }
    art = fit_preview("cap_x", "Kettle", [example])
    assert art is not None
    state = art["states"][0]
    # rise: +-max(0.15 * 2000, 120 / 2) = +-300
    assert state["rise_w"] == {"lo": 1700.0, "p50": 2000.0, "hi": 2300.0}
    # fall: the same rule on its own (negative) value
    assert state["fall_w"] == {"lo": -2300.0, "p50": -2000.0, "hi": -1700.0}
    # plateau: +-max(0.15 * 1980, 60) = +-297
    assert state["plateau_w"] == {"lo": 1683.0, "p50": 1980.0, "hi": 2277.0}
    # duration: +-max(0.30 * 180, 30) = +-54, lo floored at 0
    assert state["dur_s"] == {"lo": 126.0, "p50": 180.0, "hi": 234.0}
    assert state["phase_share"] is None


async def test_capture_record_schema_is_complete(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """The stored record carries exactly the 22 D-05 keys and the holdout marker."""
    entry, data = await _setup(hass, energy_manager, std_prefs)
    on_ms, off_ms = data.kettle_runs[-1]

    await _capture(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)

    records = _stored_captures(hass_storage)
    assert len(records) == 1
    rec = records[0]
    assert set(rec.keys()) == CAPTURE_WINDOW_KEYS
    assert rec["holdout_excluded"] is True
    grid_s = int(rec["result"]["grid_s"])
    assert rec["holdout_window_ms"] == [
        rec["result"]["rise_ms"],
        rec["result"]["fall_ms"] + grid_s * 1000,
    ]


async def test_capture_window_without_pair_fails_with_none(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A window over a quiet hour stores a failed record with no_edge_pair and no preview."""
    entry, data = await _setup(hass, energy_manager, std_prefs)
    hub = entry.runtime_data
    off_ms = data.kettle_runs[-1][1]
    quiet_start = off_ms + 45 * 60 * 1000

    res = await _capture(
        hass, start_ms=quiet_start, end_ms=quiet_start + 5 * 60 * 1000
    )

    assert res["result"]["verdict"] == "none"
    assert res["result"]["reason_code"] == "no_edge_pair"
    records = _stored_captures(hass_storage)
    assert len(records) == 1
    assert records[0]["state"] == "failed"
    assert hub.learning.visible_candidates() == []


async def test_capture_name_is_normalised_or_refused(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A target name is NFC-normalised and whitespace-collapsed, or refused."""
    entry, data = await _setup(hass, energy_manager, std_prefs)
    on_ms, off_ms = data.kettle_runs[-1]

    raw = "  Ke\u0301ttle  "
    res = await _capture(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000, target=raw)
    assert res["new_device_name"] == "K\u00e9ttle"
    assert res["new_device_name"] == unicodedata.normalize("NFC", raw).strip()

    count = len(_stored_captures(hass_storage))

    with pytest.raises(ServiceValidationError) as long_err:
        await _capture(
            hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000, target="K" * 65
        )
    assert long_err.value.translation_key == "capture_invalid_name"

    with pytest.raises(ServiceValidationError) as ctrl_err:
        await _capture(
            hass,
            start_ms=on_ms - 45_000,
            end_ms=off_ms + 50_000,
            target="Ket\u0001tle",
        )
    assert ctrl_err.value.translation_key == "capture_invalid_name"

    # Neither refused call created a record.
    assert len(_stored_captures(hass_storage)) == count


# ---------------------------------------------------------------------------
# Task 2: capture_list, the tolerant loader and the admin / not-loaded guards
# ---------------------------------------------------------------------------


async def test_capture_list_returns_all_or_one(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """capture_list returns every record (or one), never rewriting the Store."""
    entry, data = await _setup(hass, energy_manager, std_prefs)
    on_ms, off_ms = data.kettle_runs[-1]
    await _capture(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000, target="Kettle")
    await _capture(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000, target="Toaster")

    before = json.dumps(_stored_captures(hass_storage), sort_keys=True)
    res = await _capture_list(hass)
    caps = res["captures"]
    assert len(caps) == 2
    assert caps[0]["created_at"] <= caps[1]["created_at"]
    names = set()
    for cap in caps:
        assert cap["result"]["verdict"] == "clean"
        assert cap["window"]["start_utc"]
        assert cap["target"]["kind"] == "candidate"
        assert cap["target"]["id"].startswith("cap_")
        names.add(cap["target"]["name"])
    assert names == {"Kettle", "Toaster"}

    one = await _capture_list(hass, capture_id=caps[0]["capture_id"])
    assert len(one["captures"]) == 1
    assert one["captures"][0]["capture_id"] == caps[0]["capture_id"]

    assert (await _capture_list(hass, capture_id="nope"))["captures"] == []

    after = json.dumps(_stored_captures(hass_storage), sort_keys=True)
    assert before == after


async def test_capture_list_marks_gone_target(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A dismissed preview's captures read target null and target_state target_gone."""
    entry, data = await _setup(hass, energy_manager, std_prefs)
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]
    await _capture(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000, target="Kettle")

    rec = _stored_captures(hass_storage)[0]
    sig_id = rec["target_id"]
    fingerprint = hub.learning.review.decisions.get(sig_id, {}).get("fingerprint") or {}
    await hub.learning.review.async_dismiss(sig_id, fingerprint, label="Kettle")

    before = json.dumps(_stored_captures(hass_storage), sort_keys=True)
    res = await _capture_list(hass)
    assert res["captures"][0]["target"] is None
    assert res["captures"][0]["target_state"] == "target_gone"
    after = json.dumps(_stored_captures(hass_storage), sort_keys=True)
    assert before == after


async def test_capture_services_admin_only(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_read_only_user: User,
) -> None:
    """A non-admin user calling capture_window or capture_list raises Unauthorized."""
    entry, data = await _setup(hass, energy_manager, std_prefs)
    on_ms, off_ms = data.kettle_runs[-1]
    ctx = Context(user_id=hass_read_only_user.id)

    with pytest.raises(Unauthorized):
        await hass.services.async_call(
            DOMAIN,
            "capture_window",
            {
                "start": _iso_utc(on_ms - 45_000),
                "end": _iso_utc(off_ms + 50_000),
                "label": "kettle full",
                "target_type": "new_device",
                "target": "Kettle",
            },
            blocking=True,
            return_response=True,
            context=ctx,
        )

    with pytest.raises(Unauthorized):
        await hass.services.async_call(
            DOMAIN,
            "capture_list",
            {},
            blocking=True,
            return_response=True,
            context=ctx,
        )


async def test_captures_store_drops_malformed_records(
    hass: HomeAssistant,
    energy_manager: EnergyManager,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """A malformed captures Store record is dropped at load and setup still succeeds."""
    valid: dict[str, Any] = {
        "capture_id": "abc",
        "mode": "window",
        "state": "done",
        "target_kind": "candidate",
        "target_id": "cap_valid",
        "new_device_name": "Kettle",
        "label": "kettle full",
        "press_start_ts": None,
        "press_stop_ts": None,
        "window_start_ms": 0,
        "window_end_ms": 1000,
        "created_at": "2026-11-20T10:00:00+00:00",
        "resolved_at": "2026-11-20T10:00:00+00:00",
        "reassigned_at": None,
        "reason_code": None,
        "gap_flagged": False,
        "result": {"verdict": "clean", "grid_s": 30},
        "features": None,
        "trace": None,
        "holdout_excluded": True,
        "holdout_window_ms": [0, 1000],
        "replay": None,
    }
    missing_target = {**valid, "capture_id": "m"}
    del missing_target["target_id"]
    malformed: list[Any] = [
        "not a dict",
        {**valid, "capture_id": "s", "state": "bogus"},
        missing_target,
        {**valid, "capture_id": "t", "created_at": "not-a-date"},
    ]
    hass_storage[CAPTURES_STORE_KEY] = {
        "version": CAPTURES_STORE_VERSION,
        "minor_version": 1,
        "key": CAPTURES_STORE_KEY,
        "data": {"captures": [valid, *malformed]},
    }

    entry, _data = await _setup(hass, energy_manager, std_prefs)

    records = entry.runtime_data.captures.records
    assert len(records) == 1
    assert records[0]["capture_id"] == "abc"


async def test_capture_not_loaded(hass: HomeAssistant) -> None:
    """Both capture services raise not_loaded without a loaded entry."""
    from custom_components.powerguesser import async_setup

    assert await async_setup(hass, {})

    with pytest.raises(ServiceValidationError) as window_err:
        await hass.services.async_call(
            DOMAIN,
            "capture_window",
            {
                "start": _iso_utc(0),
                "end": _iso_utc(1000),
                "label": "k",
                "target_type": "new_device",
                "target": "Kettle",
            },
            blocking=True,
            return_response=True,
        )
    assert window_err.value.translation_key == "not_loaded"

    with pytest.raises(ServiceValidationError) as list_err:
        await hass.services.async_call(
            DOMAIN, "capture_list", {}, blocking=True, return_response=True
        )
    assert list_err.value.translation_key == "not_loaded"
