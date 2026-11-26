"""HA tests for the D-07 dismissed-shape note and the D-06 orphan notice (Plan 08-07 Task 2).

A deliberate capture of a dismissed preview's shape stays visible in Review with
a "similar to a candidate you dismissed" label note and a ``similar_dismissed``
capture result; orphaned captures surface in Review either as a
``captures_notice`` placeholder (when candidates remain) or as the
``orphan_captures`` abort carrying the count, in the locked D-06 wording.
"""

import copy
import json
from pathlib import Path
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.util import dt as dt_util

from custom_components.powerguesser.const import DOMAIN
from custom_components.powerguesser.engine.intervals import ProfileSnapshot, gate
from tests.conftest import (
    async_fill_synth_buffer,
    async_open_options_step,
    async_set_power,
    async_setup_via_flow,
    make_switched_candidate,
    seed_candidates,
)
from tests.fixtures.learning.synth import SynthSpec

_GRID_S = 30


def _pin_grid(entry: Any) -> None:
    enabled, disabled = gate(float(_GRID_S), False)
    entry.runtime_data.profile_coordinator.async_set_updated_data(
        ProfileSnapshot(
            sources={},
            effective_s=float(_GRID_S),
            assumed=False,
            enabled=enabled,
            disabled=disabled,
        )
    )


def _iso_utc(ms: int) -> str:
    return dt_util.utc_from_timestamp(ms / 1000.0).isoformat()


async def _setup_capture(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
    *,
    candidates: list[dict[str, Any]] | None = None,
    spec: SynthSpec | None = None,
) -> tuple[Any, Any]:
    if candidates:
        seed_candidates(hass_storage, candidates)
    energy_manager.data = copy.deepcopy(std_prefs)
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 0)
    async_set_power(hass, "sensor.battery_power", 0)
    async_set_power(hass, "sensor.washer_power", 0)
    entry = await async_setup_via_flow(hass)
    _pin_grid(entry)
    await hass.async_block_till_done()
    data = await async_fill_synth_buffer(hass, entry, spec or SynthSpec(days=3))
    return entry, data


async def _capture_window(
    hass: HomeAssistant,
    *,
    start_ms: int,
    end_ms: int,
    label: str = "run",
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


async def _dismiss(hass: HomeAssistant, entry: Any, sig_id: str) -> None:
    """Dismiss a preview/candidate through the Review flow."""
    res_review = await async_open_options_step(hass, entry, "review")
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": sig_id}
    )
    await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "dismiss"}
    )


async def test_capture_of_dismissed_shape_is_visible_with_note(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """Recapturing a dismissed preview's shape stays visible with a note."""
    entry, data = await _setup_capture(hass, energy_manager, std_prefs, hass_storage)
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]

    first = await _capture_window(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    dismissed_id = first["target"]["id"]
    # No dismissed candidate yet: neither note nor similar_dismissed.
    assert first["result"]["similar_dismissed"] is None
    before = next(
        c
        for c in hub.learning.visible_candidates()
        if c["signature_id"] == dismissed_id
    )
    assert "similar to a candidate you dismissed" not in hub.learning.candidate_label(
        before
    )

    await _dismiss(hass, entry, dismissed_id)
    assert hub.learning.review.decisions[dismissed_id]["status"] == "dismissed"

    # Capture the same run again as a new device: the preview matches D's boxes.
    second = await _capture_window(
        hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000, target="Kettle again"
    )
    assert second["result"]["similar_dismissed"] == dismissed_id

    visible = hub.learning.visible_candidates()
    assert len(visible) == 1
    label = hub.learning.candidate_label(visible[0])
    assert label.endswith(" — similar to a candidate you dismissed")


async def test_review_shows_orphan_notice(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """The review form's captures_notice carries the D-06 wording for orphans, else ''."""
    cand = make_switched_candidate("sig-toaster", "Toaster")
    entry, data = await _setup_capture(
        hass, energy_manager, std_prefs, hass_storage, candidates=[cand]
    )
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]

    review = await async_open_options_step(hass, entry, "review")
    assert review["type"] is FlowResultType.FORM
    assert review["description_placeholders"]["captures_notice"] == ""

    first = await _capture_window(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    sig = first["target"]["id"]
    await _capture_window(
        hass,
        start_ms=on_ms - 45_000,
        end_ms=off_ms + 50_000,
        target_type="candidate",
        target="Kettle",
    )
    await _dismiss(hass, entry, sig)
    assert len(hub.captures.orphans()) == 2

    review = await async_open_options_step(hass, entry, "review")
    assert review["type"] is FlowResultType.FORM
    notice = review["description_placeholders"]["captures_notice"]
    assert "2 captured run(s)" in notice
    assert "target no longer exists: reassign or delete" in notice


async def test_review_orphan_only_aborts_with_count(
    hass: HomeAssistant,
    energy_manager: Any,
    std_prefs: dict[str, Any],
    hass_storage: dict[str, Any],
) -> None:
    """With no candidates but an orphan capture, Review aborts orphan_captures; else no_candidates."""
    entry, data = await _setup_capture(hass, energy_manager, std_prefs, hass_storage)
    hub = entry.runtime_data
    on_ms, off_ms = data.kettle_runs[-1]

    # No candidates and no captures: still aborts no_candidates.
    result = await async_open_options_step(hass, entry, "review")
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "no_candidates"

    first = await _capture_window(hass, start_ms=on_ms - 45_000, end_ms=off_ms + 50_000)
    sig = first["target"]["id"]
    await _dismiss(hass, entry, sig)
    assert hub.learning.visible_candidates() == []
    assert len(hub.captures.orphans()) == 1

    result = await async_open_options_step(hass, entry, "review")
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "orphan_captures"
    assert result["description_placeholders"]["count"] == "1"
    strings = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "custom_components"
            / "powerguesser"
            / "strings.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        "target no longer exists: reassign or delete"
        in strings["options"]["abort"]["orphan_captures"]
    )
