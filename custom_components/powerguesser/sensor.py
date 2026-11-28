"""Sensor platform for PowerGuesser."""

from typing import Any

import homeassistant.util.dt as dt_util

from homeassistant.components.sensor import (
    RestoreSensor,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import (
    PERCENTAGE,
    EntityCategory,
    UnitOfEnergy,
    UnitOfPower,
    UnitOfTime,
)
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
)

from . import PowerGuesserConfigEntry
from .const import (
    CONF_LLM_MODEL,
    CONF_LLM_PROBE,
    DOMAIN,
    SIGNAL_CAPTURE_UPDATED,
    SIGNAL_PROFILE_SOURCES,
)
from .engine.intervals import ProfileSnapshot, ProfileSource, SourceProfile
from .engine.llm_wire import model_matches
from .engine.model import Snapshot

UNRECORDED_ATTRIBUTES = frozenset(
    {
        "source_age_s",
        "stale_sources",
        "over_attribution",
        "clipped",
        "metered_w",
        "metered_unavailable",
        "solar_unavailable",
        "on_change_sources",
        "estimated_w",
    }
)


def _device_info() -> DeviceInfo:
    """Return shared DeviceInfo for PowerGuesser entities."""
    return DeviceInfo(
        identifiers={(DOMAIN, DOMAIN)},
        name="PowerGuesser",
        entry_type=DeviceEntryType.SERVICE,
    )


class _PowerGuesserSensor(
    CoordinatorEntity[DataUpdateCoordinator[Snapshot]], SensorEntity
):
    """Base class for PowerGuesser power sensors."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_device_class = SensorDeviceClass.POWER
    _attr_native_unit_of_measurement = UnitOfPower.WATT
    _attr_state_class = SensorStateClass.MEASUREMENT
    _unrecorded_attributes = UNRECORDED_ATTRIBUTES

    def __init__(self, coordinator: DataUpdateCoordinator[Snapshot]) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._attr_device_info = _device_info()

    @property
    def available(self) -> bool:
        """Return True if sensor is available."""
        return (
            super().available
            and self.coordinator.data is not None
            and self.coordinator.data.available
        )


class GrossPowerSensor(_PowerGuesserSensor):
    """Sensor exposing gross household power consumption."""

    _attr_translation_key = "gross_power"
    _attr_unique_id = "powerguesser_gross_power"

    @property
    def native_value(self) -> float | None:
        """Return the current gross power reading in W."""
        if self.coordinator.data is None:
            return None
        return self.coordinator.data.gross_w

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        if self.coordinator.data is None:
            return {
                "source_age_s": {},
                "stale_sources": [],
                "solar_unavailable": [],
                "on_change_sources": [],
                "plausibility": "ok",
            }
        return {
            "source_age_s": dict(self.coordinator.data.source_age_s),
            "stale_sources": list(self.coordinator.data.stale_sources),
            "solar_unavailable": list(self.coordinator.data.solar_unavailable),
            "on_change_sources": list(self.coordinator.data.on_change_sources),
            "plausibility": self.coordinator.data.plausibility,
        }



class RemainderPowerSensor(_PowerGuesserSensor):
    """Sensor exposing unmetered remainder power consumption."""

    _attr_translation_key = "remainder_power"
    _attr_unique_id = "powerguesser_remainder_power"

    @property
    def native_value(self) -> float | None:
        """Return the current remainder power reading in W."""
        if self.coordinator.data is None:
            return None
        return self.coordinator.data.remainder_w

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return entity specific state attributes."""
        if self.coordinator.data is None:
            return {
                "metered_w": 0.0,
                "metered_unavailable": [],
                "over_attribution": 0.0,
                "clipped": False,
                "estimated_w": 0.0,
                "source_age_s": {},
                "stale_sources": [],
            }
        return {
            "metered_w": round(self.coordinator.data.metered_w, 1),
            "metered_unavailable": list(self.coordinator.data.metered_unavailable),
            "over_attribution": round(self.coordinator.data.over_attribution_wh, 2),
            "clipped": self.coordinator.data.clipped,
            "estimated_w": round(self.coordinator.data.estimated_w, 1),
            "source_age_s": dict(self.coordinator.data.source_age_s),
            "stale_sources": list(self.coordinator.data.stale_sources),
        }


class SampleIntervalSensor(
    CoordinatorEntity[DataUpdateCoordinator[ProfileSnapshot]], SensorEntity
):
    """Sensor exposing sample interval for an individual power source."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_device_class = SensorDeviceClass.DURATION
    _attr_native_unit_of_measurement = UnitOfTime.SECONDS
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_suggested_display_precision = 0
    _attr_translation_key = "sample_interval"
    _unrecorded_attributes = frozenset({"p90_s", "delta_count"})

    def __init__(
        self,
        coordinator: DataUpdateCoordinator[ProfileSnapshot],
        source: ProfileSource,
    ) -> None:
        """Initialize the sample interval sensor."""
        super().__init__(coordinator)
        self.source = source
        self.source_entity_id = source.entity_id
        self._attr_unique_id = f"powerguesser_sample_interval_{source.entity_id}"
        self._attr_translation_placeholders = {"source": source.label}
        self._attr_device_info = _device_info()

    @property
    def _profile(self) -> SourceProfile | None:
        if self.coordinator.data is None:
            return None
        return self.coordinator.data.sources.get(self.source_entity_id)

    @property
    def native_value(self) -> int | None:
        """Return native value in whole seconds."""
        profile = self._profile
        if profile is None or profile.median_s is None:
            return None
        return round(profile.median_s)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return diagnostic attributes."""
        profile = self._profile
        if profile is None:
            return {
                "source_entity_id": self.source_entity_id,
                "mode": "on_change",
                "measured_from": None,
                "restored": False,
                "p90_s": None,
                "delta_count": 0,
            }
        return {
            "source_entity_id": self.source_entity_id,
            "mode": (
                profile.mode.value
                if hasattr(profile.mode, "value")
                else str(profile.mode)
            ),
            "measured_from": (
                profile.measured_from.value
                if profile.measured_from is not None
                else None
            ),
            "restored": profile.restored,
            "p90_s": round(profile.p90_s, 1) if profile.p90_s is not None else None,
            "delta_count": profile.delta_count,
        }


class EffectiveSampleIntervalSensor(
    CoordinatorEntity[DataUpdateCoordinator[ProfileSnapshot]], SensorEntity
):
    """Sensor exposing effective gross sample interval and event type gating."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_device_class = SensorDeviceClass.DURATION
    _attr_native_unit_of_measurement = UnitOfTime.SECONDS
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_suggested_display_precision = 0
    _attr_translation_key = "effective_sample_interval"
    _attr_unique_id = "powerguesser_effective_sample_interval"

    def __init__(
        self,
        coordinator: DataUpdateCoordinator[ProfileSnapshot],
    ) -> None:
        """Initialize effective sample interval sensor."""
        super().__init__(coordinator)
        self._attr_device_info = _device_info()

    @property
    def native_value(self) -> int | None:
        """Return effective interval in whole seconds."""
        if self.coordinator.data is None:
            return 30
        return round(self.coordinator.data.effective_s)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return gating attributes."""
        if self.coordinator.data is None:
            return {
                "enabled_event_types": ["step", "cycle_shape", "continuous"],
                "disabled_event_types": {},
                "interval_assumed": True,
            }
        return {
            "enabled_event_types": [e.value for e in self.coordinator.data.enabled],
            "disabled_event_types": {
                e.value: reason for e, reason in self.coordinator.data.disabled.items()
            },
            "interval_assumed": self.coordinator.data.assumed,
        }


class BootstrappedUsableHistorySensor(
    CoordinatorEntity[DataUpdateCoordinator[dict[str, Any]]], SensorEntity
):
    """Diagnostic sensor exposing bootstrapped usable history duration."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_translation_key = "bootstrapped_usable_history"
    _attr_unique_id = "powerguesser_bootstrapped_usable_history"
    _attr_device_class = SensorDeviceClass.DURATION
    _attr_native_unit_of_measurement = UnitOfTime.DAYS
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_suggested_display_precision = 1
    _attr_state_class = None
    _unrecorded_attributes = frozenset({"channels"})

    def __init__(
        self, coordinator: DataUpdateCoordinator[dict[str, Any]]
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._attr_device_info = _device_info()

    @property
    def native_value(self) -> float | None:
        """Return usable history in days."""
        if not self.coordinator.data:
            return None
        return self.coordinator.data.get("usable_days")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return bootstrap status, report timestamp, and per-channel details."""
        if not self.coordinator.data:
            return {
                "bootstrap_status": None,
                "computed_at": None,
                "channels": {},
            }
        return {
            "bootstrap_status": self.coordinator.data.get("status"),
            "computed_at": self.coordinator.data.get("computed_at"),
            "channels": self.coordinator.data.get("channels", {}),
        }


class LearningStatusSensor(
    CoordinatorEntity[DataUpdateCoordinator[dict[str, Any]]], SensorEntity
):
    """Diagnostic sensor reporting the status and results of device learning."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_translation_key = "learning_status"
    _attr_unique_id = "powerguesser_learning_status"
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = ["idle", "running", "blocked", "capped", "failed"]
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    # Failure counters (``evidence_failures``, ``rescore_failures``) are
    # deliberately NOT listed here: both are low-cardinality counters kept in
    # recorder history for trend debugging.
    _unrecorded_attributes = frozenset(
        {
            "tokens_this_run",
            "tokens_this_month",
            "cost_this_month",
            "last_error",
            "last_result",
            "last_attempt_start",
            "last_attempt_end",
            "last_attempt_outcome",
            "model_mismatch_this_run",
            "trimmed_this_run",
            "captures_left_out_this_run",
        }
    )

    def __init__(
        self,
        coordinator: DataUpdateCoordinator[dict[str, Any]],
        entry: PowerGuesserConfigEntry | None = None,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._entry = entry
        self._attr_device_info = _device_info()

    def _probe_model_attrs(self) -> tuple[bool | None, str | None]:
        """Return (model mismatch, returned model) of the stored endpoint probe."""
        if self._entry is None:
            return None, None
        probe = self._entry.data.get(CONF_LLM_PROBE)
        if not isinstance(probe, dict):
            return None, None
        returned = probe.get("model_returned")
        mismatch = probe.get("model_mismatch")
        if mismatch is None:
            # Probe stored before the mismatch check existed
            requested = self._entry.data.get(CONF_LLM_MODEL)
            mismatch = (
                not model_matches(str(requested), returned) if requested else None
            )
        return (bool(mismatch) if mismatch is not None else None), returned

    @property
    def native_value(self) -> str | None:
        """Return the current learning state."""
        if not self.coordinator.data:
            return "idle"
        return self.coordinator.data.get("status", "idle")

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return learning run and token attributes."""
        data = self.coordinator.data or {}
        last_run = data.get("last_run")
        last_result = None
        if last_run:
            last_result = {
                "stage": last_run.get("stage"),
                "candidates": last_run.get("candidates"),
                "train_share": last_run.get("train_share"),
                "val_share": last_run.get("val_share"),
                "preview": last_run.get("preview"),
                "rounds": last_run.get("rounds"),
                "accepted_round": last_run.get("accepted_round"),
                "model_mismatch": last_run.get("model_mismatch"),
                "trimmed_fields": last_run.get("trimmed_fields", 0),
            }
            if "days" in last_run:
                last_result["days"] = last_run["days"]

        tokens_run = data.get("tokens_run") or {}
        tokens_this_run = int(tokens_run.get("input", 0)) + int(tokens_run.get("output", 0))

        month_data = data.get("month") or {}
        curr_month = dt_util.now().strftime("%Y-%m")
        if month_data.get("key") == curr_month:
            tokens_this_month = int(month_data.get("input", 0)) + int(month_data.get("output", 0))
            cost_raw = month_data.get("cost")
            cost_this_month = round(float(cost_raw), 4) if cost_raw is not None else None
        else:
            tokens_this_month = 0
            cost_this_month = None

        usable_days_data = data.get("usable_days") or {}
        buffer_usable_days = usable_days_data.get("value")
        llm_model_mismatch, llm_model_returned = self._probe_model_attrs()

        # CLEAN-02 (D-15): the newest attempt, whatever its outcome. outcome is one of
        # running, success, failed, capped, stopped, interrupted ("interrupted" is the
        # D-15a restore value for a run the previous process left still running -- an
        # unclean restart; a graceful restart records it as "stopped").
        last_attempt = data.get("last_attempt")
        if not isinstance(last_attempt, dict):
            last_attempt = {}

        return {
            "last_run_start": last_run.get("started") if last_run else None,
            "last_run_end": last_run.get("ended") if last_run else None,
            "last_attempt_start": last_attempt.get("started"),
            "last_attempt_end": last_attempt.get("ended"),
            "last_attempt_outcome": last_attempt.get("outcome"),
            "last_result": last_result,
            "pending_candidates": data.get("pending_candidates", 0),
            "tokens_this_run": tokens_this_run,
            "tokens_this_month": tokens_this_month,
            "cost_this_month": cost_this_month,
            "running_job": data.get("running_job"),
            "rescore_failures": data.get("rescore_failures", 0),
            "evidence_failures": data.get("evidence_failures", 0),
            "last_error": data.get("last_error"),
            "buffer_usable_days": buffer_usable_days,
            "blocked_reason": data.get("blocked_reason"),
            "llm_model_mismatch": llm_model_mismatch,
            "llm_model_returned": llm_model_returned,
            "model_mismatch_this_run": data.get("model_mismatch_run"),
            "trimmed_this_run": data.get("trimmed_run", 0),
            "captures_left_out_this_run": data.get("captures_left_out_run", 0),
        }


class CaptureStatusSensor(SensorEntity):
    """Hub-level enum sensor reporting the live capture state (CAP-07, D-03).

    A plain ``SensorEntity`` (no coordinator): it derives its state from
    ``hub.captures.status_view()`` and refreshes through the manager's
    ``SIGNAL_CAPTURE_UPDATED`` dispatcher signal. The state changes only on
    capture transitions; no attribute ticks.
    """

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_translation_key = "capture_status"
    _attr_unique_id = "powerguesser_capture_status"
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = ["idle", "recording", "resolving"]
    _unrecorded_attributes = frozenset(
        {"capture_id", "target_name", "started_at", "last_result"}
    )

    def __init__(self, hub: Any, entry: PowerGuesserConfigEntry) -> None:
        """Initialize the capture status sensor."""
        self.hub = hub
        self._entry_id = entry.entry_id
        self._attr_device_info = _device_info()

    @property
    def native_value(self) -> str:
        """Return the capture state (idle, recording or resolving)."""
        return self.hub.captures.status_view()["state"]

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the open capture's volatile attributes and the last result."""
        view = self.hub.captures.status_view()
        return {
            "capture_id": view["capture_id"],
            "target_kind": view["target_kind"],
            "target_name": view["target_name"],
            "started_at": view["started_at"],
            "last_result": view["last_result"],
        }

    async def async_added_to_hass(self) -> None:
        """Subscribe to capture state changes."""
        self.async_on_remove(
            async_dispatcher_connect(
                self.hass,
                f"{SIGNAL_CAPTURE_UPDATED}_{self._entry_id}",
                self.async_write_ha_state,
            )
        )


def _device_model(hub: Any, device_uuid: str) -> str:
    """Return the DeviceInfo model string for a confirmed device record (D-05).

    A program appliance reads "Estimated program appliance"; every other
    record (and a missing one) keeps the switched label. HA does not translate
    DeviceInfo model strings (per convention), so these are plain literals.
    """
    rec = hub.devices.record(device_uuid) or {}
    return (
        "Estimated program appliance"
        if rec.get("kind") == "program"
        else "Estimated switching load"
    )


class DevicePowerSensor(_PowerGuesserSensor):
    """Per-device virtual power sensor for a confirmed device.

    Reports 0.0 W while the device is OFF; Plan 04-04's detector sets ON
    devices' power behind the same DeviceRuntime.power_w method.
    """

    _attr_translation_key = "device_power"
    _unrecorded_attributes = frozenset(
        {
            "match_score",
            "unattributed_edges",
            "update_available",
            "overlap_refused",
            "validation_session_start",
            "validation_session_end",
        }
    )

    def __init__(
        self,
        hub: Any,
        entry_id: str,
        device_uuid: str,
        device_name: str,
    ) -> None:
        """Initialize the device power sensor."""
        super().__init__(hub.coordinator)
        self._hub = hub
        self._uuid = device_uuid
        self._attr_unique_id = f"{entry_id}_{device_uuid}_power"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, device_uuid)},
            name=device_name,
            manufacturer="PowerGuesser",
            model=_device_model(hub, device_uuid),
        )

    @property
    def available(self) -> bool:
        """Return True when this device's power estimate is available.

        Device availability follows the detector (D-28/D-29/D-31), not the
        raw source availability in the snapshot: a source gap marks an ON
        device state_unknown and leaves an OFF device at 0.0, so the sensor
        never reads unavailable just because a source dropped out. Only the
        hub coordinator itself failing (last_update_success) hides it.
        """
        return (
            self.coordinator.last_update_success
            and self._hub.devices.power_available(self._uuid)
        )

    @property
    def native_value(self) -> float:
        """Return the current device power reading in W (0.0 while OFF)."""
        return self._hub.devices.power_w(self._uuid)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the static OUT-04 attributes plus the live fit (D-14, D-17).

        The static keys keep their fixed order and the live keys
        (match_score, unattributed_edges) follow in fixed order, so the key
        order is identical on every state write, ON or OFF.
        """
        return {
            **self._hub.devices.attributes(self._uuid),
            **self._hub.devices.live_attributes(self._uuid),
        }


class DeviceEnergySensor(
    CoordinatorEntity[DataUpdateCoordinator[Snapshot]], RestoreSensor
):
    """Per-device virtual energy sensor for a confirmed device.

    TOTAL_INCREASING counter fed from the Store-canonical hot total; the
    value never decreases. RestoreSensor is the secondary fallback only
    (HA discards restore data of entities not re-added for 7 days, and
    dumps every 15 minutes): on add, a numeric restored value raises the
    floor to restored + increments since load, never lowering and never
    double-counting (T-04-14).
    """

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_translation_key = "device_energy"
    _attr_device_class = SensorDeviceClass.ENERGY
    _attr_native_unit_of_measurement = UnitOfEnergy.KILO_WATT_HOUR
    _attr_state_class = SensorStateClass.TOTAL_INCREASING
    _attr_suggested_display_precision = 3
    # Plan 06-09 (REV-04, D-13): the plug-session period rides on both device
    # sensors; like the other live provenance keys it is kept out of the recorder.
    _unrecorded_attributes = frozenset(
        {"validation_session_start", "validation_session_end"}
    )

    def __init__(
        self,
        hub: Any,
        entry_id: str,
        device_uuid: str,
        device_name: str,
    ) -> None:
        """Initialize the device energy sensor."""
        super().__init__(hub.coordinator)
        self._hub = hub
        self._uuid = device_uuid
        self._attr_unique_id = f"{entry_id}_{device_uuid}_energy"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, device_uuid)},
            name=device_name,
            manufacturer="PowerGuesser",
            model=_device_model(hub, device_uuid),
        )

    async def async_added_to_hass(self) -> None:
        """Restore the last sensor value as an energy floor (kWh -> Wh)."""
        await super().async_added_to_hass()
        last = await self.async_get_last_sensor_data()
        if last is None or last.native_value is None:
            return
        try:
            restored_wh = float(last.native_value) * 1000.0
        except (TypeError, ValueError):
            return
        # A value recorded during this session (e.g. a deleted device's
        # state re-added on re-confirm) already contains the increments
        # delta_since_load counts; only a value from the previous session
        # needs the delta added (T-04-14 ordering).
        include_delta = True
        last_state = await self.async_get_last_state()
        if last_state is not None and last_state.last_updated is not None:
            include_delta = (
                last_state.last_updated.timestamp()
                < self._hub.devices.load_timestamp
            )
        self._hub.devices.raise_energy_floor(
            self._uuid, restored_wh, include_delta=include_delta
        )

    @property
    def available(self) -> bool:
        """Return True when this device's energy total is available.

        The energy sensor keeps its frozen total through every power-only
        reason (POWER_ONLY_REASONS, D-28/D-29/D-32/D-33) and through source
        gaps (D-29a), so it does not track the snapshot's source availability;
        only the hub coordinator itself failing (last_update_success) and
        record_missing (Plan 04-05) hide it.
        """
        return (
            self.coordinator.last_update_success
            and self._hub.devices.energy_available(self._uuid)
        )

    @property
    def native_value(self) -> float:
        """Return the stored energy total in kWh."""
        return round(self._hub.devices.energy_wh(self._uuid) / 1000.0, 6)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the static OUT-04 attribute set (D-17; live keys are power-only)."""
        return dict(self._hub.devices.attributes(self._uuid))


class DeviceValidationSensor(
    CoordinatorEntity[DataUpdateCoordinator[Snapshot]], SensorEntity
):
    """Diagnostic sensor with the device's holdout-backed energy error % (D-14, D-19).

    One per confirmed device, on the device's own DeviceInfo; the unique_id
    (``{entry_id}_{device_uuid}_validation``) and the % unit with
    state_class measurement anchor recorder history (one-way, D-14). The
    state is the holdout-backed error % and reads unknown whenever the
    device has no such figure; the VAL-03 signal is the boolean unvalidated
    attribute, which stays recorded (filterable) while the eight high-churn
    provenance keys are not. Plan 05-05 appends drift_evidence
    ("insufficient" | "sufficient") after train_cutoff: low-churn and
    user-relevant, so it stays recorded too.

    Provenance: when validated_via names a class, the state and nde, mae_w,
    baseline_nde and n_events are medians (n_events the total) over the
    class's plug peers — plug-metered peers that learned without their plugs
    and were replayed on the masked shadow pipeline — not measurements of
    this device, which has no plug of its own. The device's class label is
    derived from its active version (with the D-06 hysteresis) while peer
    classes come from the plug artefacts, so a device near a band edge can
    read unvalidated even though a passing peer of its real load exists;
    that mismatch is conservative (it never claims a validation it lacks)
    and accepted.
    """

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_translation_key = "device_validation"
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 1
    _unrecorded_attributes = frozenset(
        {
            "validated_via",
            "nde",
            "mae_w",
            "baseline_nde",
            "n_events",
            "holdout_start",
            "holdout_end",
            "train_cutoff",
        }
    )

    def __init__(
        self,
        hub: Any,
        entry_id: str,
        device_uuid: str,
        device_name: str,
    ) -> None:
        """Initialize the device's validation sensor."""
        super().__init__(hub.coordinator)
        self._hub = hub
        self._uuid = device_uuid
        self._attr_unique_id = f"{entry_id}_{device_uuid}_validation"
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, device_uuid)},
            name=device_name,
            manufacturer="PowerGuesser",
            model=_device_model(hub, device_uuid),
        )

    @property
    def available(self) -> bool:
        """Return True when the device's energy total is available.

        Availability follows the energy sensor (D-14): every power-only
        reason and record_missing hide it together with the device sensors,
        only the hub coordinator failing (last_update_success) hides a
        healthy device's figure.
        """
        return (
            self.coordinator.last_update_success
            and self._hub.devices.energy_available(self._uuid)
        )

    @property
    def native_value(self) -> float | None:
        """Return the holdout-backed error % (None -> unknown, D-19)."""
        state, _attrs = self._hub.devices.validation_details(self._uuid)
        return state

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the fixed thirteen-key validation attribute set (D-14)."""
        _state, attrs = self._hub.devices.validation_details(self._uuid)
        return attrs


class DetectionTimingSensor(_PowerGuesserSensor):
    """Diagnostic sensor exposing the per-tick detector cost and state (DET-04).

    The state is the p95 per-tick detector time in ms; before the first tick
    it reads unknown. The attributes carry the rest of timing_summary():
    p50/max, tick count, modelled device count, the detector status, the
    unattributed edge total and the map of unavailable devices to their
    reasons.
    """

    _attr_translation_key = "detection_timing"
    _attr_unique_id = "powerguesser_detection_timing"
    _attr_entity_category = EntityCategory.DIAGNOSTIC
    _attr_device_class = SensorDeviceClass.DURATION
    _attr_native_unit_of_measurement = UnitOfTime.MILLISECONDS
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 3
    _unrecorded_attributes = frozenset(
        {
            "p50_ms",
            "max_ms",
            "ticks",
            "devices",
            "detector",
            "unattributed_total",
            "unavailable_devices",
        }
    )

    def __init__(self, hub: Any) -> None:
        """Initialize the detection timing sensor."""
        super().__init__(hub.coordinator)
        self._hub = hub
        self._attr_device_info = _device_info()

    @property
    def native_value(self) -> float | None:
        """Return the p95 per-tick detector time in ms (None before the first tick)."""
        summary = self._hub.devices.timing_summary()
        if summary["ticks"] == 0:
            return None
        return float(summary["p95_ms"])

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the detector state attributes (all unrecorded)."""
        summary = self._hub.devices.timing_summary()
        return {
            "p50_ms": summary["p50_ms"],
            "max_ms": summary["max_ms"],
            "ticks": summary["ticks"],
            "devices": summary["devices"],
            "detector": summary["detector"],
            "unattributed_total": summary["unattributed_total"],
            "unavailable_devices": dict(summary["unavailable_devices"]),
        }


async def async_setup_entry(
    hass: HomeAssistant,
    entry: PowerGuesserConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up PowerGuesser sensors from a config entry."""
    hub = entry.runtime_data
    entities: list[SensorEntity] = [
        GrossPowerSensor(hub.coordinator),
        RemainderPowerSensor(hub.coordinator),
        EffectiveSampleIntervalSensor(hub.profile_coordinator),
        BootstrappedUsableHistorySensor(hub.bootstrap_coordinator),
        LearningStatusSensor(hub.learning.coordinator, entry),
        DetectionTimingSensor(hub),
        CaptureStatusSensor(hub, entry),
    ]
    known_sources: set[str] = set()
    for source in hub.profile_sources.values():
        entities.append(SampleIntervalSensor(hub.profile_coordinator, source))
        known_sources.add(source.entity_id)
    async_add_entities(entities)

    # Bind the platform callback so device confirms can add entities later,
    # then reconcile any device subentries restored from storage.
    hub.devices.bind_platform(async_add_entities)
    await hub.devices.async_reconcile()

    @callback
    def _on_profile_sources_updated() -> None:
        """Handle dynamic profile source changes."""
        current_sources = set(hub.profile_sources.keys())
        new_sources = current_sources - known_sources
        retired_sources = known_sources - current_sources

        if new_sources:
            new_entities = [
                SampleIntervalSensor(hub.profile_coordinator, hub.profile_sources[eid])
                for eid in new_sources
            ]
            known_sources.update(new_sources)
            async_add_entities(new_entities)

        if retired_sources:
            ent_reg = er.async_get(hass)
            for eid in retired_sources:
                known_sources.remove(eid)
                unique_id = f"powerguesser_sample_interval_{eid}"
                sensor_entity_id = ent_reg.async_get_entity_id(
                    "sensor", DOMAIN, unique_id
                )
                if sensor_entity_id:
                    ent_reg.async_remove(sensor_entity_id)

    entry.async_on_unload(
        async_dispatcher_connect(
            hass,
            f"{SIGNAL_PROFILE_SOURCES}_{entry.entry_id}",
            _on_profile_sources_updated,
        )
    )
