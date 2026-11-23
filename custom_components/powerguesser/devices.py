"""Device runtime for confirmed PowerGuesser devices.

Owns the two device Stores (cold ``powerguesser.devices`` v1 with the complete
device record schema, hot ``powerguesser.device_state`` v1 with energy totals and
detector state) and the reconcile/confirm lifecycle:

- Confirm is the ONLY way a device comes into existence (REV-02): it writes the
  cold record first (status ``confirmed``, ``subentry_id`` None — the exact crash
  window Plan 04-05's startup reconcile turns into ``unconfirmed``), then adds the
  config subentry whose ``unique_id`` is the device uuid (D-03, D-04).
- ``async_reconcile`` is an idempotent, lock-guarded diff of the entry's device
  subentries that adds power/energy entities for confirmed records. It only reads
  subentries and writes Stores; it never adds or removes subentries.
- Nothing here touches ``entry.options`` and nothing reloads anything.
"""

import asyncio
from collections.abc import Callable, Iterable, Mapping
import dataclasses
import json
import logging
import math
import time
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
import uuid

from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.data_entry_flow import AbortFlow
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_connect
from homeassistant.helpers.storage import Store
import homeassistant.util.dt as dt_util

from .const import (
    DEVICES_STORE_KEY,
    DEVICES_STORE_VERSION,
    DEVICE_STATE_SAVE_DELAY_S,
    DEVICE_STATE_STORE_KEY,
    DEVICE_TRANSITION_SAVE_DELAY_S,
    DOMAIN,
    GRID_GUARD_S,
    SIGNAL_PREFS_UPDATED,
    SUBENTRY_TYPE_DEVICE,
)
from .energy_prefs import async_get_prefs
from .engine.detect.online import (
    DeviceModel,
    OnlineDetector,
    ProgramModel,
    compile_model,
    state_keys,
)
from .engine.detect.stats import TickTimer
from .engine.detect.stream import GridResampler
from .engine.intervals import EVENT_TYPE_MIN_S, EventType, gate
from .engine.learning.artefact import SignatureArtefact, detection_differs
from .engine.learning.episodes import MIN_EDGE_W
from .engine.model import Problem, Severity, Snapshot
from .engine.validation.classes import derive_class
from .engine.validation.drift import REASON_VALIDATION_ERROR_RISING, LiveDrift
from .issues import IssueReconciler

if TYPE_CHECKING:
    from .hub import PowerGuesserHub

_LOGGER = logging.getLogger(__name__)

# Complete v1 device-record key set (costly contract, D-11). Plans 04-05..04-12
# fill values into these keys; Phase 5 (Plan 05-04) deliberately adds exactly
# ONE key, "validation", and later per-device validation state (drift, a
# rejected proposal) lives inside it, never as new record keys (Pitfall 8).
# Every record loaded from disk is normalised to exactly this set (missing
# keys get defaults).
DEVICE_RECORD_DEFAULTS: dict[str, Any] = {
    "name": "",
    "signature_id": "",
    "aliases": [],
    "kind": "switched",
    "preview": False,
    "status": "unconfirmed",
    "retired_into": None,
    "subentry_id": None,
    "versions": [],
    "active_v": 0,
    "pending_proposal": None,
    "created_at": None,
    "alias_artefacts": {},
    "pending_refit": False,
    "last_refit": None,
    "validation": {},
}

# Keys of one parameter version entry (v1).
VERSION_KEYS = ("v", "artefact", "trained_at", "scores", "accepted_at", "source")

# Availability reasons that make only the POWER sensor unavailable while the
# energy sensor stays available with its frozen total (D-28/D-29, D-32, D-33
# and D-12: a retired device is not modelled, so its energy sensor keeps its
# last value while its power sensor becomes unavailable). record_missing is
# the only energy-unavailable reason: a device subentry without a cold record
# has no verified total to show (backup mismatch, T-04-13).
POWER_ONLY_REASONS = frozenset(
    {"disabled_interval", "grid_mismatch", "model_error", "state_unknown", "retired"}
)

# A device's detection gate is its OWN kind's event type (RESEARCH A5 Gate,
# folded 60 s todo): a switched device needs STEP (enabled up to a 45 s
# interval), a program device CYCLE_SHAPE (enabled up to 90 s), so a program
# device keeps working at 60 s sampling while a switched one reads
# disabled_interval. Module level on purpose: it is a kind->type mapping, not
# a new consumer of the event types.
_KIND_GATE: dict[str, EventType] = {
    "switched": EventType.STEP,
    "program": EventType.CYCLE_SHAPE,
}


def _normalise_record(raw: Any) -> dict[str, Any] | None:
    """Return a record with exactly the v1 key set, or None for unusable input."""
    if not isinstance(raw, dict):
        return None
    rec: dict[str, Any] = {}
    for key, default in DEVICE_RECORD_DEFAULTS.items():
        value = raw.get(key, default)
        if isinstance(default, list):
            value = list(value) if isinstance(value, list) else []
        elif isinstance(default, dict):
            value = dict(value) if isinstance(value, dict) else {}
        elif isinstance(default, bool):
            value = bool(value)
        rec[key] = value
    versions = []
    for entry in rec["versions"]:
        if not isinstance(entry, dict):
            continue
        versions.append({k: entry.get(k) for k in VERSION_KEYS})
    rec["versions"] = versions
    return rec


def _artefact_states_differ(
    active: Mapping[str, Any] | None, new: Mapping[str, Any]
) -> bool:
    """Return True when the candidate differs from the active version's template.

    Delegates to the engine's ``detection_differs``: a switched pair compares
    its states, a program pair also compares the stage template (stages count
    for program devices, RESEARCH Pitfall 1), and no usable active version
    always differs.
    """
    return detection_differs(active, new)


class DeviceRuntime:
    """Runtime manager for confirmed devices: Stores, confirm, reconcile."""

    def __init__(
        self, hass: HomeAssistant, entry: Any, hub: PowerGuesserHub
    ) -> None:
        """Initialize the device runtime with its two Stores."""
        self.hass = hass
        self.entry = entry
        self.hub = hub
        self._store = Store[dict[str, Any]](
            hass, DEVICES_STORE_VERSION, DEVICES_STORE_KEY
        )
        self._state_store = Store[dict[str, Any]](
            hass, DEVICES_STORE_VERSION, DEVICE_STATE_STORE_KEY
        )
        self._devices: dict[str, dict[str, Any]] = {}
        self._energy_wh: dict[str, float] = {}
        self._detector_state: Any = None
        self._add_entities: Any = None
        self._added_subentries: set[str] = set()
        self._lock = asyncio.Lock()
        # Online detector state (Plan 04-04): built from the confirmed records
        # on the hub's effective sample interval (D-26) and rebuilt when the
        # confirmed set or the interval profile changes.
        self._detector: OnlineDetector | None = None
        self._resampler: GridResampler | None = None
        self._detector_grid_s: int = 30
        self._enabled_types: tuple[EventType, ...] = ()
        self._suppress_alignment: bool = True
        self._models: dict[str, DeviceModel | ProgramModel] = {}
        self._grid_mismatch: set[str] = set()
        self._model_errors: set[str] = set()
        self._device_w: dict[str, float] = {}
        self._last_estimated: float = 0.0
        self._last_on: frozenset[str] = frozenset()
        self._last_unknown: frozenset[str] = frozenset()
        self._last_push: tuple[float, float | None, tuple | None] | None = None
        self._timer = TickTimer()
        self._unattributed_total: int = 0
        self._last_step: Any = None
        self._detector_fingerprint: tuple[tuple[str, int, int], ...] = ()
        self._profile_unsub: Callable[[], None] | None = None
        # D-34: a dedicated reconciler for the detection_interval_too_slow
        # Repairs issue. IssueReconciler.apply is a desired-set diff over its
        # own ids, so sharing hub.reconciler (Energy-prefs problems) or the
        # device problem reconciler below would let their passes delete it.
        self._interval_issues = IssueReconciler(hass, entry)
        entry.async_on_unload(self._interval_issues.clear)
        # Device problem set (backup mismatches): device_record_missing today,
        # retired_device_in_energy in Plan 04-11 — one owned reconciler, kept
        # separate from self._interval_issues (D-34) and hub.reconciler.
        self._issues = IssueReconciler(hass, entry)
        entry.async_on_unload(self._issues.clear)
        # Desired problem sets of the two device problems, kept separately so
        # each refresh can re-apply the union through self._issues: the
        # record-missing set is computed in async_reconcile, the retired set
        # in async_check_retired_issues (D-12).
        self._device_problems: list[Problem] = []
        self._retired_problems: list[Problem] = []
        # Energy added per device since async_load, in memory only (T-04-14:
        # raise_energy_floor is max(current, restored + delta_since_load),
        # so the detector may step before the entity restores its state).
        self._delta_since_load: dict[str, float] = {}
        # Session start (async_load time): a restore value recorded after
        # this timestamp belongs to the current session (e.g. a deleted
        # device's state re-added minutes later), so its increments are
        # already inside both the restored value and delta_since_load — the
        # floor must not add the delta again. Values recorded before it are
        # from the previous session and need the delta.
        self._load_ts: float = 0.0
        # Pending load-time restore (D-26 with D-29a, restore grid check):
        # (stored detector state, D-30b unknown seed). async_load runs before
        # hub.async_start, so the effective grid is unknown there; the first
        # profile publish (always inside hub.async_start, before any device
        # entity exists) owns the detector build and the restore, never the
        # provisional 30 s default.
        self._pending_restore: tuple[Any, list[str]] | None = None
        # Plan 05-04: the latest ok report's class results keyed by class
        # name (D-08 passes, medians and event totals) and its holdout meta.
        # Anything else (no report, a refusal, a corrupt Store, T-05-12)
        # clears the results so nothing stale validates a class (T-05-11).
        # The FULL class entries stay: Plan 05-06 reads them for the
        # class-holdout drift (D-23), so no subset of fields is kept.
        self._class_results: dict[str, Mapping[str, Any]] = {}
        self._report_meta: dict[str, Any] = {
            "holdout_start": None,
            "holdout_end": None,
            "train_cutoff": None,
        }
        # Plan 05-05 (D-15 live proxy, RESEARCH Pattern 4): per-device live
        # drift trackers persisted in the hot Store's top-level "drift" key
        # (high churn, delay-saved) and each device's latest verdict, in
        # memory only, for the diagnostic sensor's drift_evidence attribute.
        self._drift: dict[str, LiveDrift] = {}
        self._drift_verdicts: dict[str, Any] = {}

    def _subentry_backed_confirmed(self) -> set[str]:
        """Uuids of confirmed records whose device subentry exists (D-30b).

        Records the startup reconcile turns unconfirmed (subentry gone) are
        not seeded: they get no entities and no unknown state.
        """
        sub_uuids = {
            str(sub.data.get("device_uuid"))
            for sub in self.entry.get_subentries_of_type(SUBENTRY_TYPE_DEVICE)
            if sub.data.get("device_uuid")
        }
        return {
            dev_uuid
            for dev_uuid, rec in self._devices.items()
            if rec.get("status") == "confirmed" and dev_uuid in sub_uuids
        }

    def _sync_from_detector(self) -> None:
        """Mirror the restored detector state into the publish seams.

        Without this an ON-restored device would read 0.0 until the first
        grid point drains; right after setup its sensors must show the
        restored step (D-26/D-29a restore grid check behaviour).
        """
        detector = self._detector
        if detector is None:
            return
        state = detector.state()
        devices_state = state.get("devices") or {}
        self._device_w = {
            key: float(value.get("on_w", 0.0) or 0.0)
            for key, value in devices_state.items()
            if isinstance(value, Mapping) and value.get("on")
        }
        self._last_estimated = float(sum(self._device_w.values()))
        self._last_on = frozenset(self._device_w)
        self._last_unknown = frozenset(detector.unknown)
        self._detector_state = state

    async def async_load(self) -> None:
        """Load both Stores defensively; unknown shapes load as empty."""
        data = await self._store.async_load()
        devices: dict[str, dict[str, Any]] = {}
        if isinstance(data, dict):
            raw = data.get("devices")
            if isinstance(raw, dict):
                for uuid, rec in raw.items():
                    norm = _normalise_record(rec)
                    if norm is not None:
                        devices[str(uuid)] = norm
        self._devices = devices

        state = await self._state_store.async_load()
        energy: dict[str, float] = {}
        detector = None
        drift: dict[str, LiveDrift] = {}
        if isinstance(state, dict):
            raw_energy = state.get("energy_wh")
            if isinstance(raw_energy, dict):
                for uuid, wh in raw_energy.items():
                    try:
                        energy[str(uuid)] = max(0.0, float(wh))
                    except (TypeError, ValueError):
                        continue
            detector = state.get("detector")
            # Plan 05-05: restore the live drift trackers; a bad entry loads
            # as a fresh tracker (T-05-17), never a crash. Plan 06-04: the
            # tracker's kind comes from the DEVICE RECORD, never from the
            # stored dict (which carries no kind), so a program device keeps
            # the D-26 window across a restart.
            raw_drift = state.get("drift")
            if isinstance(raw_drift, dict):
                for dev_uuid, raw in raw_drift.items():
                    rec = devices.get(str(dev_uuid)) or {}
                    kind = str(rec.get("kind") or "switched")
                    drift[str(dev_uuid)] = LiveDrift.from_dict(raw, kind=kind)
        self._energy_wh = energy
        self._detector_state = detector
        self._drift = drift
        self._delta_since_load = {key: 0.0 for key in energy}
        self._load_ts = dt_util.utcnow().timestamp()
        # Pending load-time restore (restore grid check): keep the stored
        # state and the D-30b seed until the first profile publish builds
        # the detector on its grid; until then apply_snapshot/on_tick only
        # remember the last pushed row (the build seeds the fresh edge
        # stream with it). `stored` is None when the hot Store is missing
        # or its detector field is null (crash before the first save, older
        # backup); the seed covers confirmed, subentry-backed uuids the
        # stored state does not cover (D-30b).
        stored = detector
        missing = sorted(self._subentry_backed_confirmed() - state_keys(stored))
        self._pending_restore = (stored, missing)
        if self.hub.profile_coordinator.data is not None:
            self._on_profile_publish()
        self._profile_unsub = self.hub.profile_coordinator.async_add_listener(
            self._on_profile_publish
        )
        self.entry.async_on_unload(self._profile_unsub)
        # D-12: re-evaluate the retired_device_in_energy issues whenever the
        # Energy dashboard preferences change (a user edit can add or remove
        # the retired entity from device_consumption).
        self.entry.async_on_unload(
            async_dispatcher_connect(
                self.hass, SIGNAL_PREFS_UPDATED, self._async_prefs_updated
            )
        )
        # Plan 05-04: derive every device's class from its active artefact on
        # load (D-06); records stored before the validation key existed gain
        # their class here (a change schedules a cold save).
        self._refresh_device_classes()

    async def _async_prefs_updated(self) -> None:
        """Re-check the retired-device issues on Energy prefs changes (D-12)."""
        await self.async_check_retired_issues()

    # -- online detector (Plan 04-04) --------------------------------------

    def _profile_settings(self) -> tuple[int, tuple[EventType, ...], bool]:
        """Return (grid_s, enabled event types, suppress_alignment).

        The detector grid is the hub's effective sample interval, or 30 s with
        the gate fallback while no profile snapshot exists — the same rule as
        the learning job builder (D-26).
        """
        data = self.hub.profile_coordinator.data
        if data is not None and getattr(data, "effective_s", None) is not None:
            grid_s = int(round(float(data.effective_s)))
            enabled = tuple(data.enabled)
        else:
            grid_s = 30
            enabled = gate(30.0, True)[0]
        suppress = EventType.SHORT_EVENT not in enabled
        return grid_s, enabled, suppress

    @staticmethod
    def _active_artefact(rec: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Return the artefact of the record's active version, if usable."""
        try:
            active_v = int(rec.get("active_v") or 0)
        except (TypeError, ValueError):
            return None
        for entry in rec.get("versions") or []:
            if isinstance(entry, Mapping) and entry.get("v") == active_v:
                artefact = entry.get("artefact")
                return artefact if isinstance(artefact, Mapping) else None
        return None

    def _compile_models(
        self, grid_s: int
    ) -> tuple[list[DeviceModel | ProgramModel], set[str], set[str], float]:
        """Compile DeviceModels for confirmed records on the detector grid.

        Returns (models, grid_mismatch uuids, model_error uuids, threshold_w):
        a device whose artefact ``extraction.grid_s`` differs is not modelled
        (D-26) and a corrupt artefact marks only that device (D-33). Every
        confirmed record is compiled through ``compile_model``, which builds a
        DeviceModel for kind switched and a ProgramModel (stage template) for
        kind program, so a program device is stepped at every drained grid
        point next to the switched ones.
        """
        models: list[DeviceModel | ProgramModel] = []
        mismatched: set[str] = set()
        errors: set[str] = set()
        thresholds: list[float] = []
        for dev_uuid, rec in sorted(self._devices.items()):
            if rec.get("status") != "confirmed":
                continue
            artefact = self._active_artefact(rec)
            if artefact is None:
                errors.add(dev_uuid)
                continue
            extraction = artefact.get("extraction")
            extraction = extraction if isinstance(extraction, Mapping) else {}
            try:
                art_grid = int(round(float(extraction.get("grid_s") or 0)))
            except (TypeError, ValueError):
                art_grid = 0
            if art_grid != grid_s:
                mismatched.add(dev_uuid)
                continue
            try:
                model = compile_model(dev_uuid, artefact)
            except (ValueError, TypeError, KeyError):
                errors.add(dev_uuid)
                continue
            models.append(model)
            try:
                thresholds.append(float(extraction.get("edge_threshold_w") or 0.0))
            except (TypeError, ValueError):
                pass
        threshold = max(MIN_EDGE_W, min(thresholds)) if thresholds else MIN_EDGE_W
        return models, mismatched, errors, threshold

    def _confirmed_fingerprint(self) -> tuple[tuple[str, int, int], ...]:
        """Hash the confirmed record set so reconcile can diff it cheaply."""
        out: list[tuple[str, int, int]] = []
        for dev_uuid, rec in sorted(self._devices.items()):
            if rec.get("status") != "confirmed":
                continue
            artefact = self._active_artefact(rec)
            blob = json.dumps(artefact, sort_keys=True, default=str)
            out.append((dev_uuid, int(rec.get("active_v") or 0), hash(blob)))
        return tuple(out)

    @callback
    def _build_detector(self, seed_unknown: Iterable[str] = ()) -> None:
        """(Re)build the resampler and detector on the current profile grid.

        ``seed_unknown`` carries flags across a rebuild: a rebuild is neither
        a matched edge nor evidence that an ON device went off (D-29a), so old
        unknown and old ON keys enter the fresh detector as unknown; parked
        keys keep the flag through unmodelled periods (e.g. grid_mismatch).
        """
        grid_s, enabled, suppress = self._profile_settings()
        models, mismatched, errors, threshold = self._compile_models(grid_s)
        old = self._detector
        detector = OnlineDetector(
            models,
            grid_s=grid_s,
            threshold_w=threshold,
            suppress_alignment=suppress,
            unknown=seed_unknown,
        )
        if old is not None:
            # Keep energy totals and grid continuity across the rebuild; every
            # stored ON period ends here without evidence (D-29a), so force the
            # stored on flags off and carry them as unknown instead.
            state = old.state()
            devices_state = {}
            for key, value in (state.get("devices") or {}).items():
                if isinstance(value, Mapping):
                    was_on = bool(value.get("on"))
                    devices_state[key] = {
                        **value,
                        "on": False,
                        "unknown": bool(value.get("unknown")) or was_on,
                    }
            state = {**state, "devices": devices_state}
            detector.restore(state, dt_util.utcnow().timestamp())
        self._detector = detector
        self._resampler = GridResampler(grid_s, guard_s=GRID_GUARD_S)
        # Seed the fresh edge stream with the current level (zero-order hold):
        # a fresh stream has no previous sample, so an edge that arrives before
        # the next tick would be its first value and never form a diff — a run
        # right after a rebuild would go undetected. Prefer the last pushed row;
        # with none (no tick since the previous rebuild), hold the last
        # published pre-estimate residual. The seed must stay strictly after
        # the restored last grid timestamp or the detector drops it.
        prev_row = self._last_push
        seed_value: float | None = None
        if prev_row is not None and prev_row[1] is not None:
            seed_value = float(prev_row[1])
        else:
            data = self.hub.coordinator.data
            if data is not None and data.signed_residual_w is not None:
                seed_value = float(data.signed_residual_w)
        if seed_value is not None:
            seed_ts = dt_util.utcnow().timestamp()
            last_grid_ts = detector.state().get("last_grid_ts")
            if last_grid_ts is not None and seed_ts <= float(last_grid_ts):
                seed_ts = float(last_grid_ts) + 1e-3
            detector.step(seed_ts, seed_value, None)
        self._detector_grid_s = grid_s
        self._enabled_types = enabled
        self._suppress_alignment = suppress
        self._models = {m.key: m for m in models}
        self._grid_mismatch = mismatched
        self._model_errors = errors
        self._device_w = {}
        self._last_estimated = 0.0
        self._last_on = frozenset()
        self._last_unknown = frozenset(detector.unknown)
        self._last_push = None
        self._last_step = None
        self._detector_fingerprint = self._confirmed_fingerprint()
        self._refresh_interval_issue()

    @callback
    def _maybe_rebuild_detector(self) -> None:
        """Rebuild the detector when the confirmed set changed.

        Same grid and event-type settings: swap models in place so existing
        ON state survives (OnlineDetector.set_models). Changed grid/settings:
        full rebuild; ON devices enter the unknown state (D-29a).
        """
        fingerprint = self._confirmed_fingerprint()
        if self._detector is None:
            if self._pending_restore is not None:
                # The first profile publish owns the load-time build; do not
                # construct a detector on the provisional grid beforehand.
                return
            self._build_detector()
            return
        if fingerprint == self._detector_fingerprint:
            return
        grid_s, enabled, suppress = self._profile_settings()
        if (
            grid_s == self._detector_grid_s
            and enabled == self._enabled_types
            and suppress == self._suppress_alignment
        ):
            models, mismatched, errors, _threshold = self._compile_models(grid_s)
            self._detector.set_models(models)
            self._models = {m.key: m for m in models}
            self._grid_mismatch = mismatched
            self._model_errors = errors
            self._detector_fingerprint = fingerprint
            self._last_unknown = frozenset(self._detector.unknown)
            self._refresh_interval_issue()
        else:
            old = self._detector
            self._build_detector(
                seed_unknown=sorted(set(old.unknown) | set(self._last_on))
            )

    @callback
    def _on_profile_publish(self) -> None:
        """Rebuild the detector when the effective interval profile changes (D-26).

        The first publish while a load-time restore is pending owns the
        detector build and the restore on its grid (restore grid check: the
        stored state is never compared with the provisional 30 s default
        that applies while profile_coordinator.data is None); restore()
        itself turns stored-ON devices unknown when the stored grid_s
        differs, even on a quick restart (D-26, D-29a), and arms the
        engine's one-shot D-31 post-gap check for devices restored OFF.

        A rebuild is neither a matched edge nor evidence that an ON device
        went off, so old unknown and old ON keys enter the fresh detector
        as unknown (D-29a); parked keys survive unmodelled periods.
        """
        if self._pending_restore is not None:
            stored, missing = self._pending_restore
            self._pending_restore = None
            self._build_detector(seed_unknown=missing)
            if stored is not None:
                self._detector.restore(
                    stored, dt_util.utcnow().timestamp()
                )
            self._sync_from_detector()
            self._refresh_interval_issue()
            return
        if self._detector is None:
            self._build_detector()
            return
        grid_s, enabled, suppress = self._profile_settings()
        if (
            grid_s == self._detector_grid_s
            and enabled == self._enabled_types
            and suppress == self._suppress_alignment
        ):
            return
        old = self._detector
        self._build_detector(
            seed_unknown=sorted(set(old.unknown) | set(self._last_on))
        )
        self._republish()
        self._refresh_interval_issue()

    def unavailable_reason(self, uuid: str) -> str | None:
        """Return why a confirmed device's power estimate is unavailable, if it is.

        Fixed check order (extension contract, Plan 04-11): record_missing
        (Plan 04-05), then "retired" for a retired record (D-12), then
        disabled_interval, grid_mismatch, model_error, state_unknown (Plan
        04-04, D-26/D-28/D-29/D-32/D-33). "retired" comes before every
        detector reason: a retired device is not modelled (Pitfall 9), so it
        never counts toward Plan 04-04's D-34 detection_interval_too_slow
        issue, a retired device whose artefact was learned on another grid
        reads retired (not grid_mismatch), and a device retired while ON
        (whose dropped model parks an unknown flag in the detector, D-29a)
        reads retired (not state_unknown). While a device's OWN kind gate is
        disabled a retired device still reads retired, only a confirmed device
        reads disabled_interval: a switched record is gated on EventType.STEP
        and a program record on EventType.CYCLE_SHAPE (RESEARCH A5 Gate), so a
        program device stays available at 60 s sampling. Every reason except
        record_missing is power-only — the energy sensor stays available with
        its frozen total (POWER_ONLY_REASONS).
        """
        rec = self._devices.get(uuid)
        if rec is None:
            # A device subentry without a cold record (backup mismatch): both
            # sensors stay unavailable instead of crashing or guessing (T-04-13).
            return "record_missing"
        # A retired device is not modelled and never reads a detector reason.
        if rec.get("status") == "retired":
            return "retired"
        if rec.get("status") != "confirmed":
            return None
        if _KIND_GATE.get(str(rec.get("kind")), EventType.STEP) not in self._enabled_types:
            return "disabled_interval"
        if uuid in self._grid_mismatch:
            return "grid_mismatch"
        if uuid in self._model_errors:
            return "model_error"
        if self._detector is not None and uuid in self._detector.unknown:
            return "state_unknown"
        return None

    def power_available(self, uuid: str) -> bool:
        """Return True when the device's power estimate is available."""
        return self.unavailable_reason(uuid) is None

    def energy_available(self, uuid: str) -> bool:
        """Return True when the device's energy sensor is available.

        Every reason in POWER_ONLY_REASONS is power-only (D-28/D-29, D-32,
        D-33): the energy sensor keeps its frozen total, so no power-side
        reason puts a gap in the Energy dashboard. record_missing (Plan 04-05)
        is the only reason that keeps the energy sensor unavailable.
        """
        reason = self.unavailable_reason(uuid)
        return reason is None or reason in POWER_ONLY_REASONS

    @callback
    def _republish(self) -> None:
        """Republish the last snapshot so sensor states reflect runtime changes.

        Detector rebuilds (profile publish, confirmed-set change) alter
        unavailable_reason and the estimated set without a new aligner publish;
        device sensors only rewrite their state on coordinator updates, so the
        adjusted copy of the last snapshot is pushed explicitly (D-26, D-29a,
        D-32, D-33).
        """
        data = self.hub.coordinator.data
        if data is None:
            return
        signed = data.signed_residual_w
        if signed is None:
            self.hub.coordinator.async_set_updated_data(
                dataclasses.replace(
                    data,
                    estimated_w=self._last_estimated,
                    device_w=dict(self._device_w),
                )
            )
            return
        self.hub.coordinator.async_set_updated_data(
            dataclasses.replace(
                data,
                estimated_w=self._last_estimated,
                device_w=dict(self._device_w),
                remainder_w=max(0.0, signed - self._last_estimated),
            )
        )

    @callback
    def _hot_store_data(self) -> dict[str, Any]:
        """Serialize the hot runtime state for the delay-saved Store."""
        return {
            "energy_wh": self._energy_wh,
            "detector": self._detector_state,
            "drift": {uuid: ld.to_dict() for uuid, ld in self._drift.items()},
        }

    @callback
    def _drain_step(
        self, now_ts: float, value: float | None, phases: tuple | None
    ) -> tuple[Any, bool, bool, bool] | None:
        """Push one row, drain due grid points and step the detector.

        Returns (last StepResult, device_w_changed, unknown_changed,
        energy_delta_seen), or None when no grid point was due. No I/O, no
        awaits and no numpy allocation in this @callback path (T-04-10).
        """
        detector = self._detector
        resampler = self._resampler
        # Remember the last pushed row even while a load-time restore is
        # pending: the pending build seeds the fresh edge stream with it, so
        # the first grid point after the build has a value (D-26 note).
        self._last_push = (float(now_ts), value, phases)
        if detector is None or resampler is None:
            return None
        resampler.push(now_ts, value, phases)
        points = resampler.drain(now_ts)
        if not points:
            return None
        prev_on = self._last_on
        last: Any = None
        device_w_changed = False
        energy_seen = False
        for point in points:
            start = time.perf_counter_ns()
            result = detector.step(point.grid_ts, point.value, point.phases)
            self._timer.record(time.perf_counter_ns() - start)
            last = result
            if result.device_w != self._device_w:
                device_w_changed = True
            for key, delta in result.energy_wh_delta.items():
                if delta > 0.0:
                    total = self._energy_wh.get(key, 0.0)
                    self._energy_wh[key] = total + delta
                    self._delta_since_load[key] = (
                        self._delta_since_load.get(key, 0.0) + delta
                    )
                    energy_seen = True
        if last is None:
            return None
        self._device_w = dict(last.device_w)
        self._last_step = last
        self._last_estimated = float(last.estimated_w)
        self._unattributed_total = int(last.unattributed_total)
        # Plan 05-05 (D-15 live proxy; RESEARCH Pattern 4 and Pitfall 7):
        # drain BOTH detector logs every batch so neither grows without bound
        # in live use (drain_episodes is replay-only, drained here and
        # discarded). Each confirmed device's closed episodes feed its
        # LiveDrift tracker; evaluation runs at episode close, pure stdlib, no
        # I/O. A changed verdict (drift or reason flipped) writes the D-16
        # flags into the record's validation dict; drift never touches
        # versions, active_v, uuid, name or entity ids.
        drift_flags_changed = False
        closed = detector.drain_closed()
        detector.drain_episodes()
        for ep in closed:
            rec = self._devices.get(ep.key)
            if rec is None or rec.get("status") != "confirmed":
                continue
            tracker = self._drift.get(ep.key)
            if tracker is None:
                tracker = LiveDrift(kind=str(rec.get("kind") or "switched"))
                self._drift[ep.key] = tracker
            tracker.record(ep.end_ts, ep.match_score, ep.unattributed)
            verdict = tracker.evaluate()
            self._drift_verdicts[ep.key] = verdict
            if verdict.changed:
                validation = rec.get("validation")
                if not isinstance(validation, dict):
                    validation = {}
                    rec["validation"] = validation
                validation["drift"] = {
                    "drift": bool(verdict.drift),
                    "reason": verdict.reason if verdict.drift else None,
                    "since": (
                        dt_util.utc_from_timestamp(verdict.since_ts).isoformat()
                        if verdict.drift and verdict.since_ts is not None
                        else None
                    ),
                }
                # Plan 06-09 (REV-04, D-14/D-38): a live drift verdict revokes a
                # plug_session validation for good — it stays revoked when drift
                # later clears. The session period stays visible (attributes()).
                if bool(verdict.drift):
                    session = validation.get("session")
                    if (
                        isinstance(session, Mapping)
                        and session.get("revoked") is None
                    ):
                        validation["session"] = {
                            **session,
                            "revoked": {
                                "reason": "drift",
                                "at": dt_util.utcnow().isoformat(),
                            },
                        }
                drift_flags_changed = True
        on = frozenset(last.on)
        unknown = frozenset(detector.unknown)
        unknown_changed = unknown != self._last_unknown
        self._last_on = on
        self._last_unknown = unknown
        delay = (
            DEVICE_TRANSITION_SAVE_DELAY_S
            if on != prev_on or closed or drift_flags_changed
            else DEVICE_STATE_SAVE_DELAY_S
        )
        if drift_flags_changed:
            # The D-16 flags live in the cold record (validation.drift); a hot
            # save is already scheduled above with the transition delay.
            self._store.async_delay_save(
                lambda: {"devices": self._devices}, DEVICE_TRANSITION_SAVE_DELAY_S
            )
            self._republish()
        self._detector_state = detector.state()
        self._state_store.async_delay_save(self._hot_store_data, delay)
        return last, device_w_changed, unknown_changed, energy_seen

    @callback
    def apply_snapshot(
        self,
        snapshot: Snapshot,
        ts: float,
        phases: tuple[float, float, float] | None,
    ) -> Snapshot:
        """Feed one aligner publish through the detector and adjust the snapshot.

        The detector is fed the pre-estimate signed residual, never the
        remainder (RESEARCH Pattern 2); a missing residual pushes NaN so the
        gap rule applies (D-29a) and the snapshot is returned without any
        estimate adjustment.
        """
        if self._detector is None:
            return snapshot
        self._drain_step(ts, snapshot.signed_residual_w, phases)
        signed = snapshot.signed_residual_w
        if signed is None:
            return snapshot
        return dataclasses.replace(
            snapshot,
            estimated_w=self._last_estimated,
            device_w=dict(self._device_w),
            remainder_w=max(0.0, signed - self._last_estimated),
        )

    @callback
    def on_tick(self, now_ts: float, last: Snapshot | None) -> Snapshot | None:
        """Drain due grid points on the 10 s tick and republish on changes.

        Returns an adjusted copy of ``last`` when any device_w changed, the
        detector's unknown set changed, or any energy_wh_delta was > 0 on the
        drained points — a steady ON period still advances the energy sensor
        at grid cadence, not only on aligner publishes.
        """
        if self._detector is None or last is None:
            return None
        row = self._last_push
        if row is None:
            return None
        stepped = self._drain_step(now_ts, row[1], row[2])
        if stepped is None:
            return None
        _last, device_w_changed, unknown_changed, energy_seen = stepped
        if not (device_w_changed or unknown_changed or energy_seen):
            return None
        signed = last.signed_residual_w
        if signed is None:
            return dataclasses.replace(
                last,
                estimated_w=self._last_estimated,
                device_w=dict(self._device_w),
            )
        return dataclasses.replace(
            last,
            estimated_w=self._last_estimated,
            device_w=dict(self._device_w),
            remainder_w=max(0.0, signed - self._last_estimated),
        )

    def timing_summary(self) -> dict[str, Any]:
        """Return the per-tick timing ring and detector state (DET-04).

        unavailable_devices maps each confirmed device's name to the reason
        its power estimate is unavailable (every confirmed record with a
        reason).
        """
        unavailable: dict[str, str] = {}
        for uuid, rec in self._devices.items():
            if rec.get("status") != "confirmed":
                continue
            reason = self.unavailable_reason(uuid)
            if reason is None:
                continue
            unavailable[str(rec.get("name") or uuid)] = reason
        summary = self._timer.summary()
        summary["devices"] = len(self._models)
        summary["detector"] = self.detector_status
        summary["unattributed_total"] = self._unattributed_total
        summary["unavailable_devices"] = unavailable
        return summary

    @callback
    def _refresh_interval_issue(self) -> None:
        """D-34: raise/clear detection_interval_too_slow (Repairs surface).

        The issue exists exactly while at least one confirmed device reads
        disabled_interval (record_missing and retired come first in the
        reason order, so they never count). Each device is gated by its OWN
        kind's event type, so a program device at 60 s is not affected while a
        switched one is (RESEARCH A5 Gate). The interval limit is derived from
        engine.intervals.EVENT_TYPE_MIN_S the way gate() formats it, never
        written as a number; with affected kinds of different gates the
        SMALLEST gate is named — the interval at which every affected device
        works again.
        """
        affected = sorted(
            (str(rec.get("name") or uuid), str(rec.get("kind") or "switched"))
            for uuid, rec in self._devices.items()
            if rec.get("status") == "confirmed"
            and self.unavailable_reason(uuid) == "disabled_interval"
        )
        if not affected:
            self._interval_issues.apply([])
            return
        data = self.hub.profile_coordinator.data
        effective_s = float(data.effective_s) if data is not None else 30.0
        smallest_gate_s = min(
            EVENT_TYPE_MIN_S[
                _KIND_GATE.get(kind, EventType.STEP)
            ]
            for _name, kind in affected
        )
        self._interval_issues.apply(
            [
                Problem(
                    issue_id="detection_interval_too_slow",
                    kind="detection_interval_too_slow",
                    severity=Severity.WARNING,
                    translation_key="detection_interval_too_slow",
                    placeholders={
                        "interval": str(round(effective_s)),
                        "max_interval": f"{smallest_gate_s / 2:g}",
                        "devices": ", ".join(name for name, _kind in affected),
                    },
                )
            ]
        )

    @property
    def detector_status(self) -> str:
        """Return "running" | "no_devices" | "disabled_interval" (DET-04).

        disabled_interval is reported only while no confirmed device's OWN
        kind gate is enabled (RESEARCH A5 Gate): one program device keeps the
        detector "running" at 60 s sampling even though a switched device in
        the same hub reads disabled_interval.
        """
        confirmed = [
            rec for rec in self._devices.values() if rec.get("status") == "confirmed"
        ]
        if not confirmed:
            return "no_devices"
        if not any(
            _KIND_GATE.get(str(rec.get("kind")), EventType.STEP)
            in self._enabled_types
            for rec in confirmed
        ):
            return "disabled_interval"
        return "running"

    def bind_platform(self, add_entities: Any) -> None:
        """Store the platform add-entities callback for reconcile()."""
        self._add_entities = add_entities

    def record(self, uuid: str) -> dict[str, Any] | None:
        """Return the in-memory device record for a uuid, if any."""
        return self._devices.get(uuid)

    def device_for_signature(self, signature_id: str) -> str | None:
        """Return the uuid of the confirmed device owning a signature, if any."""
        for uuid, rec in self._devices.items():
            if rec.get("status") == "confirmed" and rec.get(
                "signature_id"
            ) == signature_id:
                return uuid
        return None

    def owned_signature_ids(self) -> set[str]:
        """Return signature ids and aliases of confirmed or retired devices (C1)."""
        owned: set[str] = set()
        for rec in self._devices.values():
            if rec.get("status") not in ("confirmed", "retired"):
                continue
            if rec.get("signature_id"):
                owned.add(str(rec["signature_id"]))
            for alias in rec.get("aliases") or []:
                owned.add(str(alias))
        return owned

    def identity_priors(self) -> list[dict[str, Any]]:
        """Return prior artefact dicts for confirmed or retired devices (D-11, OUT-05).

        For every record with status confirmed or retired (sorted by
        signature_id), the active version's artefact dict with signature_id
        set to the record's signature id, followed by each entry of the
        record's alias_artefacts (empty until Plan 04-12 merges a candidate
        in) with signature_id set to its alias id. The compile job appends
        these after the stored candidates and match_identities keeps the
        last prior per id, so a device's frozen version replaces any stored
        library artefact with the same signature id: identity is judged
        against the confirmed parameters, not against an unaccepted later
        run. Records without a usable active version contribute nothing.
        """
        priors: list[dict[str, Any]] = []
        records = sorted(
            self._devices.values(),
            key=lambda r: str(r.get("signature_id") or ""),
        )
        for rec in records:
            if rec.get("status") not in ("confirmed", "retired"):
                continue
            sig = str(rec.get("signature_id") or "")
            artefact = self._active_artefact(rec)
            if sig and artefact is not None:
                d = dict(artefact)
                d["signature_id"] = sig
                priors.append(d)
            alias_artefacts = rec.get("alias_artefacts") or {}
            if isinstance(alias_artefacts, Mapping):
                for alias_id in sorted(alias_artefacts):
                    alias_art = alias_artefacts[alias_id]
                    if not isinstance(alias_art, Mapping):
                        continue
                    d = dict(alias_art)
                    d["signature_id"] = str(alias_id)
                    priors.append(d)
        return priors

    def owner_name(self, sig_id: str) -> str | None:
        """Return the name of the device owning a signature id, if any (D-27).

        The confirmed record whose signature_id or aliases contain sig_id
        wins; otherwise the name of any record (e.g. unconfirmed) with that
        signature id or alias; else None. After Plan 04-11 a retired
        device's signature is an alias of the device it was merged into, so
        the note names the surviving device.
        """
        fallback: str | None = None
        for rec in self._devices.values():
            ids = {str(rec.get("signature_id") or "")}
            ids |= {str(a) for a in rec.get("aliases") or []}
            if sig_id not in ids:
                continue
            name = str(rec.get("name") or "") or None
            if rec.get("status") == "confirmed":
                return name
            if fallback is None:
                fallback = name
        return fallback

    # -- update proposals and versioned activation (D-11, D-25) ------------

    def active_version_differs(self, uuid: str, artefact: Mapping[str, Any]) -> bool:
        """True when the artefact's states differ from the device's active version."""
        rec = self._devices.get(uuid)
        if rec is None:
            return False
        return _artefact_states_differ(self._active_artefact(rec), artefact)

    def gate_actives(self) -> dict[str, dict[str, Any]]:
        """Return the frozen active artefact of every confirmed device (D-17).

        Keyed by signature id; compile jobs carry the map as "actives" so the
        worker can gate a re-learn proposal against the frozen version. The
        returned dicts are plain artefact dicts (copies), never references
        into the records.
        """
        out: dict[str, dict[str, Any]] = {}
        for rec in self._devices.values():
            if rec.get("status") != "confirmed":
                continue
            sig = str(rec.get("signature_id") or "")
            artefact = self._active_artefact(rec)
            if sig and artefact is not None:
                out[sig] = dict(artefact)
        return out

    def active_artefact_and_version(
        self, uuid: str
    ) -> tuple[Mapping[str, Any] | None, int]:
        """Return (active artefact, active_v) of a device, or (None, 0) (Plan 06-09).

        The validate_sessions job carries the frozen active artefact of a
        device-target session's device together with its version, so a session
        validation is revoked the moment that version is no longer active
        (D-38). A missing device or an unusable version yields (None, 0).
        """
        rec = self._devices.get(uuid)
        if rec is None:
            return None, 0
        try:
            active_v = int(rec.get("active_v") or 0)
        except (TypeError, ValueError):
            active_v = 0
        return self._active_artefact(rec), active_v

    async def async_record_rejected_proposal(
        self, uuid: str, info: Mapping[str, Any]
    ) -> None:
        """Record a gate-rejected (or ungated) re-learn proposal for diagnosis (D-17).

        Diagnostics only: the record's pending_proposal, versions and active_v
        are never touched — a proposal that replays worse than the frozen
        active version never reaches Review, and the user decides every
        parameter change explicitly through Accept.
        """
        rec = self._devices.get(uuid)
        if rec is None or rec.get("status") != "confirmed":
            return
        validation = rec.get("validation")
        if not isinstance(validation, dict):
            validation = {}
            rec["validation"] = validation
        validation["last_rejected_proposal"] = dict(info)
        await self._async_save_cold()

    def update_available(self, uuid: str) -> bool:
        """True while a learning-run proposal is pending for the device (D-11)."""
        rec = self._devices.get(uuid)
        return bool(
            rec is not None
            and rec.get("status") == "confirmed"
            and rec.get("pending_proposal")
        )

    def confirmed_entries(self) -> list[dict[str, Any]]:
        """Review entries for confirmed devices (Review contract C2).

        Each entry carries option_value ``device_<uuid>`` (never the signature
        id, which C1 reserves to candidates), the record's name as label and
        first_label, the active version's scores, the update_available flag
        and the stored versions count. Sorted by name then signature_id.
        """
        out: list[dict[str, Any]] = []
        for dev_uuid, rec in sorted(self._devices.items()):
            if rec.get("status") != "confirmed":
                continue
            name = str(rec.get("name") or rec.get("signature_id") or dev_uuid)
            scores: dict[str, Any] = {}
            for version in rec.get("versions") or []:
                if isinstance(version, Mapping) and version.get("v") == rec.get(
                    "active_v"
                ):
                    raw = version.get("scores")
                    scores = dict(raw) if isinstance(raw, Mapping) else {}
            out.append(
                {
                    "signature_id": str(rec.get("signature_id") or ""),
                    "device_uuid": dev_uuid,
                    "option_value": f"device_{dev_uuid}",
                    "kind": rec.get("kind") or "switched",
                    "label": name,
                    "first_label": name,
                    "scores": scores,
                    "confirmed": True,
                    "update_available": self.update_available(dev_uuid),
                    "versions_count": len(rec.get("versions") or []),
                    "drift": self._drift_view(dev_uuid)["drift"],
                }
            )
        out.sort(key=lambda e: (str(e["label"]), str(e["signature_id"])))
        return out

    async def async_set_proposal(
        self,
        uuid: str,
        artefact: Mapping[str, Any],
        *,
        run_id: str,
        proposed_at: str,
        trained_at: str | None,
    ) -> None:
        """Store a learning-run proposal as the device's pending proposal (D-11).

        The proposal replaces an older pending one and never touches the
        parameter versions: the stored artefact drops the identity, fingerprint
        and first_label decorations (the version entry holds the plain
        artefact, exactly like a confirm). A coordinator refresh is pushed so
        the power sensor's update_available attribute shows without a new
        aligner publish.
        """
        rec = self._devices.get(uuid)
        if rec is None or rec.get("status") != "confirmed":
            return
        art = {
            k: v
            for k, v in artefact.items()
            if k not in ("identity", "fingerprint", "first_label")
        }
        rec["pending_proposal"] = {
            "artefact": art,
            "run_id": run_id,
            "proposed_at": proposed_at,
            "trained_at": trained_at,
            "scores": dict(artefact.get("scores") or {}),
        }
        await self._async_save_cold()
        self._republish()

    async def async_activate_version(self, uuid: str, v: int) -> None:
        """Activate a stored parameter version: the single path for accept and rollback (D-25).

        ``v`` must be a stored version (ValueError otherwise). Activation sets
        active_v and preview (not the activated version's scores.validated),
        saves the cold Store, resets the device's live drift evidence (D-22:
        the old baseline describes other parameters), updates the
        review-record fingerprint to the activated artefact's fingerprint and
        rebuilds the detector models — same grid: OnlineDetector.set_models
        keeps the ON state; a version learned on another grid becomes
        grid_mismatch per D-26. It never appends or removes a version, never
        touches pending_proposal and never touches energy totals, so the
        uuid, unique_ids, entity_ids and both energy totals stay unchanged.
        """
        rec = self._devices.get(uuid)
        if rec is None:
            raise ValueError(f"Unknown device: {uuid}")
        try:
            v_int = int(v)
        except (TypeError, ValueError):
            raise ValueError(f"Unknown version {v!r} for device {uuid}") from None
        version: Mapping[str, Any] | None = None
        for entry in rec.get("versions") or []:
            if isinstance(entry, Mapping):
                try:
                    if int(entry.get("v")) == v_int:
                        version = entry
                        break
                except (TypeError, ValueError):
                    continue
        if version is None:
            raise ValueError(f"Unknown version {v_int} for device {uuid}")
        rec["active_v"] = v_int
        scores = version.get("scores") or {}
        rec["preview"] = not bool(scores.get("validated"))
        # D-22: the old baseline describes other parameters, so the live
        # drift evidence, the baseline and the hysteresis all start over and
        # the D-16 flags clear; another device's tracker is untouched. Plan
        # 06-04: the fresh tracker keeps the record's kind, so a program
        # device restarts on the D-26 window, not the switched one. The fresh
        # tracker is delay-saved into the hot Store so a restart cannot
        # resurrect the pre-reset evidence.
        self._drift[uuid] = LiveDrift(kind=str(rec.get("kind") or "switched"))
        self._drift_verdicts.pop(uuid, None)
        validation = rec.get("validation")
        if isinstance(validation, dict):
            validation.pop("drift", None)
            # Plan 06-09 (REV-04, D-38): accepting or rolling back a version
            # ends a plug_session validation for good — the validated artefact
            # version is no longer active, so the session basis is revoked
            # (it must not resurface if the user later rolls back to it).
            session = validation.get("session")
            if isinstance(session, Mapping) and session.get("revoked") is None:
                validation["session"] = {
                    **session,
                    "revoked": {
                        "reason": "version_changed",
                        "at": dt_util.utcnow().isoformat(),
                    },
                }
        self._state_store.async_delay_save(
            self._hot_store_data, DEVICE_STATE_SAVE_DELAY_S
        )
        await self._async_save_cold()
        sig = str(rec.get("signature_id") or "")
        artefact = version.get("artefact")
        if sig and isinstance(artefact, Mapping):
            try:
                fingerprint = SignatureArtefact.from_dict(artefact).fingerprint()
            except (KeyError, TypeError, ValueError):
                fingerprint = None
            if fingerprint is not None:
                await self.hub.learning.review.async_update_fingerprint(
                    sig, fingerprint
                )
        self._refresh_device_classes()
        self._maybe_rebuild_detector()
        self._republish()

    async def async_accept_proposal(self, uuid: str) -> int:
        """Accept the pending proposal as parameter version n+1 and activate it (D-11).

        The proposal's artefact is appended with source "proposal" under the
        same device uuid, every earlier version is kept, the proposal is
        cleared and the new version is activated through async_activate_version
        (the same path rollback uses, D-25). Raises ValueError without a
        pending proposal.
        """
        rec = self._devices.get(uuid)
        proposal = rec.get("pending_proposal") if rec is not None else None
        if not isinstance(proposal, Mapping):
            raise ValueError(f"No pending proposal for device {uuid}")
        next_v = (
            max(
                (
                    int(entry.get("v") or 0)
                    for entry in rec.get("versions") or []
                    if isinstance(entry, Mapping)
                ),
                default=0,
            )
            + 1
        )
        rec["versions"].append(
            {
                "v": next_v,
                "artefact": proposal["artefact"],
                "trained_at": proposal.get("trained_at"),
                "scores": dict(proposal.get("scores") or {}),
                "accepted_at": dt_util.utcnow().isoformat(),
                "source": "proposal",
            }
        )
        rec["pending_proposal"] = None
        await self._async_save_cold()
        await self.async_activate_version(uuid, next_v)
        return next_v

    async def async_retire(self, source_uuid: str, *, into: str) -> None:
        """Retire a confirmed device into another confirmed device (D-12, D-23).

        The target keeps everything: name, uuid, parameter versions and both
        energy totals are untouched — the absorbed total is NEVER added into
        the survivor's counter (no counter jumps ever, OUT-02). The source is
        retired in place: status "retired", ``retired_into`` the target uuid,
        any pending proposal cleared; its subentry and entities persist so
        the user's history and Energy-dashboard entries stay intact until the
        user deletes it under Settings. Its signature id and aliases become
        aliases of the target (deduplicated, order kept): aliases route
        identity only, never detection (Pitfall 9) — the detector rebuild
        below drops the retired device's model. Raises ValueError unless both
        records exist, are distinct and confirmed (T-04-22).
        """
        source = self._devices.get(source_uuid)
        if source is None or source.get("status") != "confirmed":
            raise ValueError(f"Cannot retire unknown device {source_uuid}")
        target = self._devices.get(into)
        if (
            into == source_uuid
            or target is None
            or target.get("status") != "confirmed"
        ):
            raise ValueError(f"Cannot retire {source_uuid} into {into}")
        target_aliases = target.setdefault("aliases", [])
        absorbed = [source.get("signature_id"), *(source.get("aliases") or [])]
        for alias in absorbed:
            if alias and alias not in target_aliases:
                target_aliases.append(str(alias))
        source["status"] = "retired"
        source["retired_into"] = into
        source["pending_proposal"] = None
        await self._async_save_cold()
        # The confirmed-set fingerprint changed: the detector rebuild drops
        # the retired device's model (it is no longer modelled, Pitfall 9)
        # and refreshes the D-34 interval issue, which never counts retired
        # devices ("retired" precedes "disabled_interval" in the reason order).
        self._maybe_rebuild_detector()
        self._republish()
        # The merge may have created a retired-device-in-Energy condition.
        await self.async_check_retired_issues()

    # -- union retrain aliases and versions (D-13) ---------------------------

    def refit_pending_uuids(self) -> list[str]:
        """Uuids of confirmed records with a retrain queued (uuid order, D-13)."""
        return sorted(
            uuid
            for uuid, rec in self._devices.items()
            if rec.get("status") == "confirmed" and rec.get("pending_refit")
        )

    async def async_mark_refit_pending(self, uuid: str) -> None:
        """Queue a union retrain for a device while a background job occupies the slot.

        The flag lives in the cold Store, so a restart resumes the retrain
        through LearningSupervisor._async_maybe_refit.
        """
        rec = self._devices.get(uuid)
        if rec is None or rec.get("status") != "confirmed":
            return
        rec["pending_refit"] = True
        await self._async_save_cold()

    async def async_add_aliases(
        self,
        uuid: str,
        items: list[tuple[str, Mapping[str, Any]]],
    ) -> None:
        """Record merged-candidate signatures as aliases with their artefacts (D-13).

        Each alias id joins the record's aliases (deduplicated, order kept) and
        its artefact lands in the record's existing alias_artefacts dict (the
        candidates Store is replaced by every learning run, so the artefact
        survives only here). Aliases route identity only — the device's own
        signature stays its only detection identity — and this method never
        touches versions, active_v or energy totals.
        """
        rec = self._devices.get(uuid)
        if rec is None:
            raise ValueError(f"Unknown device: {uuid}")
        aliases = rec.setdefault("aliases", [])
        alias_artefacts = rec.setdefault("alias_artefacts", {})
        for sig_id, artefact in items:
            if sig_id not in aliases:
                aliases.append(str(sig_id))
            alias_artefacts[str(sig_id)] = dict(artefact)
        await self._async_save_cold()

    async def async_apply_refit(
        self,
        uuid: str,
        artefact: Mapping[str, Any],
        *,
        trained_at: str | None,
    ) -> int:
        """Append an accepted union retrain as version n+1 and activate it (D-13).

        The version carries source "union"; activation goes through
        async_activate_version (the D-25 path shared with accept and rollback:
        active_v, preview, review fingerprint, detector rebuild). The uuid,
        unique_ids, entity_ids and energy totals never change.
        """
        rec = self._devices.get(uuid)
        if rec is None:
            raise ValueError(f"Unknown device: {uuid}")
        next_v = (
            max(
                (
                    int(entry.get("v") or 0)
                    for entry in rec.get("versions") or []
                    if isinstance(entry, Mapping)
                ),
                default=0,
            )
            + 1
        )
        rec["versions"].append(
            {
                "v": next_v,
                "artefact": dict(artefact),
                "trained_at": trained_at,
                "scores": dict(artefact.get("scores") or {}),
                "accepted_at": dt_util.utcnow().isoformat(),
                "source": "union",
            }
        )
        await self._async_save_cold()
        await self.async_activate_version(uuid, next_v)
        return next_v

    async def async_record_refit(self, uuid: str, result: Mapping[str, Any]) -> None:
        """Persist the outcome of a retrain attempt and clear the pending flag.

        ``result`` is either {"at", "accepted", "current_S", "new_S"} for a
        finished refit (kept or rejected) or {"at", "error"} for a failed one;
        both shapes keep the device's current version and identity untouched.
        """
        rec = self._devices.get(uuid)
        if rec is None:
            return
        rec["last_refit"] = dict(result)
        rec["pending_refit"] = False
        await self._async_save_cold()

    async def async_check_retired_issues(self) -> None:
        """D-12: raise/clear the retired_device_in_energy Repairs issues.

        A retired device whose energy entity is still listed in the Energy
        dashboard's ``device_consumption`` gets a WARNING issue telling the
        user to swap in the survivor's sensor (its frozen total stays in the
        history either way). The issue exists exactly while the condition
        holds: it is re-evaluated at the end of async_reconcile and on every
        Energy prefs change (SIGNAL_PREFS_UPDATED), so a prefs edit or the
        deletion of the retired device clears it. The desired set applied
        through the DeviceRuntime-owned IssueReconciler is the union with the
        device_record_missing problems (Plan 04-05), so one pass can never
        delete the other set's issues. Confirmed devices never raise it.
        """
        prefs = await async_get_prefs(self.hass)
        consumption: set[str] = set()
        if isinstance(prefs, Mapping):
            device_consumption = prefs.get("device_consumption")
            if isinstance(device_consumption, list):
                for item in device_consumption:
                    if isinstance(item, dict):
                        stat = item.get("stat_consumption")
                        if isinstance(stat, str) and stat:
                            consumption.add(stat)
        ent_reg = er.async_get(self.hass)
        retired: list[Problem] = []
        for dev_uuid, rec in sorted(self._devices.items()):
            if rec.get("status") != "retired":
                continue
            entity_id = ent_reg.async_get_entity_id(
                "sensor", DOMAIN, f"{self.entry.entry_id}_{dev_uuid}_energy"
            )
            if entity_id is None or entity_id not in consumption:
                continue
            target = self._devices.get(str(rec.get("retired_into") or ""))
            retired.append(
                Problem(
                    issue_id=f"retired_device_in_energy_{dev_uuid}",
                    kind="retired_device_in_energy",
                    severity=Severity.WARNING,
                    translation_key="retired_device_in_energy",
                    placeholders={
                        "name": str(rec.get("name") or dev_uuid),
                        "entity": entity_id,
                        "target": str((target or {}).get("name") or "another device"),
                    },
                )
            )
        self._retired_problems = retired
        self._issues.apply([*self._device_problems, *self._retired_problems])

    def unconfirmed_candidates(self) -> list[dict[str, Any]]:
        """Candidate-shaped dicts for unconfirmed records (D-05).

        Built from the record's active artefact so a deleted device can
        always be re-confirmed from Review, even when the current learning
        run's candidates no longer contain its signature: the device record
        is the surviving artefact of truth.
        """
        out: list[dict[str, Any]] = []
        for dev_uuid, rec in sorted(self._devices.items()):
            if rec.get("status") != "unconfirmed":
                continue
            artefact = self._active_artefact(rec)
            sig = str(rec.get("signature_id") or "")
            if artefact is None or not sig:
                continue
            name = str(rec.get("name") or sig)
            cand: dict[str, Any] = dict(artefact)
            scores: dict[str, Any] = {}
            for version in rec.get("versions") or []:
                if isinstance(version, Mapping) and version.get("v") == rec.get(
                    "active_v"
                ):
                    raw = version.get("scores")
                    scores = dict(raw) if isinstance(raw, Mapping) else {}
            cand.update(
                {
                    "signature_id": sig,
                    "kind": rec.get("kind") or "switched",
                    "label": name,
                    "first_label": name,
                    "scores": scores,
                    "identity": {
                        "status": "matched",
                        "matched": sig,
                        "merged_into": None,
                        "twin_of": None,
                    },
                    "fingerprint": {
                        "kind": rec.get("kind") or "switched",
                        "states": artefact.get("states"),
                    },
                }
            )
            out.append(cand)
        return out

    def energy_wh(self, uuid: str) -> float:
        """Return the stored energy total for a device (0.0 when absent)."""
        return self._energy_wh.get(uuid, 0.0)

    def attributes(self, uuid: str) -> dict[str, Any]:
        """Return the OUT-04 attribute set: six static keys plus the D-09/D-16 tail.

        The key order (status, confidence, validation_error_pct,
        validation_basis, last_trained, preview, validation_class,
        validation_peers, unvalidated, drift, drift_reason, drift_since) is
        part of the D-17 contract with the Plan 05-04 keys and the Plan 05-05
        drift keys appended after the static set and before the conditional
        retired key: both device sensors expose the same keys in the same
        order on every state write. drift, drift_reason and drift_since are
        the D-16 drift flags (false/None/None unless set): the live proxy
        from Plan 05-05 merged with the D-23 class-holdout drift of the class
        figure this device shows; drift only sets flags and never changes
        the device's versions, active_v, uuid, name or entity ids. status is
        "provisional" unless a stored plug-session result validates the device
        (Plan 06-09: it passed the P5 device bar, its version is still active,
        it is not revoked and the live drift flag is false), in which case it is
        "validated" and validation_error_pct/validation_basis show the session
        figure with basis "plug_session"; a preview-confirmed device carries
        preview true instead of a different status. confidence is the D-21 clip of the
        validation share over the train share, None for preview versions or a
        zero train share. validation_error_pct is the D-16 learning-split
        figure with basis "learning_split", replaced by the class median with
        basis "class_holdout" only while the device's class passes D-08 in
        the latest ok report (D-09, D-20); both are None for preview
        candidates, program candidates and when the reference energy is 0
        (D-16), and for candidates stored before the figure existed until the
        next learning run or rescore. validation_class is the D-06 class
        derived from the active version (refreshed on load, confirm, version
        activation and each validation), validation_peers the class's peer
        count in the latest ok report (0 without one) and unvalidated the one
        filterable VAL-03 flag: True on every device without a holdout-backed
        figure.
        """
        keys = (
            "status",
            "confidence",
            "validation_error_pct",
            "validation_basis",
            "last_trained",
            "preview",
            "validation_class",
            "validation_peers",
            "unvalidated",
            "drift",
            "drift_reason",
            "drift_since",
        )
        rec = self._devices.get(uuid)
        if rec is None:
            return {k: None for k in keys}
        scores_view = self._score_view(rec)
        view = self._validation_view(
            rec,
            scores_view["validation_error_pct"],
            scores_view["validation_basis"],
        )
        # D-23: the D-16 drift attributes merge the live drift flags with the
        # class-holdout drift of the class figure this device shows; keys and
        # order are the unchanged twelve-key D-17 contract.
        drift_view = self._drift_view(uuid)
        drift, drift_reason, drift_since = (
            drift_view["drift"],
            drift_view["reason"],
            drift_view["since"],
        )
        attrs: dict[str, Any] = {
            "status": "validated" if view["session_validated"] else "provisional",
            "confidence": scores_view["confidence"],
            "validation_error_pct": view["validation_error_pct"],
            "validation_basis": view["validation_basis"],
            "last_trained": scores_view["last_trained"],
            "preview": bool(rec.get("preview")),
            "validation_class": view["validation_class"],
            "validation_peers": view["validation_peers"],
            "unvalidated": view["unvalidated"],
            "drift": drift,
            "drift_reason": drift_reason,
            "drift_since": drift_since,
        }
        # Plan 06-09 (REV-04, D-13): the session period, appended after
        # drift_since and only when a session result is stored, so both device
        # sensors show it after a revocation too (D-14) while a device without a
        # session keeps the unchanged twelve-key contract. Both keys are listed
        # in the sensors' _unrecorded_attributes.
        if view["has_session"]:
            attrs["validation_session_start"] = view["session_start"]
            attrs["validation_session_end"] = view["session_end"]
        # A retired device marks both sensors (D-12): the key follows the
        # fixed twelve-key order and only exists on retired records, so the
        # D-17 key-order contract for confirmed devices is unchanged.
        if rec.get("status") == "retired":
            attrs["retired"] = True
        return attrs

    @staticmethod
    def _drift_flags(rec: Mapping[str, Any]) -> tuple[bool, str | None, str | None]:
        """Return the record's live drift flags (D-16): false/None/None by default.

        The flags live in the record's validation dict (RESEARCH Pitfall 8:
        one record key, written only when a verdict changes); drift itself
        never touches versions, active_v, uuid, name or entity ids.
        """
        validation = rec.get("validation")
        validation = validation if isinstance(validation, Mapping) else {}
        stored = validation.get("drift")
        if not isinstance(stored, Mapping):
            return False, None, None
        drift = bool(stored.get("drift"))
        reason = stored.get("reason")
        since = stored.get("since")
        return (
            drift,
            str(reason) if drift and reason else None,
            str(since) if drift and since else None,
        )

    def _drift_view(self, uuid: str) -> dict[str, Any]:
        """Merge live drift and class-holdout drift for one device (D-23, D-16).

        Live drift is the record's validation.drift flags (Plan 05-05).
        Class-holdout drift is derived at read time from the shown class
        entry's drift fields, taken only when the record is holdout-backed by
        the same validation view helper Plan 05-04 uses for class_holdout
        (status confirmed, kind switched or program, its class entry passes)
        and that entry reads drift true. It is never written into the device
        record: it lives in the powerguesser.validation Store next to the class
        figure it qualifies, so the displayed figure and its decay flag can
        never disagree (D-26 (3)). drift is true when either is true; when
        both are true the class reason (validation_error_rising, measured
        against real plugs) and the class drift_since win (D-26 (2)); live
        drift_since stays an ISO timestamp and the class's an ISO date
        (D-26 (4)). A pure read of in-memory state: no I/O, safe in
        @callback paths.
        """
        rec = self._devices.get(uuid)
        if rec is not None:
            live_drift, live_reason, live_since = self._drift_flags(rec)
        else:
            live_drift, live_reason, live_since = False, None, None
        if rec is not None:
            scores_view = self._score_view(rec)
            view = self._validation_view(
                rec,
                scores_view["validation_error_pct"],
                scores_view["validation_basis"],
            )
            class_result = view.get("class_result")
            if (
                view.get("holdout_backed")
                and isinstance(class_result, Mapping)
                and class_result.get("drift") is True
            ):
                since = class_result.get("drift_since")
                return {
                    "drift": True,
                    "reason": REASON_VALIDATION_ERROR_RISING,
                    "since": str(since) if since else None,
                }
        return {"drift": bool(live_drift), "reason": live_reason, "since": live_since}

    def _score_view(self, rec: Mapping[str, Any]) -> dict[str, Any]:
        """Return the Phase 4 score view of a record (D-16, D-21).

        confidence, the learning-split validation_error_pct with its basis
        (the "learning_split" fallback, D-20) and last_trained come from the
        active version's scores, exactly as Phase 4 computed them; Plan 05-04
        only overrides the figure/basis downstream when a holdout basis
        exists, it never changes how the Phase 4 values are derived.
        """
        version: Mapping[str, Any] | None = None
        for entry in rec.get("versions") or []:
            if isinstance(entry, Mapping) and entry.get("v") == rec.get("active_v"):
                version = entry
                break
        raw_scores = version.get("scores") if version is not None else None
        scores = dict(raw_scores) if isinstance(raw_scores, Mapping) else {}
        train = scores.get("train")
        train = train if isinstance(train, Mapping) else {}
        val = scores.get("val")
        try:
            train_share = float(train.get("share") or 0.0)
        except (TypeError, ValueError):
            train_share = 0.0
        confidence = None
        if scores.get("validated") and isinstance(val, Mapping) and train_share > 0.0:
            try:
                ratio = float(val.get("share") or 0.0) / train_share
            except (TypeError, ValueError):
                ratio = 0.0
            confidence = round(min(1.0, max(0.0, ratio)), 3)
        vep = scores.get("validation_error_pct")
        try:
            vep = float(vep) if vep is not None else None
        except (TypeError, ValueError):
            vep = None
        basis = scores.get("validation_basis")
        if vep is None:
            basis = None
        elif not basis:
            basis = "learning_split"
        return {
            "confidence": confidence,
            "validation_error_pct": vep,
            "validation_basis": basis,
            "last_trained": version.get("trained_at") if version is not None else None,
        }

    def _session_view(self, rec: Mapping[str, Any]) -> dict[str, Any]:
        """Return the plug_session validation view of a record (Plan 06-09; REV-04, D-13).

        A device is validated via a plug session only when a stored session
        result passed the P5 device bar, its artefact version is still the
        device's active version, it has not been revoked (a live drift verdict
        or a version change, D-14/D-38) and the live drift flag is currently
        false. Otherwise the record keeps its Phase 4/5 basis and reads
        provisional. The session period is exposed whenever a result is stored,
        so it stays visible after a revocation (D-14). A pure read of in-memory
        state: no I/O, safe in @callback paths.
        """
        validation = rec.get("validation")
        validation = validation if isinstance(validation, Mapping) else {}
        stored = validation.get("session")
        if not isinstance(stored, Mapping):
            return {
                "has_session": False,
                "validated": False,
                "validation_error_pct": None,
                "session_start": None,
                "session_end": None,
            }
        try:
            artefact_v = int(stored.get("artefact_v") or 0)
        except (TypeError, ValueError):
            artefact_v = 0
        try:
            active_v = int(rec.get("active_v") or 0)
        except (TypeError, ValueError):
            active_v = 0
        live_drift, _drift_reason, _drift_since = self._drift_flags(rec)
        validated = bool(
            rec.get("status") == "confirmed"
            and stored.get("passes")
            and stored.get("revoked") is None
            and not live_drift
            and artefact_v == active_v
        )
        return {
            "has_session": True,
            "validated": validated,
            "validation_error_pct": stored.get("energy_error_pct"),
            "session_start": stored.get("window_start"),
            "session_end": stored.get("window_end"),
        }

    def _validation_view(
        self,
        rec: Mapping[str, Any],
        vep: float | None,
        basis: str | None,
    ) -> dict[str, Any]:
        """Return the validation view of a record (D-09, D-10, D-19, D-20).

        A figure is holdout-backed only for a confirmed switched OR program
        device whose stored class passes D-08 in the latest ok report (T-05-11,
        Plan 06-04): it then shows the class median with basis "class_holdout".
        A program device's class comes from its stage template (Plan 06-11
        derive_class), never from its degenerate session rise boxes (RESEARCH
        Pitfall 1). Every other record — no report, a refused latest report, a
        failing class, a retired record — keeps its Phase 4 learning-split
        figure (None for program candidates, D-20) and reads unvalidated True.
        validation_class is the stored derived class (None only before the
        first refresh) and validation_peers the class's peer count in the
        latest ok report, 0 without one.

        Plan 06-09 (REV-04, D-13): a stored passing plug-session result whose
        version is still active, is not revoked and whose live drift is false
        outranks everything: the device shows the session figure with basis
        "plug_session" and reads validated.
        """
        validation = rec.get("validation")
        validation = validation if isinstance(validation, Mapping) else {}
        stored_class = validation.get("class")
        stored_class = str(stored_class) if stored_class else None
        class_result = (
            self._class_results.get(stored_class)
            if stored_class is not None
            else None
        )
        session = self._session_view(rec)
        session_validated = bool(session["validated"])
        if session_validated:
            vep = session["validation_error_pct"]
            basis = "plug_session"
        holdout_backed = (
            rec.get("status") == "confirmed"
            and rec.get("kind") in ("switched", "program")
            and isinstance(class_result, Mapping)
            and bool(class_result.get("passes"))
        )
        if holdout_backed and not session_validated:
            vep = class_result.get("median_error_pct")
            basis = "class_holdout"
        peers = 0
        if isinstance(class_result, Mapping):
            try:
                peers = int(class_result.get("peers") or 0)
            except (TypeError, ValueError):
                peers = 0
        return {
            "validation_error_pct": vep,
            "validation_basis": basis,
            "validation_class": stored_class,
            "validation_peers": peers,
            "unvalidated": not (holdout_backed or session_validated),
            # Internal: consumed by attributes()/validation_details; never published.
            "holdout_backed": holdout_backed,
            "session_validated": session_validated,
            "has_session": bool(session["has_session"]),
            "session_start": session["session_start"],
            "session_end": session["session_end"],
            "class_result": class_result if holdout_backed else None,
        }

    def validation_details(self, uuid: str) -> tuple[float | None, dict[str, Any]]:
        """Return (state value, attributes) for the device's validation sensor (D-14, D-19).

        The state is the holdout-backed energy error %, None — the sensor
        reads unknown — whenever the device has no holdout-backed figure.
        Attributes are the fixed thirteen-key order; all except unvalidated,
        validation_basis, validation_class, validation_peers and
        drift_evidence are in the sensor's _unrecorded_attributes. The metric
        values (nde, mae_w, baseline_nde, n_events), the holdout window and
        train_cutoff ride from the class result and report meta only when
        holdout-backed, None otherwise. drift_evidence ("insufficient" |
        "sufficient") comes from the device's latest drift verdict, falling
        back to the tracker's live window state; it reads "insufficient"
        before any evaluation (D-15, T-05-18).
        """
        if uuid not in self._devices:
            return None, {
                "unvalidated": True,
                "validation_basis": None,
                "validation_class": None,
                "validation_peers": 0,
                "validated_via": None,
                "nde": None,
                "mae_w": None,
                "baseline_nde": None,
                "n_events": None,
                "holdout_start": None,
                "holdout_end": None,
                "train_cutoff": None,
                "drift_evidence": "insufficient",
            }
        rec = self._devices[uuid]
        scores_view = self._score_view(rec)
        view = self._validation_view(
            rec,
            scores_view["validation_error_pct"],
            scores_view["validation_basis"],
        )
        verdict = self._drift_verdicts.get(uuid)
        tracker = self._drift.get(uuid)
        if verdict is not None:
            drift_evidence = "insufficient" if verdict.insufficient else "sufficient"
        elif tracker is not None:
            drift_evidence = tracker.evidence_state()
        else:
            drift_evidence = "insufficient"
        class_result = view["class_result"]
        if view["session_validated"]:
            validation = rec.get("validation")
            stored = (
                validation.get("session")
                if isinstance(validation, Mapping)
                else None
            )
            stored = stored if isinstance(stored, Mapping) else {}
            state = stored.get("energy_error_pct")
            validated_via = "plug_session"
            nde = None
            mae_w = None
            baseline_nde = None
            n_events = None
            holdout_start = stored.get("holdout_start")
            holdout_end = stored.get("holdout_end")
            train_cutoff = self._report_meta.get("train_cutoff")
        elif view["holdout_backed"] and isinstance(class_result, Mapping):
            state = class_result.get("median_error_pct")
            validated_via: str | None = view["validation_class"]
            nde = class_result.get("median_nde")
            mae_w = class_result.get("median_mae_w")
            baseline_nde = class_result.get("median_baseline_nde")
            n_events = class_result.get("n_events")
            holdout_start = self._report_meta.get("holdout_start")
            holdout_end = self._report_meta.get("holdout_end")
            train_cutoff = self._report_meta.get("train_cutoff")
        else:
            state = None
            validated_via = None
            nde = None
            mae_w = None
            baseline_nde = None
            n_events = None
            holdout_start = None
            holdout_end = None
            train_cutoff = None
        return state, {
            "unvalidated": view["unvalidated"],
            "validation_basis": view["validation_basis"],
            "validation_class": view["validation_class"],
            "validation_peers": view["validation_peers"],
            "validated_via": validated_via,
            "nde": nde,
            "mae_w": mae_w,
            "baseline_nde": baseline_nde,
            "n_events": n_events,
            "holdout_start": holdout_start,
            "holdout_end": holdout_end,
            "train_cutoff": train_cutoff,
            "drift_evidence": drift_evidence,
        }

    def coverage_summary(self) -> list[dict[str, Any]]:
        """Return the VAL-03 coverage list: one entry per confirmed device (D-09).

        Each entry carries the device's name, validation_class,
        validation_basis and unvalidated, built from the same validation
        view as the sensors, sorted by name. validation.py stamps this onto
        the returned and stored report as ``confirmed_devices`` after
        applying the report.
        """
        out: list[dict[str, Any]] = []
        for rec in self._devices.values():
            if rec.get("status") != "confirmed":
                continue
            scores_view = self._score_view(rec)
            view = self._validation_view(
                rec,
                scores_view["validation_error_pct"],
                scores_view["validation_basis"],
            )
            out.append(
                {
                    "name": str(rec.get("name") or ""),
                    "validation_class": view["validation_class"],
                    "validation_basis": view["validation_basis"],
                    "unvalidated": view["unvalidated"],
                }
            )
        out.sort(key=lambda entry: entry["name"])
        return out

    @callback
    def apply_validation(self, report: Mapping[str, Any] | None) -> None:
        """Apply the latest validation report to classes and device views (D-06, D-09).

        Only a Mapping report with status "ok" contributes class results and
        holdout meta; anything else — no report, a refusal, a corrupt Store
        (T-05-12) — clears them, so a stale or refused report never validates
        a class (T-05-11). The full class entries stay in
        ``self._class_results``: Plan 05-06 reads them for the class-holdout
        drift (D-23), so no subset of fields is kept. Classes are refreshed
        and the snapshot republished, so after run_validation the device
        sensors show the new values without waiting for an aligner publish.
        """
        classes: dict[str, Mapping[str, Any]] = {}
        meta: dict[str, Any] = {
            "holdout_start": None,
            "holdout_end": None,
            "train_cutoff": None,
        }
        if isinstance(report, Mapping) and report.get("status") == "ok":
            raw_classes = report.get("classes")
            if isinstance(raw_classes, list):
                for entry in raw_classes:
                    if isinstance(entry, Mapping) and entry.get("class") is not None:
                        classes[str(entry["class"])] = entry
            holdout = report.get("holdout")
            if isinstance(holdout, Mapping):
                meta["holdout_start"] = holdout.get("start")
                meta["holdout_end"] = holdout.get("end")
            meta["train_cutoff"] = report.get("train_cutoff")
        self._class_results = classes
        self._report_meta = meta
        # Plan 06-09 (REV-04, D-13): the separate sessions list updates a stored
        # session result (only with a newer entry); a report without one leaves
        # every stored result and the validated status untouched (D-14).
        sessions_changed = self._apply_sessions(report)
        self._refresh_device_classes()
        if sessions_changed:
            self._store.async_delay_save(
                lambda: {"devices": self._devices}, DEVICE_STATE_SAVE_DELAY_S
            )
        self._republish()

    def _apply_sessions(self, report: Mapping[str, Any] | None) -> bool:
        """Store the report's sessions list into device records (Plan 06-09).

        Only a report carrying a sessions list updates a stored session result,
        and only with an entry for the same device that is newer by
        ``window_end``. A report without one — a shadow-only run, or a refused
        no_shadow_set report without device-target sessions — leaves every
        stored result and the validated status untouched (D-14), so a passing
        session is not cleared by an unrelated report. A fresh result is written
        with ``revoked`` None; drift and version activations revoke it (D-38),
        never this path. A revocation of the *same* session is permanent: a
        still-open device-target session is re-judged on every validation run
        with the run time as its ``window_end``, and that re-emission must not
        clear the revocation (WR-01, D-14/D-38). A different session replaces
        the revoked result as usual.
        """
        if not isinstance(report, Mapping):
            return False
        entries = report.get("sessions")
        if not isinstance(entries, list):
            return False
        changed = False
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            dev_uuid = str(entry.get("device_uuid") or "")
            rec = self._devices.get(dev_uuid)
            if rec is None or rec.get("status") != "confirmed":
                continue
            validation = rec.get("validation")
            if not isinstance(validation, dict):
                validation = {}
                rec["validation"] = validation
            stored = validation.get("session")
            new_end = str(entry.get("window_end") or "")
            if isinstance(stored, Mapping):
                # A revocation of this very session is permanent (D-14/D-38):
                # a still-open device-target session is re-judged on every
                # validation run with the run time as its window_end, and the
                # newer end time must not resurrect it (WR-01). A different
                # session is a new result and still replaces the stored one.
                same_session = stored.get("session_id") == str(
                    entry.get("session_id") or ""
                )
                if same_session and stored.get("revoked") is not None:
                    continue
                old_end = str(stored.get("window_end") or "")
                if new_end and old_end and new_end <= old_end:
                    continue
            try:
                artefact_v = int(entry.get("artefact_v") or 0)
            except (TypeError, ValueError):
                artefact_v = 0
            validation["session"] = {
                "session_id": str(entry.get("session_id") or ""),
                "artefact_v": artefact_v,
                "passes": bool(entry.get("passes")),
                "energy_error_pct": entry.get("energy_error_pct"),
                "window_start": entry.get("window_start"),
                "window_end": entry.get("window_end"),
                "holdout_start": entry.get("holdout_start"),
                "holdout_end": entry.get("holdout_end"),
                "validated_at": dt_util.utcnow().isoformat(),
                "revoked": None,
            }
            changed = True
        return changed

    @callback
    def _refresh_device_classes(self) -> None:
        """Derive every confirmed or retired device's class from its active artefact (D-06).

        ``derive_class`` applies the D-06 band hysteresis against the class
        stored in the record's validation dict (RESEARCH Pitfall 8: exactly
        one record key holds all per-device validation state). Called on
        load, confirm, version activation and each validation. Only a change
        schedules a cold save.
        """
        changed = False
        for rec in self._devices.values():
            if rec.get("status") not in ("confirmed", "retired"):
                continue
            artefact = self._active_artefact(rec)
            if artefact is None:
                continue
            validation = rec.get("validation")
            if not isinstance(validation, dict):
                validation = {}
                rec["validation"] = validation
            previous = validation.get("class")
            cls = derive_class(artefact, previous=str(previous) if previous else None)
            if cls is None:
                continue
            if cls != previous:
                validation["class"] = cls
                changed = True
        if changed:
            self._store.async_delay_save(
                lambda: {"devices": self._devices}, DEVICE_STATE_SAVE_DELAY_S
            )

    def live_attributes(self, uuid: str) -> dict[str, Any]:
        """Return the live power-sensor attributes (D-14, D-17, D-07).

        match_score is the live box fit of the current ON episode, rounded to
        3 decimals, None while the device is OFF (D-14). unattributed_edges is
        this device's refused-edge count, from the latest StepResult and the
        restored detector state (D-08/D-17). update_available is True while a
        learning-run proposal is pending (D-11). overlap_refused is this
        device's ambiguous-overlap refusal count (D-07/D-41), from the
        detector state's per-device map. All four keys are power-sensor only
        and listed in its _unrecorded_attributes.
        """
        match_score = None
        unattributed = 0
        overlap_refused = 0
        state = self._detector_state
        if isinstance(state, Mapping):
            dev_state = (state.get("devices") or {}).get(uuid)
            if isinstance(dev_state, Mapping):
                unattributed = int(dev_state.get("unattributed", 0) or 0)
            by_device = state.get("overlap_refused_by_device")
            if isinstance(by_device, Mapping):
                overlap_refused = int(by_device.get(uuid, 0) or 0)
        last = self._last_step
        if last is not None:
            if uuid in last.on:
                score = last.match_score.get(uuid)
                if score is not None:
                    match_score = round(float(score), 3)
            unattributed = int(last.unattributed_edges.get(uuid, unattributed))
        return {
            "match_score": match_score,
            "unattributed_edges": unattributed,
            "update_available": self.update_available(uuid),
            "overlap_refused": overlap_refused,
        }

    @callback
    def raise_energy_floor(
        self, uuid: str, restored_wh: float, *, include_delta: bool = True
    ) -> None:
        """Raise the energy total to at least restored (+ increments since load).

        RestoreSensor fallback (T-04-14, review concern L153@4e11826): the
        detector may step before the entity restores its state (hub.async_start
        runs before the platform forward), so the floor is
        ``max(current, restored + delta_since_load)`` — order-independent,
        it never lowers the total and never double-counts an increment.
        ``include_delta`` is False when the restored value was recorded
        during this session (its increments are already in both numbers,
        e.g. a deleted device re-confirmed minutes later). Non-numeric
        restore data is ignored by the caller.
        """
        try:
            restored = float(restored_wh)
        except (TypeError, ValueError):
            return
        if not math.isfinite(restored) or restored <= 0.0:
            return
        delta = self._delta_since_load.get(uuid, 0.0) if include_delta else 0.0
        total = max(self._energy_wh.get(uuid, 0.0), restored + delta)
        if total > self._energy_wh.get(uuid, 0.0):
            self._energy_wh[uuid] = total
            self._state_store.async_delay_save(
                self._hot_store_data, DEVICE_STATE_SAVE_DELAY_S
            )
            # The raised floor must reach the sensors without waiting for
            # the next aligner publish.
            self._republish()

    @property
    def load_timestamp(self) -> float:
        """Timestamp of the current session's async_load (restore guard)."""
        return self._load_ts

    async def async_flush(self) -> None:
        """Flush both device Stores (called on entry unload).

        A pending delay-save of a stale Store instance must never overwrite
        a reload (Pitfall 5): the hot data includes the live detector state.
        """
        if self._detector is not None:
            self._detector_state = self._detector.state()
        await self._async_save_cold()
        await self._async_save_hot()

    def power_w(self, uuid: str) -> float:
        """Return the detector-reported device power (0.0 while OFF).

        Plan 04-04's detector sets ON devices' power behind this same method.
        """
        return float(self._device_w.get(uuid, 0.0))

    async def _async_save_cold(self) -> None:
        await self._store.async_save({"devices": self._devices})

    async def _async_save_hot(self) -> None:
        await self._state_store.async_save(self._hot_store_data())

    async def async_confirm(
        self,
        candidate: dict[str, Any],
        name: str,
        *,
        preview: bool,
        trained_at: str | None,
    ) -> str:
        """Confirm a candidate as a device: cold record, subentry, entities.

        The visibility/dedupe check (``device_for_signature``) and the in-memory
        record write happen with no await in between, so a concurrent second
        confirm returns the same uuid and creates nothing. The record is saved
        as status ``confirmed`` / ``subentry_id`` None before the subentry add;
        that window is exactly what the startup reconcile repairs.
        On ``AbortFlow`` (a subentry with this unique_id already exists, e.g.
        the record/subentry pair survived a crash between the two cold saves)
        the existing subentry's id is adopted.

        Re-confirm (D-05): an unconfirmed record for the same signature keeps
        its uuid, parameter versions and energy total, so the re-added
        entities get the same unique_ids and the energy sensor continues from
        its retained total (HA restores the user's entity_id from the
        deleted-entity registry row).
        """
        signature_id = str(candidate.get("signature_id") or "")
        existing = self.device_for_signature(signature_id)
        if existing is not None:
            return existing

        now = dt_util.utcnow().isoformat()
        artefact = {
            k: v
            for k, v in candidate.items()
            if k not in ("identity", "fingerprint", "first_label")
        }
        scores = dict(candidate.get("scores") or {})

        # Re-confirm reuses the unconfirmed record for the same signature.
        reuse_uuid: str | None = None
        for dev_uuid, rec in self._devices.items():
            if (
                rec.get("status") == "unconfirmed"
                and rec.get("signature_id") == signature_id
            ):
                reuse_uuid = dev_uuid
                break

        if reuse_uuid is not None:
            device_uuid = reuse_uuid
            rec = self._devices[device_uuid]
            rec["status"] = "confirmed"
            rec["subentry_id"] = None
            rec["name"] = name
            rec["preview"] = bool(preview)
            if _artefact_states_differ(self._active_artefact(rec), artefact):
                next_v = (
                    max(
                        (
                            int(v.get("v") or 0)
                            for v in rec.get("versions") or []
                            if isinstance(v, Mapping)
                        ),
                        default=0,
                    )
                    + 1
                )
                rec["versions"].append(
                    {
                        "v": next_v,
                        "artefact": artefact,
                        "trained_at": trained_at,
                        "scores": scores,
                        "accepted_at": now,
                        "source": "confirm",
                    }
                )
                rec["active_v"] = next_v
        else:
            device_uuid = uuid.uuid4().hex
            rec = {
                "name": name,
                "signature_id": signature_id,
                "aliases": [],
                "kind": candidate.get("kind") or "switched",
                "preview": bool(preview),
                "status": "confirmed",
                "retired_into": None,
                "subentry_id": None,
                "versions": [
                    {
                        "v": 1,
                        "artefact": artefact,
                        "trained_at": trained_at,
                        "scores": scores,
                        "accepted_at": now,
                        "source": "confirm",
                    }
                ],
                "active_v": 1,
                "pending_proposal": None,
                "created_at": now,
                "alias_artefacts": {},
                "pending_refit": False,
                "last_refit": None,
                "validation": {},
            }
            # No await between the dedupe check above and this in-memory write.
            self._devices[device_uuid] = rec
        self._energy_wh.setdefault(device_uuid, 0.0)
        await self._async_save_cold()
        await self._async_save_hot()

        subentry = ConfigSubentry(
            data=MappingProxyType(
                {
                    "device_uuid": device_uuid,
                    "name": name,
                    "signature_id": signature_id,
                }
            ),
            subentry_type=SUBENTRY_TYPE_DEVICE,
            title=name,
            unique_id=device_uuid,
        )
        subentry_id: str | None = None
        try:
            self.hass.config_entries.async_add_subentry(self.entry, subentry)
            subentry_id = subentry.subentry_id
        except AbortFlow:
            # A subentry with this unique_id already exists: adopt it instead of
            # creating a second one, so the reconcile never un-confirms a
            # device whose subentry exists.
            for sub in self.entry.get_subentries_of_type(SUBENTRY_TYPE_DEVICE):
                if sub.unique_id == device_uuid:
                    subentry_id = sub.subentry_id
                    break
            if subentry_id is None:
                raise
        rec["subentry_id"] = subentry_id
        await self._async_save_cold()
        await self.hub.learning.review.async_set_device_link(
            signature_id,
            name=name,
            fingerprint=candidate.get("fingerprint"),
            # The review-record label stays the plain recognition label; the
            # D-27 " — overlaps …" note is a display-only suffix (D-27).
            label=self.hub.learning.candidate_label(candidate, include_overlaps=False),
            device_uuid=device_uuid,
            subentry_id=subentry_id,
        )
        await self.hub.learning.async_after_confirm()
        # Plan 05-04: the confirmed device gains its D-06 class before the
        # reconcile adds its entities, so the first attribute write already
        # carries validation_class.
        self._refresh_device_classes()
        await self.async_reconcile()
        return device_uuid

    async def async_reconcile(self) -> None:
        """Idempotent diff of device subentries against confirmed records.

        For every device subentry whose ``device_uuid`` maps to a confirmed
        record without added entities, add the power and energy entities once.
        Lock-guarded because HA schedules update listeners as concurrent tasks.

        Lifecycle truth table (RESEARCH Pattern 5, D-05):

        - A confirmed record whose subentry is gone (deleted in Settings, or
          lost in the confirm crash window) becomes ``unconfirmed``: uuid,
          parameter versions, aliases, name and the hot energy total are kept;
          the review link is cleared so the candidate is reviewable again.
        - A confirmed record whose ``subentry_id`` is ``None`` while a
          subentry with its uuid exists (crash between ``async_add_subentry``
          and the second cold save) re-adopts that subentry instead.
        - A subentry whose title differs from the record name (rename through
          the Settings reconfigure flow) renames the record, the review
          record and the device registry entry; entity ids never change.
        """
        async with self._lock:
            subs = self.entry.get_subentries_of_type(SUBENTRY_TYPE_DEVICE)
            # Backup-mismatch problems (T-04-13): a device subentry without a
            # usable cold record gets unavailable sensors and a Repairs issue
            # instead of a crash or guessed parameters. Applied on every
            # reconcile (desired-set diff); deleting the subentry clears it.
            problems: list[Problem] = []
            for sub in subs:
                data: Mapping[str, Any] = sub.data
                device_uuid = data.get("device_uuid")
                rec = self._devices.get(device_uuid) if device_uuid else None
                if rec is None:
                    problems.append(
                        Problem(
                            issue_id=f"device_record_missing_{sub.subentry_id}",
                            kind="device_record_missing",
                            severity=Severity.WARNING,
                            translation_key="device_record_missing",
                            placeholders={
                                "name": str(data.get("name") or sub.title or "Device")
                            },
                        )
                    )
            # Keep the record-missing desired set for the union applied by
            # async_check_retired_issues (D-12), so neither pass can delete
            # the other set's issues.
            self._device_problems = problems
            self._issues.apply([*self._device_problems, *self._retired_problems])
            if self._add_entities is None:
                return
            sub_ids = {sub.subentry_id for sub in subs}
            # A removed subentry drops its entity-tracking id: a re-added
            # subentry (re-confirm) must get fresh entities.
            self._added_subentries &= sub_ids
            by_unique_id = {
                sub.unique_id: sub for sub in subs if sub.unique_id is not None
            }
            changed = False
            renamed: list[tuple[str, str, str]] = []
            for dev_uuid, rec in list(self._devices.items()):
                if rec.get("status") == "retired":
                    # A retired device whose subentry is gone (deleted in
                    # Settings) is removed entirely (D-12): unlike a
                    # confirmed device it never returns to Review, the
                    # survivor keeps its signature as an alias, and its
                    # retired_device_in_energy issue leaves the desired set
                    # at the async_check_retired_issues run below.
                    sub = None
                    subentry_id = rec.get("subentry_id")
                    if subentry_id is not None:
                        sub = next(
                            (s for s in subs if s.subentry_id == subentry_id), None
                        )
                    if sub is None and subentry_id is None:
                        sub = by_unique_id.get(dev_uuid)
                    if sub is None:
                        del self._devices[dev_uuid]
                        self._energy_wh.pop(dev_uuid, None)
                        self._state_store.async_delay_save(
                            self._hot_store_data, DEVICE_STATE_SAVE_DELAY_S
                        )
                        changed = True
                    continue
                if rec.get("status") != "confirmed":
                    continue
                sub = None
                subentry_id = rec.get("subentry_id")
                if subentry_id is not None:
                    sub = next(
                        (s for s in subs if s.subentry_id == subentry_id), None
                    )
                if sub is None and subentry_id is None:
                    sub = by_unique_id.get(dev_uuid)
                if sub is None:
                    # Un-confirm (D-05): the device record keeps everything
                    # the user can re-confirm with; the candidate returns to
                    # Review; the detector drops the model via the
                    # confirmed-set rebuild below (a deleted-while-ON device
                    # parks its unknown flag, D-29a).
                    rec["status"] = "unconfirmed"
                    rec["subentry_id"] = None
                    # The candidate returns to Review with the same
                    # parameters, but its live drift evidence no longer
                    # describes a confirmed device: drop the tracker (and the
                    # last verdict) and delay-save the hot Store so a restart
                    # cannot resurrect it (same pattern as the retired-removal
                    # branch above). The D-16 flags leave the cold record too
                    # (same D-05 reconcile philosophy: the record returns to
                    # its pre-confirm validation state) — a fresh tracker only
                    # rewrites them when a verdict *changes*, so a stale
                    # validation.drift would reappear on re-confirm while the
                    # new tracker still sits in insufficient evidence.
                    self._drift.pop(dev_uuid, None)
                    self._drift_verdicts.pop(dev_uuid, None)
                    validation = rec.get("validation")
                    if isinstance(validation, dict):
                        validation.pop("drift", None)
                    self._state_store.async_delay_save(
                        self._hot_store_data, DEVICE_STATE_SAVE_DELAY_S
                    )
                    changed = True
                    await self.hub.learning.review.async_clear_device_link(
                        str(rec.get("signature_id") or "")
                    )
                    continue
                if rec.get("subentry_id") != sub.subentry_id:
                    rec["subentry_id"] = sub.subentry_id
                    changed = True
                if sub.title and sub.title != rec.get("name"):
                    rec["name"] = sub.title
                    changed = True
                    renamed.append(
                        (dev_uuid, str(rec.get("signature_id") or ""), sub.title)
                    )
            if changed:
                await self._async_save_cold()
            if renamed:
                dev_reg = dr.async_get(self.hass)
                for dev_uuid, signature_id, name in renamed:
                    await self.hub.learning.review.async_rename_linked(
                        signature_id, name
                    )
                    device = dev_reg.async_get_device_by_identifier(
                        (DOMAIN, dev_uuid), self.entry.entry_id
                    )
                    if device is not None:
                        # Never touches name_by_user: the registry name is the
                        # fallback shown only while the user set no own name.
                        dev_reg.async_update_device(device.id, name=name)
            for sub in subs:
                if sub.subentry_id in self._added_subentries:
                    continue
                data = sub.data
                device_uuid = data.get("device_uuid")
                if not device_uuid:
                    # No unique_id can be formed for this subentry; the
                    # Repairs issue above is its whole surface (review
                    # concern L155@4e11826).
                    continue
                rec = self._devices.get(device_uuid)
                if rec is not None and rec.get("status") not in (
                    "confirmed",
                    "retired",
                ):
                    continue
                # Retired records re-add their entities too (D-12): the
                # subentry intentionally persists so the user's history and
                # Energy-dashboard entries survive restarts — the power
                # sensor reads unavailable (reason "retired") and the energy
                # sensor keeps serving the frozen total.
                # Imported here to avoid the package-level import cycle
                # (__init__ -> devices -> sensor -> __init__).
                from .sensor import (
                    DeviceEnergySensor,
                    DevicePowerSensor,
                    DeviceValidationSensor,
                )

                name = str((rec or {}).get("name") or sub.title or "Device")
                self._add_entities(
                    [
                        DevicePowerSensor(
                            self.hub, self.entry.entry_id, device_uuid, name
                        ),
                        DeviceEnergySensor(
                            self.hub, self.entry.entry_id, device_uuid, name
                        ),
                        DeviceValidationSensor(
                            self.hub, self.entry.entry_id, device_uuid, name
                        ),
                    ],
                    config_subentry_id=sub.subentry_id,
                )
                self._added_subentries.add(sub.subentry_id)
        # A new, removed or changed confirmed record must reach the detector
        # even when no entity was added (e.g. a second device confirmed while
        # one runs, or an un-confirm dropping a model).
        self._maybe_rebuild_detector()
        self._republish()
        # D-12: the retired_device_in_energy issues are re-evaluated at every
        # reconcile (a deleted retired device's issue disappears here).
        await self.async_check_retired_issues()


async def async_remove_device_data(hass: HomeAssistant) -> None:
    """Remove the device Stores (called from async_remove_entry).

    The cold (powerguesser.devices) and hot (powerguesser.device_state)
    Stores belong to the integration, not the config entry storage, so the
    entry removal must delete them explicitly.
    """
    for key in (DEVICES_STORE_KEY, DEVICE_STATE_STORE_KEY):
        await Store(hass, DEVICES_STORE_VERSION, key).async_remove()