"""PowerGuesser coordinator and runtime hub."""

import asyncio
from collections.abc import Callable, Mapping
import dataclasses
import functools
import logging
import math
from typing import Any

from homeassistant.config_entries import ConfigEntry, ConfigEntryState
from homeassistant.const import (
    ATTR_UNIT_OF_MEASUREMENT,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
    UnitOfPower,
)
from homeassistant.core import Event, EventStateChangedData, HomeAssistant, State, callback
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.event import (
    async_call_later,
    async_track_state_change_event,
    async_track_state_report_event,
    async_track_time_interval,
)
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.storage import Store
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util import dt as dt_util
from homeassistant.util.unit_conversion import PowerConverter

from .bootstrap import BootstrapRunner, load_stored_report
from .buffer_writer import BufferWriter
from .const import (
    BOOTSTRAP_START_TIMEOUT_S,
    CONF_INVERT_PHASES,
    CONF_PHASE_L1,
    CONF_PHASE_L2,
    CONF_PHASE_L3,
    DOMAIN,
    PHASE_OPTION_ROLES,
    PROFILE_SAVE_DELAY_S,
    PROFILE_STORE_KEY,
    PROFILE_STORE_VERSION,
    SIGNAL_PROFILE_SOURCES,
    STRUCTURAL_GRACE_S,
    TICK_INTERVAL,
)
from .energy_prefs import (
    async_get_prefs,
    collect_registry_facts,
    device_consumption_power_entities,
)
from .engine.buffer import (
    ChannelSpec,
    KIND_METERED,
    KIND_PHASE,
    KIND_SESSION,
    ROLE_SESSION,
    channel_specs_from_model,
)
from .engine.compute import OverAttribution, compute_snapshot
from .engine.intervals import (
    MIN_DELTAS,
    IntervalProfiler,
    ProfileSnapshot,
    ProfileSource,
    effective_interval,
    gate,
)
from .engine.model import Problem, Severity, Snapshot, SourceModel
from .engine.prefs import build_source_model
from .engine.selfcheck import SignSelfCheck, evaluate_plausibility
from .engine.settle import STALE_AFTER_S, ReReportDetector, SettleAligner
from .issues import IssueReconciler
from .review_store import (
    SESSION_END_DATE,
    SESSION_END_PLUG_METERED,
    resolve_session_target,
)


_LOGGER = logging.getLogger(__name__)


def parse_watts(state: State | None) -> float | None:
    """Parse and convert an entity state to watts, or None if invalid."""
    if state is None:
        return None
    if state.state in (STATE_UNAVAILABLE, STATE_UNKNOWN):
        return None
    try:
        val = float(state.state)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(val):
        return None
    unit = state.attributes.get(ATTR_UNIT_OF_MEASUREMENT)
    if unit not in PowerConverter.VALID_UNITS:
        return None
    try:
        return float(PowerConverter.convert(val, unit, UnitOfPower.WATT))
    except Exception:
        return None


class PowerGuesserHub:
    """Central manager for event tracking, calculation, and sensor updates."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        writer: BufferWriter,
    ) -> None:
        """Initialize the hub."""
        self.hass = hass
        self.entry = entry
        self.writer = writer
        self.modes: dict[str, ReReportDetector] = {}
        self.profilers: dict[str, IntervalProfiler] = {}
        self.profile_sources: dict[str, ProfileSource] = {}
        self._entity_channels: dict[str, tuple[int, ...]] = {}
        self._channel_entities: dict[int, str] = {}
        self.coordinator = DataUpdateCoordinator[Snapshot](
            hass,
            _LOGGER,
            config_entry=entry,
            name=DOMAIN,
            update_interval=None,
        )
        self.profile_coordinator = DataUpdateCoordinator[ProfileSnapshot](
            hass,
            _LOGGER,
            config_entry=entry,
            name=f"{DOMAIN}_profile",
            update_interval=None,
        )
        self.bootstrap_coordinator = DataUpdateCoordinator[dict[str, Any]](
            hass,
            _LOGGER,
            config_entry=entry,
            name=f"{DOMAIN}_bootstrap",
            update_interval=None,
        )
        self.bootstrap_coordinator.data = {"status": None}
        self.bootstrap_task: asyncio.Task[None] | None = None
        self._stored_bootstrap_report: dict[str, Any] = {}
        self._bootstrap_gate = asyncio.Event()
        self._unsub_bootstrap_timer: Callable[[], None] | None = None
        self._last_profile_publish_key: Any = None
        self.reconciler = IssueReconciler(hass, entry)
        self._profile_store = Store[dict[str, Any]](
            hass, PROFILE_STORE_VERSION, PROFILE_STORE_KEY
        )
        self._stored_profiles: dict[str, Any] = {}
        self.model: SourceModel = SourceModel()
        self.aligner: SettleAligner = SettleAligner({})
        self.over_attribution = OverAttribution()
        self.self_check = SignSelfCheck()
        self._cached_active_checks: frozenset[str] = frozenset()
        self._metered_values: dict[str, float | None] = {}
        # Active plug sessions (Plan 06-06): the plug value per session plug
        # entity, and the subset whose target is a candidate and is therefore
        # merged into the live metered mapping (D-25).
        self._session_values: dict[str, float | None] = {}
        self._session_metered_plugs: set[str] = set()
        # Earliest planned end date across the active sessions, cached at every
        # rebuild and checked on the hub's existing tick (D-11); the task slot
        # keeps the ending single-flight.
        self._session_planned_end_ts: str | None = None
        self._session_end_task: asyncio.Task[None] | None = None

        self._unsub_trackers: list[Callable[[], None]] = []
        self._unsub_at_started: Callable[[], None] | None = None
        self._unsub_grace: Callable[[], None] | None = None
        self._unsub_er: Callable[[], None] | None = None
        self._unsub_tick: Callable[[], None] | None = None
        self._last_available: bool | None = None
        self._structural_ready: bool = False

        self._rebuild_running: bool = False
        self._rebuild_pending: bool = False
        self._stopped: bool = False
        self._setup_done: bool = False

    async def async_start(self) -> None:
        """Load preferences, build model, start trackers, and initialize timers."""
        self._structural_ready = False

        @callback
        def _start_grace_timer(_hass: HomeAssistant) -> None:
            self._unsub_grace = async_call_later(
                self.hass, STRUCTURAL_GRACE_S, self._on_structural_grace_expired
            )

        self._unsub_at_started = async_at_started(self.hass, _start_grace_timer)

        @callback
        def _on_er_updated(event: Event) -> None:
            entity_id = event.data.get("entity_id")
            old_entity_id = event.data.get("old_entity_id")
            if old_entity_id and old_entity_id in self._entity_channels and entity_id:
                self.hass.async_create_task(
                    self._async_follow_rename(old_entity_id, entity_id)
                )
                return
            phase_eids = {
                self.entry.options[k]
                for k in PHASE_OPTION_ROLES
                if k in self.entry.options and self.entry.options[k]
            }
            refs = set(self.model.referenced_entity_ids) | phase_eids
            if entity_id in refs or old_entity_id in refs:
                self.async_request_rebuild()

        self._unsub_er = self.hass.bus.async_listen(
            er.EVENT_ENTITY_REGISTRY_UPDATED, _on_er_updated
        )
        self._unsub_tick = async_track_time_interval(
            self.hass, self._on_tick, TICK_INTERVAL
        )

        self._stored_profiles = {}
        stored = await self._profile_store.async_load()
        if isinstance(stored, dict) and isinstance(stored.get("sources"), dict):
            self._stored_profiles = stored["sources"]

        # Awaited before the bootstrap task exists, so its first waiting publish has the report
        self._stored_bootstrap_report = await self.hass.async_add_executor_job(
            load_stored_report, self.writer.db
        )

        await self.async_rebuild()


    @callback
    def _on_structural_grace_expired(self, _now: Any) -> None:
        """Flip structural readiness flag and update Repairs issues after grace period."""
        self._structural_ready = True
        self._reconcile_issues()

    @callback
    def async_request_rebuild(self) -> None:
        """Request a model and tracker rebuild scheduled on the event loop, coalescing bursts."""
        if self._rebuild_running:
            self._rebuild_pending = True
            return
        self.hass.async_create_task(self._run_rebuild())

    async def _run_rebuild(self) -> None:
        """Run the rebuild task and process any pending requests."""
        self._rebuild_running = True
        try:
            await self.async_rebuild()
            while self._rebuild_pending:
                self._rebuild_pending = False
                await self.async_rebuild()
        finally:
            self._rebuild_running = False

    async def async_rebuild(self) -> None:
        """Re-read Energy dashboard preferences, rebuild engine model, and restart tracking."""
        self._clear_trackers()

        prefs = await async_get_prefs(self.hass)
        facts = collect_registry_facts(self.hass, prefs)
        self.model = build_source_model(prefs, facts=facts)

        self.self_check.reset()
        self._cached_active_checks = frozenset()
        self._reconcile_issues()


        # Configuration faults never raise setup errors; gross/remainder go unavailable while entry stays loaded
        if self.model.fatal:
            _LOGGER.warning(
                "PowerGuesser configuration is fatal; sensors marked unavailable"
            )
            now_ms = round(dt_util.utcnow().timestamp() * 1000)
            self.writer.set_active((), now_ms)
            self.coordinator.async_set_updated_data(
                Snapshot(
                    available=False,
                    gross_w=None,
                    remainder_w=None,
                    signed_residual_w=None,
                    metered_w=0.0,
                    metered_unavailable=(),
                    over_attribution_wh=self.over_attribution.total_wh,
                    clipped=False,
                    ts=dt_util.utcnow().timestamp(),
                )
            )
            self._republish_profiles()
            return

        specs, layout = channel_specs_from_model(self.model)

        gross_or_metered = set(self.model.gross_entity_ids) | set(self.model.metered_entity_ids)
        phase_specs: list[ChannelSpec] = []
        phase_eids: list[str] = []
        for opt_key, role in PHASE_OPTION_ROLES.items():
            eid = self.entry.options.get(opt_key)
            if not eid:
                continue
            if eid in gross_or_metered:
                _LOGGER.warning(
                    "Phase entity %s is already configured as a gross or metered entity; skipping phase mapping",
                    eid,
                )
                continue
            phase_specs.append(ChannelSpec(eid, role, KIND_PHASE))
            phase_eids.append(eid)

        specs.extend(phase_specs)

        # Active plug sessions (Plan 06-06, D-09/D-10/D-25): one session channel
        # per plug entity, the plug tracked and recorded on it, and — for a
        # candidate target only — the plug value merged into the live metered
        # mapping so the remainder drops by it exactly once. A device target is
        # recorded but not subtracted live, so the device's own sensor and drift
        # evidence keep running.
        session_specs: list[ChannelSpec] = []
        session_plugs: list[str] = []
        session_planned_ends: list[str] = []
        self._session_values = {}
        self._session_metered_plugs = set()
        review = self._learning_review()
        if review is not None:
            # D-10/T-06-15: a later Energy dashboard edit can turn an active
            # session's plug into a metered device or a gross source. Subtracting
            # it live as a session plug AND as metered would double-count it, so
            # the session ends here (reason plug_metered) and the plug is
            # subtracted once, as metered, by the specs below.
            now_metered_or_source = (
                set(self.model.gross_entity_ids)
                | set(self.model.referenced_entity_ids)
                | set(phase_eids)
                | set(device_consumption_power_entities(prefs, facts))
            )
            for session in review.active_sessions(dt_util.utcnow()):
                plug = str(session.get("plug_entity_id") or "")
                if not plug or plug not in now_metered_or_source:
                    continue
                await review.async_end_session(
                    session.get("session_id"), SESSION_END_PLUG_METERED
                )
                _LOGGER.info(
                    "Plug session %s on %s ended with reason plug_metered: the "
                    "plug is now an Energy dashboard source or metered device",
                    session.get("session_id"),
                    plug,
                )

            decisions = review.decisions
            for session in review.active_sessions(dt_util.utcnow()):
                plug = str(session.get("plug_entity_id") or "")
                if (
                    not plug
                    or plug in gross_or_metered
                    or plug in phase_eids
                    or plug in self._session_values
                ):
                    continue
                resolved = resolve_session_target(
                    session, decisions, self._device_lookup
                )
                if resolved is None:
                    continue
                planned = session.get("planned_end_ts")
                if isinstance(planned, str):
                    session_planned_ends.append(planned)
                session_specs.append(ChannelSpec(plug, ROLE_SESSION, KIND_SESSION))
                session_plugs.append(plug)
                if resolved[0] == "candidate":
                    self._session_metered_plugs.add(plug)

        self._session_planned_end_ts = (
            min(session_planned_ends) if session_planned_ends else None
        )

        specs.extend(session_specs)

        phase_invert = bool(self.entry.options.get(CONF_INVERT_PHASES, False))
        id_map = await self.hass.async_add_executor_job(
            functools.partial(
                self.writer.db.ensure_channels,
                specs,
                layout,
                phase_invert=phase_invert,
            )
        )
        entity_channels: dict[str, list[int]] = {}
        channel_entities: dict[int, str] = {}
        metered_channels: set[int] = set()
        session_channels: set[int] = set()
        for spec in specs:
            key = (spec.entity_id, spec.role)
            if key in id_map:
                ch_id = id_map[key]
                entity_channels.setdefault(spec.entity_id, []).append(ch_id)
                channel_entities[ch_id] = spec.entity_id
                if spec.kind == KIND_METERED:
                    metered_channels.add(ch_id)
                elif spec.kind == KIND_SESSION:
                    session_channels.add(ch_id)

        self._entity_channels = {e: tuple(ch_ids) for e, ch_ids in entity_channels.items()}
        self._channel_entities = channel_entities

        now_ms = round(dt_util.utcnow().timestamp() * 1000)
        newly = self.writer.set_active(channel_entities.keys(), now_ms)
        # Session channels are held like metered ones: no age limit, never
        # stale-marked. The assignment replaces the set every rebuild, so an
        # ended session leaves held_channels here (Plan 06-06 retention contract).
        self.writer.held_channels = frozenset(metered_channels | session_channels)
        self.writer.re_reporting = lambda ch: (
            self.modes[self._channel_entities[ch]].re_reporting
            if ch in self._channel_entities and self._channel_entities[ch] in self.modes
            else False
        )
        self.writer.stale_after_ms = lambda ch: (
            round(self.profilers[self._channel_entities[ch]].stale_after_s() * 1000)
            if ch in self._channel_entities and self._channel_entities[ch] in self.profilers
            else 90_000
        )

        all_tracked = (
            set(self.model.gross_entity_ids)
            | set(self.model.metered_entity_ids)
            | set(phase_eids)
        )
        for eid in all_tracked:
            if eid not in self.modes:
                self.modes[eid] = ReReportDetector()
            if eid not in self.profilers:
                profiler = IntervalProfiler(self.modes[eid])
                if eid in self._stored_profiles:
                    entry_data = self._stored_profiles[eid]
                    if isinstance(entry_data, dict):
                        try:
                            profiler.restore(
                                entry_data.get("median_s"),
                                entry_data.get("p90_s"),
                                entry_data.get("mode"),
                            )
                        except Exception:
                            pass
                self.profilers[eid] = profiler
        self.modes = {eid: d for eid, d in self.modes.items() if eid in all_tracked}
        self.profilers = {eid: p for eid, p in self.profilers.items() if eid in all_tracked}

        profile_sources: dict[str, ProfileSource] = {}
        for ch in self.model.gross_channels:
            for eid in ch.entity_ids:
                state = self.hass.states.get(eid)
                label = state.name if state and state.name else eid
                profile_sources[eid] = ProfileSource(
                    entity_id=eid,
                    role=ch.role.value if hasattr(ch.role, "value") else str(ch.role),
                    label=label,
                )
        for eid in self.model.metered_entity_ids:
            state = self.hass.states.get(eid)
            label = state.name if state and state.name else eid
            profile_sources[eid] = ProfileSource(
                entity_id=eid,
                role="metered",
                label=label,
            )
        for spec in phase_specs:
            eid = spec.entity_id
            state = self.hass.states.get(eid)
            label = state.name if state and state.name else eid
            profile_sources[eid] = ProfileSource(
                entity_id=eid,
                role=spec.role,
                label=label,
            )
        self.profile_sources = profile_sources

        role_map = {
            entity_id: channel.role
            for channel in self.model.gross_channels
            for entity_id in channel.entity_ids
        }
        self.aligner = SettleAligner(
            role_map,
            detectors=self.modes,
            stale_after=self._stale_after_s,
            profiled_interval=self._profiled_interval_s,
        )
        self._metered_values = {e: None for e in self.model.metered_entity_ids}

        tracked = (
            set(self.model.gross_entity_ids)
            | set(self.model.metered_entity_ids)
            | set(phase_eids)
            | set(session_plugs)
        )
        if tracked:
            tracked_list = list(tracked)
            self._unsub_trackers.append(
                async_track_state_change_event(
                    self.hass, tracked_list, self._on_state_changed
                )
            )
            self._unsub_trackers.append(
                async_track_state_report_event(
                    self.hass, tracked_list, self._on_state_reported
                )
            )

        # Seed metered entities
        for entity_id in self.model.metered_entity_ids:
            state = self.hass.states.get(entity_id)
            self._metered_values[entity_id] = parse_watts(state)

        # Seed session plugs (Plan 06-06): an unavailable plug stays None so
        # compute_snapshot reports it in metered_unavailable, never 0 W.
        for plug in session_plugs:
            self._session_values[plug] = parse_watts(self.hass.states.get(plug))

        # Seed gross entities
        now_ts = dt_util.utcnow().timestamp()
        for entity_id in self.model.gross_entity_ids:
            state = self.hass.states.get(entity_id)
            ts = state.last_reported_timestamp if state else now_ts
            self._handle_gross_sample(entity_id, state, ts, reported=False)

        # Seed newly activated channels in the buffer
        for ch in newly:
            if ch in self._channel_entities:
                entity_id = self._channel_entities[ch]
                state = self.hass.states.get(entity_id)
                if state is not None:
                    seed_ts = max(state.last_reported_timestamp, (now_ms + 1) / 1000)
                    self.writer.record(ch, parse_watts(state), seed_ts)

        self._republish_profiles()
        async_dispatcher_send(
            self.hass, f"{SIGNAL_PROFILE_SOURCES}_{self.entry.entry_id}"
        )
        if self._setup_done:
            self.async_schedule_bootstrap()

    def _reconcile_issues(self, active_checks: frozenset[str] | None = None) -> None:
        """Apply current diagnostic problems to the Repairs issue reconciler."""
        if self._structural_ready:
            problems_to_apply = list(self.model.problems)
        else:
            problems_to_apply = [p for p in self.model.problems if not p.structural]

        if active_checks is not None:
            self._cached_active_checks = active_checks

        for check in self._cached_active_checks:
            problems_to_apply.append(
                Problem(
                    issue_id=f"implausible_{check}",
                    kind=f"implausible_{check}",
                    severity=Severity.WARNING,
                    translation_key=f"implausible_{check}",
                    placeholders={},
                )
            )

        self.reconciler.apply(problems_to_apply)


    def _learning_review(self) -> Any:
        """Return the review store once learning exists (None before, Plan 06-06)."""
        learning = getattr(self, "learning", None)
        return getattr(learning, "review", None) if learning is not None else None

    def _device_lookup(self, device_uuid: str) -> dict[str, Any] | None:
        """Return the device record for a uuid, for session target resolution."""
        devices = getattr(self, "devices", None)
        return devices.record(device_uuid) if devices is not None else None

    @callback
    def _check_session_end_dates(self) -> None:
        """End the sessions whose planned end date has passed, on the existing tick."""
        due = self._session_planned_end_ts
        if due is None or self._session_end_task is not None:
            return
        parsed = dt_util.parse_datetime(due)
        if parsed is None or parsed > dt_util.utcnow():
            return
        self._session_end_task = self.hass.async_create_task(
            self._async_end_due_sessions()
        )

    async def _async_end_due_sessions(self) -> None:
        """Close every session whose end date passed and request one rebuild (D-11)."""
        try:
            review = self._learning_review()
            if review is None:
                return
            now = dt_util.utcnow()
            ended = 0
            for session in review.sessions:
                if session.get("end_ts") is not None:
                    continue
                planned = session.get("planned_end_ts")
                if not isinstance(planned, str):
                    continue
                parsed = dt_util.parse_datetime(planned)
                if parsed is None or parsed > now:
                    continue
                await review.async_end_session(session["session_id"], SESSION_END_DATE)
                _LOGGER.info(
                    "Plug session %s on %s ended on its end date",
                    session.get("session_id"),
                    session.get("plug_entity_id"),
                )
                ended += 1
            if ended:
                self.async_request_rebuild()
        except Exception:  # noqa: BLE001
            _LOGGER.exception("Ending due plug sessions failed")
        finally:
            self._session_end_task = None

    def _clear_trackers(self) -> None:
        """Cancel tracked state listeners."""
        for unsub in self._unsub_trackers:
            unsub()
        self._unsub_trackers.clear()

    @callback
    def _check_bootstrap_gate(self) -> None:
        """Release bootstrap gate if all current gross profilers have >= MIN_DELTAS."""
        if self._bootstrap_gate.is_set():
            return
        gross_eids = self.model.gross_entity_ids
        if not gross_eids:
            return
        if all(
            eid in self.profilers and self.profilers[eid].delta_count >= MIN_DELTAS
            for eid in gross_eids
        ):
            _LOGGER.info(
                "Bootstrap start gate released: all gross profilers reached %d deltas",
                MIN_DELTAS,
            )
            if self._unsub_bootstrap_timer is not None:
                self._unsub_bootstrap_timer()
                self._unsub_bootstrap_timer = None
            self._bootstrap_gate.set()

    @callback
    def async_schedule_bootstrap(self) -> None:
        """Schedule the recorder bootstrap background task if not already running."""
        if self._stopped:
            return
        self._setup_done = True
        if self._unsub_bootstrap_timer is None and not self._bootstrap_gate.is_set():
            @callback
            def _on_bootstrap_timeout(_now: Any) -> None:
                self._unsub_bootstrap_timer = None
                if not self._bootstrap_gate.is_set():
                    _LOGGER.info(
                        "Bootstrap start gate released after %d s timeout",
                        BOOTSTRAP_START_TIMEOUT_S,
                    )
                    self._bootstrap_gate.set()

            self._unsub_bootstrap_timer = async_call_later(
                self.hass,
                BOOTSTRAP_START_TIMEOUT_S,
                _on_bootstrap_timeout,
            )

        if self.bootstrap_task is None or self.bootstrap_task.done():
            self.bootstrap_task = self.entry.async_create_background_task(
                self.hass,
                BootstrapRunner(self.hass, self).async_run(),
                "powerguesser_bootstrap",
            )

    async def async_stop(self) -> None:
        """Unsubscribe all event listeners and clear active Repairs issues."""
        self._stopped = True
        self._clear_trackers()
        if self.bootstrap_task is not None and not self.bootstrap_task.done():
            self.bootstrap_task.cancel()
        if self._unsub_bootstrap_timer is not None:
            self._unsub_bootstrap_timer()
            self._unsub_bootstrap_timer = None
        if self._unsub_tick is not None:
            self._unsub_tick()
            self._unsub_tick = None
        if self._unsub_grace is not None:
            self._unsub_grace()
            self._unsub_grace = None
        if self._unsub_at_started is not None:
            self._unsub_at_started()
            self._unsub_at_started = None
        if self._unsub_er is not None:
            self._unsub_er()
            self._unsub_er = None
        self.reconciler.clear()

    async def _async_follow_rename(
        self, old_entity_id: str, new_entity_id: str
    ) -> None:
        """Follow an entity registry rename in buffer and detectors before rebuilding."""
        if self._stopped or self.entry.state is not ConfigEntryState.LOADED:
            return
        await self.hass.async_add_executor_job(
            self.writer.db.rename_entity, old_entity_id, new_entity_id
        )
        if old_entity_id in self.modes:
            self.modes[new_entity_id] = self.modes.pop(old_entity_id)
        if old_entity_id in self.profilers:
            self.profilers[new_entity_id] = self.profilers.pop(old_entity_id)

        phase_keys = [
            k
            for k, v in self.entry.options.items()
            if v == old_entity_id and k in PHASE_OPTION_ROLES
        ]
        if (
            phase_keys
            and not self._stopped
            and self.entry.state is ConfigEntryState.LOADED
        ):
            new_options = dict(self.entry.options)
            for k in phase_keys:
                new_options[k] = new_entity_id
            self.hass.config_entries.async_update_entry(
                self.entry, options=new_options
            )
            return

        self.async_request_rebuild()

    @callback
    def _on_tick(self, _now: Any) -> None:
        """Periodic staleness and on-change evaluation tick."""
        now_ts = dt_util.utcnow().timestamp()
        self._check_session_end_dates()
        self.writer.mark_stale(round(now_ts * 1000))
        if self.aligner.on_tick(now_ts):
            self._publish_current_state(now_ts)
            return
        # No aligner publish: still drain due detector grid points so energy
        # accrues and ON/OFF changes surface at grid cadence (Plan 04-04).
        devices = getattr(self, "devices", None)
        if devices is not None and self.coordinator.data is not None:
            updated = devices.on_tick(now_ts, self.coordinator.data)
            if updated is not None:
                self.coordinator.async_set_updated_data(updated)

    @callback
    def _phase_values(self) -> tuple[float, float, float] | None:
        """Read the three mapped phase powers, negated when configured.

        Returns None unless every mapped phase entity reports a numeric
        value. Only called from _publish_current_state (aligner publishes),
        so the 10 s tick path does no state lookups and the per-tick timing
        measures the detector alone.
        """
        invert = bool(self.entry.options.get(CONF_INVERT_PHASES, False))
        vals: list[float] = []
        for key in (CONF_PHASE_L1, CONF_PHASE_L2, CONF_PHASE_L3):
            entity_id = self.entry.options.get(key)
            if not entity_id:
                return None
            watts = parse_watts(self.hass.states.get(entity_id))
            if watts is None:
                return None
            vals.append(-watts if invert else watts)
        return (vals[0], vals[1], vals[2])

    @callback
    def _publish_current_state(self, ts: float) -> None:
        """Produce a snapshot from aligner status and publish to coordinator."""
        align_status = self.aligner.status(ts)
        emit_ts = align_status.emit_ts or ts
        metered_values: Mapping[str, float | None] = self._metered_values
        if self._session_metered_plugs:
            # D-25: only a candidate-target session plug is subtracted live; a
            # device-target plug is recorded but never merged here.
            metered_values = {
                **self._metered_values,
                **{
                    plug: self._session_values.get(plug)
                    for plug in self._session_metered_plugs
                },
            }
        snapshot = compute_snapshot(
            self.model,
            align_status.values,
            metered_values,
            emit_ts,
            accumulator=self.over_attribution,
            align_status=align_status,
        )

        if snapshot.available and snapshot.gross_w is not None:
            failing = evaluate_plausibility(snapshot.gross_w, snapshot.metered_w)
            plausibility = "ok" if not failing else ",".join(failing)
            active_checks = self.self_check.observe(
                emit_ts,
                failing,
                gross_w=snapshot.gross_w,
                metered_w=snapshot.metered_w,
            )
            snapshot = dataclasses.replace(snapshot, plausibility=plausibility)
            self._reconcile_issues(active_checks=active_checks)

        devices = getattr(self, "devices", None)
        if devices is not None:
            snapshot = devices.apply_snapshot(
                snapshot, emit_ts, self._phase_values()
            )

        if self._last_available is not None and self._last_available != snapshot.available:

            if not snapshot.available:
                _LOGGER.info(
                    "PowerGuesser became unavailable due to blocking sources: %s",
                    list(align_status.blocking),
                )
            else:
                _LOGGER.info("PowerGuesser became available")
        self._last_available = snapshot.available
        self.coordinator.async_set_updated_data(snapshot)

    def _stale_after_s(self, entity_id: str) -> float:
        """Return dynamic staleness timeout for an entity (D-15)."""
        if entity_id in self.profilers:
            return self.profilers[entity_id].stale_after_s()
        return STALE_AFTER_S

    def _profiled_interval_s(self, entity_id: str) -> float | None:
        """Return the published median sample interval of an entity, or None if not profiled yet."""
        profiler = self.profilers.get(entity_id)
        return profiler.median_s if profiler is not None else None

    def _build_profile_snapshot(self) -> ProfileSnapshot:
        """Construct the current ProfileSnapshot across all tracked sources."""
        sources = {
            eid: profiler.snapshot(eid)
            for eid, profiler in self.profilers.items()
        }
        gross_profiles = [
            sources[eid]
            for eid in self.model.gross_entity_ids
            if eid in sources
        ]
        eff_s, assumed = effective_interval(gross_profiles)
        enabled, disabled = gate(eff_s, assumed)
        return ProfileSnapshot(
            sources=sources,
            effective_s=eff_s,
            assumed=assumed,
            enabled=enabled,
            disabled=disabled,
        )

    @callback
    def _profile_store_data(self) -> dict[str, Any]:
        """Return serialized interval profile data for Store."""
        sources: dict[str, Any] = {}
        for eid, profiler in self.profilers.items():
            if profiler.median_s is not None:
                mode_str = (
                    profiler.mode.value
                    if hasattr(profiler.mode, "value")
                    else str(profiler.mode)
                )
                sources[eid] = {
                    "median_s": profiler.median_s,
                    "p90_s": profiler.p90_s,
                    "mode": mode_str,
                }
        return {"sources": sources}

    @callback
    def _republish_profiles(self) -> None:
        """Publish updated profiles to profile_coordinator when publish key changes."""
        snapshot = self._build_profile_snapshot()
        source_keys = tuple(
            (
                eid,
                prof.mode,
                round(prof.median_s) if prof.median_s is not None else None,
                prof.restored,
            )
            for eid, prof in sorted(snapshot.sources.items())
        )
        publish_key = (
            source_keys,
            round(snapshot.effective_s),
            snapshot.assumed,
            snapshot.enabled,
        )
        if publish_key != self._last_profile_publish_key:
            self._last_profile_publish_key = publish_key
            self.profile_coordinator.async_set_updated_data(snapshot)
            self._profile_store.async_delay_save(
                self._profile_store_data, PROFILE_SAVE_DELAY_S
            )

    @callback
    def _on_state_changed(self, event: Event[EventStateChangedData]) -> None:
        """Handle state change event for a tracked entity."""
        entity_id = event.data["entity_id"]
        new_state = event.data.get("new_state")
        ts = new_state.last_reported_timestamp if new_state else dt_util.utcnow().timestamp()

        _LOGGER.debug("Tracked entity state changed: %s", entity_id)
        val = parse_watts(new_state)
        for ch in self._entity_channels.get(entity_id, ()):
            self.writer.record(ch, val, ts)

        if entity_id in self.profilers:
            if self.profilers[entity_id].observe(ts, reported=False, w=val):
                self._republish_profiles()
            self._check_bootstrap_gate()

        if entity_id in self.aligner.entity_roles:
            self._handle_gross_sample(entity_id, new_state, ts, reported=False)
        elif entity_id in self._metered_values:
            self._metered_values[entity_id] = val
        elif entity_id in self._session_values:
            self._session_values[entity_id] = val

    @callback
    def _on_state_reported(self, event: Event[dict[str, Any]]) -> None:
        """Handle state report (unchanged re-report) event for a tracked entity."""
        entity_id = event.data["entity_id"]
        last_reported_dt = event.data.get("last_reported")
        ts = last_reported_dt.timestamp() if last_reported_dt else dt_util.utcnow().timestamp()

        if entity_id in self.modes and self.modes[entity_id].observe_report(ts):
            _LOGGER.info(
                "Tracked entity %s is now classified as a re-reporting source; will be marked stale when silent",
                entity_id,
            )
            self._republish_profiles()

        state = self.hass.states.get(entity_id)
        _LOGGER.debug("Tracked entity state reported: %s", entity_id)
        val = parse_watts(state)
        for ch in self._entity_channels.get(entity_id, ()):
            self.writer.record(ch, val, ts)

        if entity_id in self.profilers:
            if self.profilers[entity_id].observe(ts, reported=True, w=val):
                self._republish_profiles()
            self._check_bootstrap_gate()

        if entity_id in self.aligner.entity_roles:
            self._handle_gross_sample(entity_id, state, ts, reported=True)
        elif entity_id in self._metered_values:
            self._metered_values[entity_id] = val
        elif entity_id in self._session_values:
            self._session_values[entity_id] = val

    @callback
    def _handle_gross_sample(
        self,
        entity_id: str,
        state: State | None,
        ts: float,
        *,
        reported: bool,
    ) -> None:
        """Process a gross entity sample through the aligner and coordinator."""
        val = parse_watts(state)
        settled = self.aligner.on_sample(entity_id, val, ts, reported=reported)
        if settled:
            self._publish_current_state(ts)

