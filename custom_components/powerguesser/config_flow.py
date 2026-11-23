"""Config flow for PowerGuesser integration."""

import asyncio
from datetime import datetime
import logging
from typing import Any

import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    OptionsFlow,
)
from homeassistant.core import CoreState, callback
from homeassistant.helpers import entity_registry as er, issue_registry as ir, selector
from homeassistant.util import dt as dt_util

from .const import (
    CONF_INVERT_PHASES,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_CLEAR_KEY,
    CONF_LLM_MODEL,
    CONF_LLM_PROBE,
    CONF_MAX_TOKENS,
    CONF_MONTH_COST_LIMIT,
    CONF_MONTH_INPUT_TOKENS,
    CONF_MONTH_OUTPUT_TOKENS,
    CONF_PHASE_L1,
    CONF_PHASE_L2,
    CONF_PHASE_L3,
    CONF_PRICE_INPUT,
    CONF_PRICE_OUTPUT,
    CONF_RUN_INPUT_TOKENS,
    CONF_RUN_OUTPUT_TOKENS,
    CONF_SCHEDULE,
    CONF_SCHEDULE_TIME,
    CONF_SCHEDULE_WEEKDAY,
    CONF_SHARE_NAMES,
    CONF_TRUNCATION_GUARD,
    CONF_VALIDATE_AFTER_LEARNING,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MONTH_INPUT_TOKENS,
    DEFAULT_MONTH_OUTPUT_TOKENS,
    DEFAULT_RUN_INPUT_TOKENS,
    DEFAULT_RUN_OUTPUT_TOKENS,
    DEFAULT_SCHEDULE,
    DEFAULT_SCHEDULE_TIME,
    DEFAULT_SCHEDULE_WEEKDAY,
    DEFAULT_VALIDATE_AFTER_LEARNING,
    DOMAIN,
    ISSUE_LLM_AUTH,
    SUBENTRY_TYPE_DEVICE,
    WEEKDAYS,
)
from .energy_prefs import (
    async_get_prefs,
    collect_registry_facts,
    device_consumption_power_entities,
)
from .engine.compute import channel_value_w
from .engine.model import Role
from .engine.phases import PhaseSumResult, check_phase_sum
from .engine.prefs import ABORT_ORDER, build_source_model
from .hub import parse_watts
from .learning import CandidateGone
from .review_store import (
    SESSION_END_USER,
    normalise_name,
    resolve_session_target,
)
from .llm_client import (
    LlmAuthError,
    LlmClient,
    LlmConnectError,
    LlmUnusableModel,
    endpoint_origin,
    normalise_base_url,
)

_LOGGER = logging.getLogger(__name__)


def _escape_markdown(text: str) -> str:
    """Backslash-escape markdown-significant characters."""
    out: list[str] = []
    for ch in text:
        if ch in "\\`*_{}[]()#+-.!<>|~":
            out.append(f"\\{ch}")
        else:
            out.append(ch)
    return "".join(out)


def _as_iso_utc(value: Any) -> str | None:
    """Normalise a DateTimeSelector value to an ISO 8601 UTC string, or None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = dt_util.parse_datetime(str(value))
    if parsed is None:
        return None
    return dt_util.as_utc(parsed).isoformat()


class PowerGuesserConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for PowerGuesser."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial setup step."""
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        prefs = await async_get_prefs(self.hass)
        facts = collect_registry_facts(self.hass, prefs)
        model = build_source_model(prefs, facts=facts)

        for kind in ABORT_ORDER:
            if kind == "source_not_found" and self.hass.state is not CoreState.running:
                continue
            matching = [p for p in model.problems if p.kind == kind]
            if matching:
                prob = matching[0]
                return self.async_abort(
                    reason=kind,
                    description_placeholders=dict(prob.placeholders),
                )

        if user_input is not None:
            return self.async_create_entry(title="PowerGuesser", data={})

        gross_labels = ", ".join(ch.label for ch in model.gross_channels)
        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({}),
            description_placeholders={
                "gross_sources": gross_labels,
                "metered_count": str(len(model.metered_roots)),
            },
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Create the options flow."""
        return PowerGuesserOptionsFlow()

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this handler."""
        return {SUBENTRY_TYPE_DEVICE: DeviceSubentryFlow}


class DeviceSubentryFlow(ConfigSubentryFlow):
    """Subentry flow for confirmed devices (RESEARCH Pattern 7).

    Devices are created only by confirming a candidate under Configure ->
    Review candidates (REV-02), so the "Add" button's user step always aborts
    with the use_review reason. Rename (the Settings "Configure" action)
    validates the name with normalise_name and updates the subentry via
    async_update_and_abort — never the reload variant, which raises
    ValueError when the entry has update listeners (Pitfall 1).
    """

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Refuse direct creation: devices come from confirming a candidate."""
        return self.async_abort(reason="use_review")

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Rename a device from Settings (OUT-05: identity never changes)."""
        entry = self._get_entry()
        sub = self._get_reconfigure_subentry()

        errors: dict[str, str] = {}
        if user_input is not None:
            raw_name = str(user_input.get("name", ""))
            try:
                name = normalise_name(raw_name)
            except ValueError:
                errors["base"] = "name_invalid"
            else:
                # async_update_and_abort (never the reload variant): the
                # update listener's reconcile mirrors the name into the cold
                # record, the review record and the device registry; the
                # sensors keep their entity IDs and history. The reason is
                # passed explicitly so our translated "The device was
                # renamed." message is used instead of HA's default.
                return self.async_update_and_abort(
                    entry,
                    sub,
                    title=name,
                    data_updates={"name": name},
                    reason="reconfigure_successful",
                )

        schema = vol.Schema({vol.Required("name"): selector.TextSelector()})
        return self.async_show_form(
            step_id="reconfigure",
            data_schema=self.add_suggested_values_to_schema(
                schema, {"name": sub.title}
            ),
            errors=errors,
            description_placeholders={"name": _escape_markdown(sub.title)},
        )


class PowerGuesserOptionsFlow(OptionsFlow):
    """Handle options flow for PowerGuesser."""

    def __init__(self) -> None:
        """Initialize options flow."""
        self._pending_mapping: dict[str, str] = {}
        self._phase_sum: float = 0.0
        self._grid_w: float = 0.0
        self._phase_check_result: PhaseSumResult = PhaseSumResult.OK
        self._pending_llm: dict[str, Any] = {}
        self._probe_task: asyncio.Task[dict[str, Any]] | None = None
        self._llm_errors: dict[str, str] = {}
        self._candidate_id: str | None = None
        self._device_uuid: str | None = None
        self._review_errors: dict[str, str] = {}

    def _get_grid_device_sensors(self) -> str:
        """Find power sensors on the grid meter's device as suggestions."""
        hub = getattr(self.config_entry, "runtime_data", None)
        if hub is None or getattr(hub, "model", None) is None:
            return "none found"
        ent_reg = er.async_get(self.hass)
        grid_channels = [
            ch for ch in hub.model.gross_channels if ch.role == Role.GRID
        ]
        for ch in grid_channels:
            for eid in ch.entity_ids:
                entry = ent_reg.async_get(eid)
                if entry and entry.device_id:
                    device_entries = er.async_entries_for_device(
                        ent_reg, entry.device_id
                    )
                    candidates = [
                        e.entity_id
                        for e in device_entries
                        if e.domain == "sensor"
                        and (
                            e.device_class == "power"
                            or e.original_device_class == "power"
                        )
                        and e.platform != DOMAIN
                        and e.entity_id != eid
                    ]
                    if candidates:
                        return ", ".join(sorted(candidates))
        return "none found"

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options menu."""
        if getattr(self.config_entry, "runtime_data", None) is None:
            return self.async_abort(reason="not_loaded")

        return self.async_show_menu(
            step_id="init",
            menu_options=["phases", "llm_endpoint", "learning", "review", "plug_sessions"],
        )

    # -- Plug sessions (REV-04, Plan 06-06 D-09..D-12, D-15) ----------------

    def _review_store(self) -> Any:
        """Return the review store, or None while PowerGuesser is not loaded."""
        hub = getattr(self.config_entry, "runtime_data", None)
        if hub is None or getattr(hub, "learning", None) is None:
            return None
        return hub.learning.review

    def _session_targets(self) -> list[selector.SelectOptionDict]:
        """Target options: visible candidates (candidate_<sig>) and confirmed devices."""
        hub = self.config_entry.runtime_data
        options: list[selector.SelectOptionDict] = []
        for cand in hub.learning.visible_candidates():
            sig_id = str(cand.get("signature_id") or "")
            if not sig_id:
                continue
            options.append(
                selector.SelectOptionDict(
                    value=f"candidate_{sig_id}",
                    label=hub.learning.candidate_label(cand),
                )
            )
        for entry in hub.devices.confirmed_entries():
            device_uuid = str(entry.get("device_uuid") or "")
            if not device_uuid:
                continue
            options.append(
                selector.SelectOptionDict(
                    value=f"device_{device_uuid}",
                    label=hub.learning.candidate_label(entry),
                )
            )
        return options

    def _parse_session_target(self, value: str) -> tuple[str | None, str]:
        """Resolve a ``candidate_<sig>`` / ``device_<uuid>`` option to its target.

        Returns (kind, id) or (None, "") when the target is gone (T-06-13).
        """
        hub = self.config_entry.runtime_data
        if value.startswith("candidate_"):
            sig_id = value[len("candidate_") :]
            if any(
                c.get("signature_id") == sig_id
                for c in hub.learning.visible_candidates()
            ):
                return ("candidate", sig_id)
            return (None, "")
        if value.startswith("device_"):
            device_uuid = value[len("device_") :]
            rec = hub.devices.record(device_uuid)
            if rec is not None and rec.get("status") == "confirmed":
                return ("device", device_uuid)
            return (None, "")
        return (None, "")

    def _target_label(self, resolved: tuple[str, str]) -> str:
        """Display label of a resolved session target, derived at read time (D-09)."""
        hub = self.config_entry.runtime_data
        kind, ident = resolved
        if kind == "candidate":
            cand = next(
                (
                    c
                    for c in hub.learning.visible_candidates()
                    if c.get("signature_id") == ident
                ),
                None,
            )
            if cand is not None:
                return hub.learning.candidate_label(cand)
            dec = hub.learning.review.decisions.get(ident) or {}
            name = dec.get("name") or dec.get("label")
            return str(name) if name else ident
        entry = next(
            (
                e
                for e in hub.devices.confirmed_entries()
                if e.get("device_uuid") == ident
            ),
            None,
        )
        if entry is not None:
            return hub.learning.candidate_label(entry)
        rec = hub.devices.record(ident)
        if rec is not None:
            return str(rec.get("name") or ident)
        return ident

    def _session_lines(self, review: Any) -> list[str]:
        """One markdown line per stored session, or the empty-state text."""
        hub = self.config_entry.runtime_data
        decisions = review.decisions
        lines: list[str] = []
        for session in review.sessions:
            plug = str(session.get("plug_entity_id") or "")
            resolved = resolve_session_target(
                session, decisions, hub.devices.record
            )
            if resolved is None:
                target = "a target that no longer exists"
            else:
                target = self._target_label(resolved)
            started = str(session.get("start_ts") or "")[:10]
            end_ts = session.get("end_ts")
            if end_ts:
                state = (
                    f"ended {str(end_ts)[:10]} "
                    f"({session.get('end_reason') or 'ended'})"
                )
            else:
                state = "active"
            lines.append(f"- {plug} — {target} — started {started} — {state}")
        return lines

    async def async_step_plug_sessions(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Plug-session menu: start, list (and end while one is active) (D-15)."""
        review = self._review_store()
        if review is None:
            return self.async_abort(reason="not_loaded")
        menu_opts = ["session_start", "session_list"]
        if review.active_sessions(dt_util.utcnow()):
            menu_opts.append("session_end")
        return self.async_show_menu(step_id="plug_sessions", menu_options=menu_opts)

    async def _session_plug_error(self, plug: str) -> str | None:
        """D-10 plug validation: own platform, a source, or an active session's plug."""
        hub = self.config_entry.runtime_data
        ent_reg = er.async_get(self.hass)
        reg_entry = ent_reg.async_get(plug)
        if (
            reg_entry is not None and reg_entry.platform == DOMAIN
        ) or plug.startswith(f"sensor.{DOMAIN}_"):
            return "plug_self_reference"

        prefs = await async_get_prefs(self.hass)
        facts = collect_registry_facts(self.hass, prefs)
        phase_eids = {
            self.config_entry.options.get(key)
            for key in (CONF_PHASE_L1, CONF_PHASE_L2, CONF_PHASE_L3)
            if self.config_entry.options.get(key)
        }
        sources = (
            set(hub.model.gross_entity_ids)
            | set(hub.model.referenced_entity_ids)
            | phase_eids
            | set(device_consumption_power_entities(prefs, facts))
        )
        if plug in sources:
            return "plug_is_source"

        review = self._review_store()
        if review is not None and any(
            s.get("plug_entity_id") == plug
            for s in review.active_sessions(dt_util.utcnow())
        ):
            return "plug_in_session"
        return None

    async def async_step_session_start(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Start a plug session for a candidate or a confirmed device (D-09..D-12)."""
        hub = getattr(self.config_entry, "runtime_data", None)
        review = self._review_store()
        if hub is None or review is None:
            return self.async_abort(reason="not_loaded")

        errors: dict[str, str] = {}
        if user_input is not None:
            plug = str(user_input.get("plug") or "")
            target_kind, target_id = self._parse_session_target(
                str(user_input.get("target") or "")
            )
            end_raw = user_input.get("end_date")
            planned_end = _as_iso_utc(end_raw)
            plug_error = await self._session_plug_error(plug)
            if plug_error is not None:
                errors["base"] = plug_error
            elif target_kind is None:
                errors["base"] = "target_gone"
            elif end_raw and (
                planned_end is None
                or dt_util.parse_datetime(planned_end) <= dt_util.utcnow()
            ):
                errors["base"] = "end_date_past"
            else:
                await review.async_start_session(
                    plug,
                    target_kind,
                    target_id,
                    planned_end_ts=planned_end,
                )
                hub.async_request_rebuild()
                return await self.async_step_plug_sessions()

        schema = vol.Schema(
            {
                vol.Required("plug"): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        filter={"domain": "sensor", "device_class": "power"}
                    )
                ),
                vol.Required("target"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=self._session_targets(),
                        mode=selector.SelectSelectorMode.LIST,
                    )
                ),
                vol.Optional("end_date"): selector.DateTimeSelector(),
            }
        )
        return self.async_show_form(
            step_id="session_start",
            data_schema=schema,
            errors=errors,
        )

    def _session_option_label(self, session: dict[str, Any]) -> str:
        """One-line label for an active session in the End selector."""
        hub = self.config_entry.runtime_data
        resolved = resolve_session_target(
            session, hub.learning.review.decisions, hub.devices.record
        )
        target = (
            "a target that no longer exists"
            if resolved is None
            else self._target_label(resolved)
        )
        started = str(session.get("start_ts") or "")[:10]
        return f"{session.get('plug_entity_id')} — {target} — started {started}"

    async def async_step_session_end(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """End an active plug session by hand (D-11)."""
        review = self._review_store()
        if review is None:
            return self.async_abort(reason="not_loaded")

        actives = review.active_sessions(dt_util.utcnow())
        if not actives:
            return await self.async_step_plug_sessions()

        errors: dict[str, str] = {}
        if user_input is not None:
            session_id = str(user_input.get("session") or "")
            if not any(s.get("session_id") == session_id for s in actives):
                errors["base"] = "target_gone"
            else:
                await review.async_end_session(session_id, SESSION_END_USER)
                self.config_entry.runtime_data.async_request_rebuild()
                return await self.async_step_plug_sessions()

        options = [
            selector.SelectOptionDict(
                value=str(s.get("session_id")),
                label=self._session_option_label(s),
            )
            for s in actives
        ]
        schema = vol.Schema(
            {
                vol.Required("session"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=options,
                        mode=selector.SelectSelectorMode.LIST,
                    )
                ),
            }
        )
        return self.async_show_form(
            step_id="session_end",
            data_schema=schema,
            errors=errors,
        )

    async def async_step_session_list(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """List the stored plug sessions (D-15)."""
        review = self._review_store()
        if review is None:
            return self.async_abort(reason="not_loaded")
        lines = self._session_lines(review)
        text = "\n".join(lines) if lines else "There are no plug sessions yet."
        return self.async_show_form(
            step_id="session_list",
            data_schema=vol.Schema({}),
            description_placeholders={"sessions": _escape_markdown(text)},
        )

    async def async_step_phases(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage phase mapping options."""
        errors: dict[str, str] = {}
        hub = getattr(self.config_entry, "runtime_data", None)

        base_options = {
            k: v
            for k, v in self.config_entry.options.items()
            if k not in (CONF_PHASE_L1, CONF_PHASE_L2, CONF_PHASE_L3, CONF_INVERT_PHASES)
        }

        if user_input is not None:
            mapped = {
                k: v
                for k, v in user_input.items()
                if k in (CONF_PHASE_L1, CONF_PHASE_L2, CONF_PHASE_L3) and v
            }

            if not mapped:
                return self.async_create_entry(data=base_options)

            # Duplicate entity check across fields
            values = list(mapped.values())
            if len(values) != len(set(values)):
                errors["base"] = "duplicate_phase"

            # Own platform check (PowerGuesser entity cannot be mapped)
            ent_reg = er.async_get(self.hass)
            for k, eid in mapped.items():
                reg_entry = ent_reg.async_get(eid)
                if (
                    reg_entry and reg_entry.platform == DOMAIN
                ) or eid.startswith(f"sensor.{DOMAIN}_"):
                    errors[k] = "phase_self_reference"

            # Gross or metered entity check
            gross_or_metered: set[str] = set()
            if hub and getattr(hub, "model", None):
                gross_or_metered = set(hub.model.gross_entity_ids) | set(
                    hub.model.metered_entity_ids
                )
            for k, eid in mapped.items():
                if eid in gross_or_metered and k not in errors:
                    errors[k] = "phase_is_source"

            if not errors:
                grid_w: float | None = None
                if hub and getattr(hub, "model", None):
                    grid_channels = [
                        ch
                        for ch in hub.model.gross_channels
                        if ch.role == Role.GRID
                    ]
                    if grid_channels:
                        grid_states = {
                            eid: parse_watts(self.hass.states.get(eid))
                            for ch in grid_channels
                            for eid in ch.entity_ids
                        }
                        ch_vals = [
                            channel_value_w(ch, grid_states)
                            for ch in grid_channels
                        ]
                        if all(v is not None for v in ch_vals):
                            grid_w = sum(v for v in ch_vals if v is not None)

                phase_values = [
                    parse_watts(self.hass.states.get(eid))
                    for eid in mapped.values()
                ]
                result = check_phase_sum(phase_values, grid_w)

                if result == PhaseSumResult.OK:
                    data = {**base_options, **mapped}
                    # Carry the stored invert flag forward on the OK path too,
                    # like the SKIPPED path: a reading that happens to match
                    # without inversion is not evidence the inversion is
                    # unwanted, and dropping it flips the sign of every
                    # recorded per-phase sample (WR-04).
                    if self.config_entry.options.get(CONF_INVERT_PHASES):
                        data[CONF_INVERT_PHASES] = True
                    return self.async_create_entry(data=data)

                if result == PhaseSumResult.SKIPPED:
                    data = {**base_options, **mapped}
                    if self.config_entry.options.get(CONF_INVERT_PHASES):
                        data[CONF_INVERT_PHASES] = True
                    return self.async_create_entry(data=data)

                # INVERTED or MISMATCH
                self._pending_mapping = dict(mapped)
                self._phase_sum = sum(
                    v for v in phase_values if v is not None
                )
                self._grid_w = grid_w if grid_w is not None else 0.0
                self._phase_check_result = result
                return await self.async_step_phase_check()

        schema = vol.Schema(
            {
                vol.Optional(CONF_PHASE_L1): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        filter={"domain": "sensor", "device_class": "power"}
                    )
                ),
                vol.Optional(CONF_PHASE_L2): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        filter={"domain": "sensor", "device_class": "power"}
                    )
                ),
                vol.Optional(CONF_PHASE_L3): selector.EntitySelector(
                    selector.EntitySelectorConfig(
                        filter={"domain": "sensor", "device_class": "power"}
                    )
                ),
            }
        )
        return self.async_show_form(
            step_id="phases",
            data_schema=self.add_suggested_values_to_schema(
                schema, self.config_entry.options
            ),
            errors=errors,
            description_placeholders={
                "grid_device_sensors": self._get_grid_device_sensors(),
            },
        )

    async def async_step_phase_check(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle non-blocking phase check confirmation step."""
        base_options = {
            k: v
            for k, v in self.config_entry.options.items()
            if k not in (CONF_PHASE_L1, CONF_PHASE_L2, CONF_PHASE_L3, CONF_INVERT_PHASES)
        }

        if user_input is not None:
            data = {**base_options, **self._pending_mapping}
            if user_input.get(CONF_INVERT_PHASES):
                data[CONF_INVERT_PHASES] = True
            return self.async_create_entry(data=data)

        is_inverted = self._phase_check_result == PhaseSumResult.INVERTED
        schema = vol.Schema(
            {
                vol.Required(CONF_INVERT_PHASES, default=is_inverted): bool,
            }
        )
        return self.async_show_form(
            step_id="phase_check",
            data_schema=schema,
            description_placeholders={
                "phase_sum": str(round(self._phase_sum)),
                "grid": str(round(self._grid_w)),
                "result": str(self._phase_check_result.value),
            },
        )

    async def async_step_llm_endpoint(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage LLM endpoint configuration."""
        if self._probe_task is not None:
            if not self._probe_task.done():
                return self.async_show_progress(
                    step_id="llm_endpoint",
                    progress_action="probe",
                    progress_task=self._probe_task,
                )
            return self.async_show_progress_done(next_step_id="llm_probe_done")

        errors = dict(self._llm_errors)
        self._llm_errors.clear()

        if user_input is not None:
            raw_url = str(user_input.get(CONF_LLM_BASE_URL) or "").strip()
            if not raw_url:
                # Clearing base URL removes LLM configuration from entry.data
                new_data = {
                    k: v
                    for k, v in self.config_entry.data.items()
                    if k
                    not in (
                        CONF_LLM_BASE_URL,
                        CONF_LLM_API_KEY,
                        CONF_LLM_MODEL,
                        CONF_LLM_PROBE,
                    )
                }
                self.hass.config_entries.async_update_entry(
                    self.config_entry, data=new_data
                )
                return self.async_create_entry(data=dict(self.config_entry.options))

            base_url = ""
            try:
                base_url = normalise_base_url(raw_url)
            except ValueError:
                errors["base"] = "invalid_url"

            model = str(user_input.get(CONF_LLM_MODEL) or "").strip()
            if not model and "base" not in errors:
                errors["base"] = "model_required"

            if not errors:
                key = str(user_input.get(CONF_LLM_API_KEY) or "").strip()
                clear_key = bool(user_input.get(CONF_LLM_CLEAR_KEY, False))
                if not key and not clear_key:
                    # Reuse the stored key only for the same scheme, host and port, so it
                    # never goes to another service on the same host or over plain http.
                    stored_url = self.config_entry.data.get(CONF_LLM_BASE_URL)
                    if stored_url:
                        try:
                            if endpoint_origin(base_url) == endpoint_origin(stored_url):
                                key = self.config_entry.data.get(CONF_LLM_API_KEY, "")
                        except Exception:
                            key = ""

                self._pending_llm = {
                    CONF_LLM_BASE_URL: base_url,
                    CONF_LLM_API_KEY: key,
                    CONF_LLM_MODEL: model,
                }

                client = LlmClient(self.hass, base_url, key)
                self._probe_task = self.hass.async_create_task(client.async_probe(model))
                return self.async_show_progress(
                    step_id="llm_endpoint",
                    progress_action="probe",
                    progress_task=self._probe_task,
                )

        suggested = {
            CONF_LLM_BASE_URL: self._pending_llm.get(
                CONF_LLM_BASE_URL, self.config_entry.data.get(CONF_LLM_BASE_URL)
            ),
            CONF_LLM_MODEL: self._pending_llm.get(
                CONF_LLM_MODEL, self.config_entry.data.get(CONF_LLM_MODEL)
            ),
        }

        schema = vol.Schema(
            {
                vol.Optional(CONF_LLM_BASE_URL): selector.TextSelector(
                    selector.TextSelectorConfig(type=selector.TextSelectorType.URL)
                ),
                vol.Optional(CONF_LLM_API_KEY): selector.TextSelector(
                    selector.TextSelectorConfig(
                        type=selector.TextSelectorType.PASSWORD
                    )
                ),
                vol.Optional(CONF_LLM_MODEL): selector.TextSelector(),
                vol.Optional(CONF_LLM_CLEAR_KEY, default=False): selector.BooleanSelector(),
            }
        )

        return self.async_show_form(
            step_id="llm_endpoint",
            data_schema=self.add_suggested_values_to_schema(schema, suggested),
            errors=errors,
        )

    async def async_step_llm_probe_done(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle completion of the LLM endpoint capability probe."""
        profile: dict[str, Any] | None = None
        task = self._probe_task
        self._probe_task = None
        if task is not None:
            try:
                profile = task.result()
            except LlmAuthError:
                self._llm_errors["base"] = "invalid_auth"
            except LlmConnectError:
                self._llm_errors["base"] = "cannot_connect"
            except LlmUnusableModel:
                self._llm_errors["base"] = "model_unusable"
            except Exception as err:
                _LOGGER.debug("Unexpected probe task error: %s", type(err).__name__)
                self._llm_errors["base"] = "cannot_connect"

        if self._llm_errors or profile is None:
            return await self.async_step_llm_endpoint()

        ir.async_delete_issue(self.hass, DOMAIN, ISSUE_LLM_AUTH)

        new_data = {
            **self.config_entry.data,
            CONF_LLM_BASE_URL: self._pending_llm[CONF_LLM_BASE_URL],
            CONF_LLM_API_KEY: self._pending_llm[CONF_LLM_API_KEY],
            CONF_LLM_MODEL: self._pending_llm[CONF_LLM_MODEL],
            CONF_LLM_PROBE: profile,
        }
        self.hass.config_entries.async_update_entry(self.config_entry, data=new_data)
        return self.async_create_entry(data=dict(self.config_entry.options))

    async def async_step_learning(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage learning options."""
        errors: dict[str, str] = {}
        curr_opts = self.config_entry.options

        suggested: dict[str, Any] = {
            CONF_SHARE_NAMES: curr_opts.get(CONF_SHARE_NAMES, False),
            CONF_MAX_TOKENS: curr_opts.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
            CONF_RUN_INPUT_TOKENS: curr_opts.get(
                CONF_RUN_INPUT_TOKENS, DEFAULT_RUN_INPUT_TOKENS
            ),
            CONF_RUN_OUTPUT_TOKENS: curr_opts.get(
                CONF_RUN_OUTPUT_TOKENS, DEFAULT_RUN_OUTPUT_TOKENS
            ),
            CONF_MONTH_INPUT_TOKENS: curr_opts.get(
                CONF_MONTH_INPUT_TOKENS, DEFAULT_MONTH_INPUT_TOKENS
            ),
            CONF_MONTH_OUTPUT_TOKENS: curr_opts.get(
                CONF_MONTH_OUTPUT_TOKENS, DEFAULT_MONTH_OUTPUT_TOKENS
            ),
            CONF_SCHEDULE: curr_opts.get(CONF_SCHEDULE, DEFAULT_SCHEDULE),
            CONF_SCHEDULE_WEEKDAY: curr_opts.get(
                CONF_SCHEDULE_WEEKDAY, DEFAULT_SCHEDULE_WEEKDAY
            ),
            CONF_SCHEDULE_TIME: curr_opts.get(
                CONF_SCHEDULE_TIME, DEFAULT_SCHEDULE_TIME
            ),
            CONF_TRUNCATION_GUARD: curr_opts.get(CONF_TRUNCATION_GUARD, True),
            CONF_VALIDATE_AFTER_LEARNING: curr_opts.get(
                CONF_VALIDATE_AFTER_LEARNING, DEFAULT_VALIDATE_AFTER_LEARNING
            ),
        }
        if CONF_PRICE_INPUT in curr_opts:
            suggested[CONF_PRICE_INPUT] = curr_opts[CONF_PRICE_INPUT]
        if CONF_PRICE_OUTPUT in curr_opts:
            suggested[CONF_PRICE_OUTPUT] = curr_opts[CONF_PRICE_OUTPUT]
        if CONF_MONTH_COST_LIMIT in curr_opts:
            suggested[CONF_MONTH_COST_LIMIT] = curr_opts[CONF_MONTH_COST_LIMIT]

        if user_input is not None:
            price_in = user_input.get(CONF_PRICE_INPUT)
            price_out = user_input.get(CONF_PRICE_OUTPUT)
            cost_lim = user_input.get(CONF_MONTH_COST_LIMIT)

            has_cost_lim = cost_lim is not None and cost_lim != ""
            has_price_in = price_in is not None and price_in != ""
            has_price_out = price_out is not None and price_out != ""

            if has_cost_lim and (not has_price_in or not has_price_out):
                errors["base"] = "cost_limit_needs_prices"

            max_tokens = int(user_input[CONF_MAX_TOKENS])
            run_output = int(user_input[CONF_RUN_OUTPUT_TOKENS])
            if max_tokens > run_output and "base" not in errors:
                errors["base"] = "max_tokens_above_run_cap"

            if not errors:
                merged = dict(self.config_entry.options)
                merged[CONF_SHARE_NAMES] = bool(user_input.get(CONF_SHARE_NAMES, False))
                merged[CONF_MAX_TOKENS] = max_tokens
                merged[CONF_RUN_INPUT_TOKENS] = int(user_input[CONF_RUN_INPUT_TOKENS])
                merged[CONF_RUN_OUTPUT_TOKENS] = run_output
                merged[CONF_MONTH_INPUT_TOKENS] = int(user_input[CONF_MONTH_INPUT_TOKENS])
                merged[CONF_MONTH_OUTPUT_TOKENS] = int(user_input[CONF_MONTH_OUTPUT_TOKENS])
                merged[CONF_SCHEDULE] = user_input.get(CONF_SCHEDULE, DEFAULT_SCHEDULE)
                merged[CONF_SCHEDULE_WEEKDAY] = user_input.get(
                    CONF_SCHEDULE_WEEKDAY, DEFAULT_SCHEDULE_WEEKDAY
                )
                merged[CONF_SCHEDULE_TIME] = user_input.get(
                    CONF_SCHEDULE_TIME, DEFAULT_SCHEDULE_TIME
                )
                merged[CONF_TRUNCATION_GUARD] = bool(
                    user_input.get(CONF_TRUNCATION_GUARD, True)
                )
                merged[CONF_VALIDATE_AFTER_LEARNING] = bool(
                    user_input.get(CONF_VALIDATE_AFTER_LEARNING, False)
                )

                if has_price_in:
                    merged[CONF_PRICE_INPUT] = float(price_in)
                else:
                    merged.pop(CONF_PRICE_INPUT, None)

                if has_price_out:
                    merged[CONF_PRICE_OUTPUT] = float(price_out)
                else:
                    merged.pop(CONF_PRICE_OUTPUT, None)

                if has_cost_lim:
                    merged[CONF_MONTH_COST_LIMIT] = float(cost_lim)
                else:
                    merged.pop(CONF_MONTH_COST_LIMIT, None)

                return self.async_create_entry(data=merged)

        schema = vol.Schema(
            {
                vol.Required(CONF_SHARE_NAMES): selector.BooleanSelector(),
                vol.Required(CONF_MAX_TOKENS): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=256,
                        max=1_000_000,
                        step=1,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(CONF_RUN_INPUT_TOKENS): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1000,
                        max=1_000_000_000,
                        step=1,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(CONF_RUN_OUTPUT_TOKENS): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1000,
                        max=1_000_000_000,
                        step=1,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(CONF_MONTH_INPUT_TOKENS): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1000,
                        max=1_000_000_000,
                        step=1,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(CONF_MONTH_OUTPUT_TOKENS): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=1000,
                        max=1_000_000_000,
                        step=1,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(CONF_PRICE_INPUT): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=1000,
                        step=0.001,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(CONF_PRICE_OUTPUT): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0,
                        max=1000,
                        step=0.001,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Optional(CONF_MONTH_COST_LIMIT): selector.NumberSelector(
                    selector.NumberSelectorConfig(
                        min=0.01,
                        max=100_000,
                        step=0.01,
                        mode=selector.NumberSelectorMode.BOX,
                    )
                ),
                vol.Required(CONF_SCHEDULE): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=["off", "weekly"],
                        translation_key="schedule",
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(CONF_SCHEDULE_WEEKDAY): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=list(WEEKDAYS),
                        translation_key="weekday",
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Required(CONF_SCHEDULE_TIME): selector.TimeSelector(),
                vol.Required(CONF_TRUNCATION_GUARD): selector.BooleanSelector(),
                vol.Required(CONF_VALIDATE_AFTER_LEARNING): selector.BooleanSelector(),
            }
        )

        return self.async_show_form(
            step_id="learning",
            data_schema=self.add_suggested_values_to_schema(
                schema, user_input if user_input is not None else suggested
            ),
            errors=errors,
        )

    async def async_step_review(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle candidate review selector step (candidates + devices, C2)."""
        hub = getattr(self.config_entry, "runtime_data", None)
        if hub is None or getattr(hub, "learning", None) is None:
            return self.async_abort(reason="not_loaded")

        entries = hub.learning.review_entries()
        if not entries:
            return self.async_abort(reason="no_candidates")

        errors: dict[str, str] = {}
        if self._review_errors:
            errors = dict(self._review_errors)
            self._review_errors.clear()

        if user_input is not None:
            cand_id = str(user_input.get("candidate", ""))
            if cand_id.startswith("device_"):
                dev_uuid = cand_id[len("device_") :]
                rec = hub.devices.record(dev_uuid)
                if rec is None or rec.get("status") != "confirmed":
                    errors["base"] = "candidate_gone"
                else:
                    self._device_uuid = dev_uuid
                    self._candidate_id = None
                    return await self.async_step_device()
            elif not any(
                c.get("signature_id") == cand_id
                for c in hub.learning.visible_candidates()
            ):
                errors["base"] = "candidate_gone"
            else:
                self._candidate_id = cand_id
                self._device_uuid = None
                return await self.async_step_candidate()

        options = [
            selector.SelectOptionDict(
                value=str(e.get("option_value") or e["signature_id"]),
                label=hub.learning.candidate_label(e),
            )
            for e in entries
        ]
        schema = vol.Schema(
            {
                vol.Required("candidate"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=options,
                        mode=selector.SelectSelectorMode.LIST,
                    )
                ),
            }
        )
        return self.async_show_form(
            step_id="review",
            data_schema=schema,
            errors=errors,
        )

    def _device_entry(self, hub: Any, dev_uuid: str | None) -> dict[str, Any] | None:
        """Return the review entry for a confirmed device uuid, if confirmed."""
        if dev_uuid is None:
            return None
        rec = hub.devices.record(dev_uuid)
        if rec is None or rec.get("status") != "confirmed":
            return None
        return next(
            (
                e
                for e in hub.devices.confirmed_entries()
                if e["device_uuid"] == dev_uuid
            ),
            None,
        )

    @staticmethod
    def _device_merge_sources(hub: Any, dev_uuid: str) -> list[dict[str, Any]]:
        """Merge sources for a confirmed device (D-12, D-13, D-23, C3, D-40).

        The sources are the other confirmed switched devices (option value
        ``device_<uuid>``) and the visible unconfirmed switched candidates
        (signature ids, C1): never a program candidate, never a confirmed
        program device, never a retired device, never the device itself. A
        program target has no sources at all: in v0.x a program device can be
        neither a merge source nor a merge target (D-40, RESEARCH Pitfall 9),
        because a union retrain has no meaning for a stage template. The device
        menu offers Merge while at least one source exists.
        """
        target = hub.devices.record(dev_uuid) or {}
        if target.get("kind") == "program":
            return []
        sources: list[dict[str, Any]] = [
            e
            for e in hub.devices.confirmed_entries()
            if e["device_uuid"] != dev_uuid
            and (e.get("kind") or "switched") == "switched"
        ]
        sources.extend(
            c
            for c in hub.learning.visible_candidates()
            if c.get("kind") == "switched"
        )
        return sources

    async def async_step_device(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the confirmed-device menu (D-11, D-12, D-23, D-25).

        Offers Accept update while a proposal is pending, Restore an earlier
        parameter version with two or more stored versions and Merge another
        confirmed device into it while at least one merge source exists; Rename
        is always available. It never offers Confirm or Dismiss — those are
        candidate actions.
        """
        hub = getattr(self.config_entry, "runtime_data", None)
        if hub is None or getattr(hub, "learning", None) is None:
            return self.async_abort(reason="not_loaded")

        entry = self._device_entry(hub, self._device_uuid)
        if entry is None:
            self._review_errors = {"base": "candidate_gone"}
            return await self.async_step_review()

        rec = hub.devices.record(self._device_uuid)
        menu_opts: list[str] = []
        if entry["update_available"]:
            menu_opts.append("accept_update")
        if entry["versions_count"] >= 2:
            menu_opts.append("rollback")
        if self._device_merge_sources(hub, self._device_uuid):
            menu_opts.append("merge")
        menu_opts.append("rename")
        return self.async_show_menu(
            step_id="device",
            menu_options=menu_opts,
            description_placeholders={
                "candidate": _escape_markdown(hub.learning.candidate_label(entry))
            },
        )

    async def async_step_rollback(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Restore an earlier stored parameter version (D-25).

        Lists every stored version except the active one (so an undone
        rollback can be redone) and re-activates the pick through
        hub.devices.async_activate_version — the same path Accept uses. No
        version is appended or removed, the uuid, unique_ids, entity_ids
        and energy totals never change, and a pending proposal is left as it
        is.
        """
        hub = getattr(self.config_entry, "runtime_data", None)
        if hub is None or getattr(hub, "learning", None) is None:
            return self.async_abort(reason="not_loaded")

        entry = self._device_entry(hub, self._device_uuid)
        if entry is None:
            self._review_errors = {"base": "candidate_gone"}
            return await self.async_step_review()
        rec = hub.devices.record(self._device_uuid)
        versions = [v for v in rec.get("versions") or [] if isinstance(v, dict)]
        others = [v for v in versions if v.get("v") != rec.get("active_v")]

        errors: dict[str, str] = {}
        if user_input is not None:
            raw_v = user_input.get("version")
            try:
                target_v = int(str(raw_v))
            except (TypeError, ValueError):
                target_v = -1
            if not any(int(v.get("v") or 0) == target_v for v in others):
                errors["base"] = "candidate_gone"
            else:
                await hub.devices.async_activate_version(self._device_uuid, target_v)
                return await self.async_step_review()

        options = [
            selector.SelectOptionDict(
                value=str(v.get("v")),
                label=(
                    f"v{v.get('v')} — {v.get('source')}, "
                    f"accepted {str(v.get('accepted_at') or '')[:10]}"
                ),
            )
            for v in sorted(others, key=lambda x: int(x.get("v") or 0))
        ]
        schema = vol.Schema(
            {
                vol.Required("version"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=options,
                        mode=selector.SelectSelectorMode.LIST,
                    )
                ),
            }
        )
        return self.async_show_form(
            step_id="rollback",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "candidate": _escape_markdown(hub.learning.candidate_label(entry))
            },
        )

    async def async_step_accept_update(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Accept the pending proposal as a new parameter version (D-11)."""
        hub = getattr(self.config_entry, "runtime_data", None)
        if hub is None or getattr(hub, "learning", None) is None:
            return self.async_abort(reason="not_loaded")

        entry = self._device_entry(hub, self._device_uuid)
        if entry is None or not entry["update_available"]:
            self._review_errors = {"base": "candidate_gone"}
            return await self.async_step_review()
        await hub.devices.async_accept_proposal(self._device_uuid)
        return await self.async_step_review()

    async def async_step_candidate(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle candidate actions menu (name, merge, dismiss)."""
        hub = getattr(self.config_entry, "runtime_data", None)
        if hub is None or getattr(hub, "learning", None) is None:
            return self.async_abort(reason="not_loaded")

        visible = hub.learning.visible_candidates()
        cand = next(
            (c for c in visible if c.get("signature_id") == self._candidate_id), None
        )
        if cand is None:
            self._review_errors = {"base": "candidate_gone"}
            return await self.async_step_review()

        label = hub.learning.candidate_label(cand)
        menu_opts = []
        # D-05 lifts the P4 D-22 restriction: a program or session candidate
        # gets the same Confirm action, subentry, UUID unique_id and
        # versioning path as a switched one. The former "becomes possible
        # once program detection is available" note is gone.
        if cand.get("kind") in ("switched", "program"):
            menu_opts.append("confirm")
        menu_opts.append("rename")
        if len(visible) > 1:
            menu_opts.append("merge")
        menu_opts.append("dismiss")
        return self.async_show_menu(
            step_id="candidate",
            menu_options=menu_opts,
            description_placeholders={"candidate": _escape_markdown(label)},
        )

    async def async_step_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle confirming a switched or program candidate as a device (D-01, D-02, D-05)."""
        hub = getattr(self.config_entry, "runtime_data", None)
        if hub is None or getattr(hub, "learning", None) is None:
            return self.async_abort(reason="not_loaded")

        cand_id = self._candidate_id
        visible = hub.learning.visible_candidates()
        cand = next(
            (c for c in visible if c.get("signature_id") == cand_id), None
        )
        if cand is None or cand_id is None:
            self._review_errors = {"base": "candidate_gone"}
            return await self.async_step_review()

        scores = cand.get("scores") or {}
        validated = scores.get("validated")
        if validated is None:
            validated = hub.learning.candidates_validated
        preview = not bool(validated)
        label = hub.learning.candidate_label(cand)

        errors: dict[str, str] = {}
        if user_input is not None:
            raw_name = str(user_input.get("name", ""))
            try:
                norm_name = normalise_name(raw_name)
            except ValueError:
                errors["base"] = "name_invalid"
            else:
                await hub.devices.async_confirm(
                    cand,
                    norm_name,
                    preview=preview,
                    trained_at=hub.learning.candidates_created_at,
                )
                return await self.async_step_review()

        dec = hub.learning.review.decisions.get(cand_id)
        current_name = (
            (dec.get("name") if dec else None)
            or cand.get("label")
            or ""
        )
        warning = (
            ""
            if not preview
            else "\n\n**Unvalidated:** this candidate is based on less than 21 days "
            "of data. Its estimates are provisional."
        )
        schema = vol.Schema(
            {
                vol.Required("name"): selector.TextSelector(),
            }
        )
        suggested = {"name": current_name}
        return self.async_show_form(
            step_id="confirm",
            data_schema=self.add_suggested_values_to_schema(schema, suggested),
            errors=errors,
            description_placeholders={
                "candidate": _escape_markdown(label),
                "warning": warning,
            },
        )

    async def async_step_rename(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle renaming a candidate or a confirmed device."""
        hub = getattr(self.config_entry, "runtime_data", None)
        if hub is None or getattr(hub, "learning", None) is None:
            return self.async_abort(reason="not_loaded")

        if self._device_uuid is not None:
            return await self._async_device_rename(hub, user_input)

        cand_id = self._candidate_id
        visible = hub.learning.visible_candidates()
        cand = next(
            (c for c in visible if c.get("signature_id") == cand_id), None
        )
        if cand is None or cand_id is None:
            self._review_errors = {"base": "candidate_gone"}
            return await self.async_step_review()

        errors: dict[str, str] = {}
        if user_input is not None:
            raw_name = str(user_input.get("name", ""))
            try:
                norm_name = normalise_name(raw_name)
            except ValueError:
                errors["base"] = "name_invalid"
            else:
                try:
                    await hub.learning.async_record_decision(
                        "rename", cand_id, name=norm_name
                    )
                    return await self.async_step_review()
                except CandidateGone:
                    self._review_errors = {"base": "candidate_gone"}
                    return await self.async_step_review()

        dec = hub.learning.review.decisions.get(cand_id)
        current_name = (dec.get("name") if dec else None) or cand.get("label") or ""
        label = hub.learning.candidate_label(cand)

        # History, not the default: the pre-fill above is the learning run's
        # latest guess (or the user's own name); when an earlier run first saw
        # this candidate under a different label, say so here so the user can
        # weigh both before naming.
        first_label = cand.get("first_label")
        first_seen = ""
        if first_label and str(first_label) != current_name:
            first_seen = "\n\nFirst seen as: " + _escape_markdown(str(first_label))

        schema = vol.Schema(
            {
                vol.Required("name"): selector.TextSelector(),
            }
        )
        suggested = {"name": current_name}
        return self.async_show_form(
            step_id="rename",
            data_schema=self.add_suggested_values_to_schema(schema, suggested),
            errors=errors,
            description_placeholders={
                "candidate": _escape_markdown(label),
                "first_seen": first_seen,
            },
        )

    async def _async_device_rename(
        self, hub: Any, user_input: dict[str, Any] | None
    ) -> ConfigFlowResult:
        """Rename a confirmed device from Review (OUT-05).

        Updates the device subentry's title and data name; the Plan 04-05
        update-listener reconcile mirrors the name into the cold record, the
        review record and the device registry. Entity ids and unique_ids
        never change and nothing reloads.
        """
        dev_uuid = self._device_uuid
        entry = self._device_entry(hub, dev_uuid)
        if entry is None:
            self._review_errors = {"base": "candidate_gone"}
            return await self.async_step_review()
        rec = hub.devices.record(dev_uuid)

        errors: dict[str, str] = {}
        if user_input is not None:
            raw_name = str(user_input.get("name", ""))
            try:
                norm_name = normalise_name(raw_name)
            except ValueError:
                errors["base"] = "name_invalid"
            else:
                sub = next(
                    (
                        s
                        for s in self.config_entry.get_subentries_of_type(
                            SUBENTRY_TYPE_DEVICE
                        )
                        if s.subentry_id == rec.get("subentry_id")
                    ),
                    None,
                )
                if sub is None:
                    self._review_errors = {"base": "candidate_gone"}
                    return await self.async_step_review()
                self.hass.config_entries.async_update_subentry(
                    self.config_entry,
                    sub,
                    title=norm_name,
                    data={**sub.data, "name": norm_name},
                )
                return await self.async_step_review()

        schema = vol.Schema(
            {
                vol.Required("name"): selector.TextSelector(),
            }
        )
        suggested = {"name": entry["label"]}
        return self.async_show_form(
            step_id="rename",
            data_schema=self.add_suggested_values_to_schema(schema, suggested),
            errors=errors,
            description_placeholders={
                "candidate": _escape_markdown(hub.learning.candidate_label(entry))
            },
        )

    async def async_step_merge(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle merging other candidates or devices into the selected one.

        Candidate target (unconfirmed): the source selector lists the other
        visible candidates only, which never include a device entry (D-23, C3).
        Device target (confirmed): the selector lists the other confirmed
        devices as ``device_<uuid>`` options plus the visible unconfirmed
        switched candidates (signature ids, D-13, C3); confirmed picks retire
        into the target, candidate picks add an alias and start a union
        retrain, both through the same review decision.
        """
        hub = getattr(self.config_entry, "runtime_data", None)
        if hub is None or getattr(hub, "learning", None) is None:
            return self.async_abort(reason="not_loaded")

        if self._device_uuid is not None:
            return await self._async_device_merge(hub, user_input)

        cand_id = self._candidate_id
        visible = hub.learning.visible_candidates()
        cand = next(
            (c for c in visible if c.get("signature_id") == cand_id), None
        )
        if cand is None or cand_id is None:
            self._review_errors = {"base": "candidate_gone"}
            return await self.async_step_review()

        other_candidates = [c for c in visible if c.get("signature_id") != cand_id]
        if not other_candidates:
            return await self.async_step_candidate()

        errors: dict[str, str] = {}
        if user_input is not None:
            sources = user_input.get("sources")
            if isinstance(sources, str):
                sources = [sources]
            elif not isinstance(sources, list):
                sources = list(sources or [])
            try:
                await hub.learning.async_record_decision(
                    "merge", cand_id, sources=sources
                )
                return await self.async_step_review()
            except CandidateGone:
                self._review_errors = {"base": "candidate_gone"}
                return await self.async_step_review()

        options = [
            selector.SelectOptionDict(
                value=c["signature_id"],
                label=hub.learning.candidate_label(c),
            )
            for c in other_candidates
        ]
        label = hub.learning.candidate_label(cand)
        schema = vol.Schema(
            {
                vol.Required("sources"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=options,
                        multiple=True,
                        mode=selector.SelectSelectorMode.LIST,
                    )
                ),
            }
        )
        return self.async_show_form(
            step_id="merge",
            data_schema=schema,
            errors=errors,
            description_placeholders={"candidate": _escape_markdown(label)},
        )

    async def _async_device_merge(
        self, hub: Any, user_input: dict[str, Any] | None
    ) -> ConfigFlowResult:
        """Merge confirmed devices or unconfirmed candidates into this device.

        The source selector lists the other confirmed devices (option value
        ``device_<uuid>``, never the device itself or a retired device) and
        the visible unconfirmed switched candidates (signature ids, D-13,
        C3); each picked option maps to its signature id for the review
        decision. After the merge the device menu offers Merge again only
        while another source exists.
        """
        entry = self._device_entry(hub, self._device_uuid)
        if entry is None:
            self._review_errors = {"base": "candidate_gone"}
            return await self.async_step_review()
        rec = hub.devices.record(self._device_uuid)
        target_sig = str((rec or {}).get("signature_id") or "")
        sources = self._device_merge_sources(hub, self._device_uuid)
        if not sources or not target_sig:
            return await self.async_step_device()

        errors: dict[str, str] = {}
        if user_input is not None:
            raw = user_input.get("sources")
            if isinstance(raw, str):
                raw = [raw]
            elif not isinstance(raw, list):
                raw = list(raw or [])
            source_sigs: list[str] = []
            invalid = False
            for value in raw:
                value = str(value)
                if value.startswith("device_"):
                    src_uuid = value[len("device_") :]
                    src_rec = hub.devices.record(src_uuid) if src_uuid else None
                    if src_rec is None or not src_rec.get("signature_id"):
                        invalid = True
                        break
                    source_sigs.append(str(src_rec["signature_id"]))
                else:
                    # An unconfirmed source: a visible switched candidate only
                    # (D-13); signature ids never start with "device_".
                    cand = next(
                        (
                            c
                            for c in hub.learning.visible_candidates()
                            if c.get("signature_id") == value
                        ),
                        None,
                    )
                    if cand is None or cand.get("kind") != "switched":
                        invalid = True
                        break
                    source_sigs.append(value)
            if invalid:
                errors["base"] = "candidate_gone"
            else:
                try:
                    await hub.learning.async_record_decision(
                        "merge", target_sig, sources=source_sigs
                    )
                    return await self.async_step_review()
                except (CandidateGone, ValueError):
                    self._review_errors = {"base": "candidate_gone"}
                    return await self.async_step_review()

        options = [
            selector.SelectOptionDict(
                # Confirmed-device sources carry option_value device_<uuid>;
                # candidate sources are keyed by their signature id (C1).
                value=str(
                    e.get("option_value") or e.get("signature_id") or ""
                ),
                label=hub.learning.candidate_label(e),
            )
            for e in sources
        ]
        schema = vol.Schema(
            {
                vol.Required("sources"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=options,
                        multiple=True,
                        mode=selector.SelectSelectorMode.LIST,
                    )
                ),
            }
        )
        return self.async_show_form(
            step_id="merge",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "candidate": _escape_markdown(hub.learning.candidate_label(entry))
            },
        )

    async def async_step_dismiss(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle dismissing the selected candidate."""
        hub = getattr(self.config_entry, "runtime_data", None)
        if hub is None or getattr(hub, "learning", None) is None:
            return self.async_abort(reason="not_loaded")

        cand_id = self._candidate_id
        if cand_id:
            try:
                await hub.learning.async_record_decision("dismiss", cand_id)
            except CandidateGone:
                self._review_errors = {"base": "candidate_gone"}
                return await self.async_step_review()

        return await self.async_step_review()
