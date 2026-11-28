"""Config flow for PowerGuesser integration."""

import asyncio
from collections.abc import Mapping
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
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import (
    config_validation as cv,
    entity_registry as er,
    issue_registry as ir,
    selector,
)
from homeassistant.helpers.translation import async_get_translations
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
from . import review_actions
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


def _format_score(value: Any) -> str:
    """Render a validation-day score for the device menu, or its raw value."""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


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


def _parse_local(value: Any) -> datetime | None:
    """Parse a stored ISO timestamp (or a datetime) for local display, or None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = dt_util.parse_datetime(str(value))
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt_util.get_default_time_zone())
    return parsed.astimezone(dt_util.get_default_time_zone())


def _local_clock(value: Any) -> str:
    """A stored timestamp as a local HH:MM clock time, or ""."""
    parsed = _parse_local(value)
    return parsed.strftime("%H:%M") if parsed is not None else ""


def _local_stamp(value: Any) -> str:
    """A stored timestamp as a local YYYY-MM-DD HH:MM stamp, or "unknown time"."""
    parsed = _parse_local(value)
    return parsed.strftime("%Y-%m-%d %H:%M") if parsed is not None else "unknown time"


# The read-only "Recent captures" attachment phrases (D-10, Plan 09-07).
_ATTACHED_PHRASES = {
    "preview": "teaches its preview",
    "candidate_example": "an example of the candidate",
    "device_example": "trains the device's next update proposal",
    "not_teaching": "kept as a labelled example; does not teach",
    "target_gone": "target no longer exists: reassign or delete",
}

# The Capture form's "new device" target value, distinct from the
# ``candidate_<sig>`` / ``device_<uuid>`` option values.
_CAPTURE_NEW_DEVICE = "new_device"


def _capture_w(value: Any) -> str:
    """A step power as ``<W> W``, or ``n/a``."""
    try:
        return f"{round(float(value))} W"
    except (TypeError, ValueError):
        return "n/a"


def _capture_min(value: Any) -> str:
    """A duration in seconds as ``<min> min``, or ``n/a``."""
    try:
        return f"{round(float(value) / 60)} min"
    except (TypeError, ValueError):
        return "n/a"


def _capture_wh(value: Any) -> str:
    """An energy in Wh as ``<Wh> Wh``, or ``n/a``."""
    try:
        return f"{round(float(value))} Wh"
    except (TypeError, ValueError):
        return "n/a"


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
        self._capture_result: str = ""

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
            menu_options=[
                "phases", "llm_endpoint", "learning", "review",
                "plug_sessions", "capture",
            ],
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

    # -- Capture (CAP-08, CAP-11, CAP-12; D-04, D-10..D-12) -----------------

    def _capture_manager(self) -> Any:
        """Return the capture manager, or None while PowerGuesser is not loaded."""
        hub = getattr(self.config_entry, "runtime_data", None)
        if hub is None:
            return None
        return getattr(hub, "captures", None)

    @staticmethod
    def _append_insight(base: str, text: str) -> str:
        """Append the markdown-escaped capture insight to a placeholder (D-11).

        Keeps the existing block untouched; prefixes two newlines so the insight
        is its own markdown paragraph (or returns ``base`` unchanged when there
        is no insight).
        """
        if not text:
            return base
        return base + "\n\n" + _escape_markdown(text)

    def _capture_status_text(self, hub: Any) -> str:
        """The Capture menu's status line, plus the last action result (D-04)."""
        status = hub.captures.status_view()
        state = str(status.get("state") or "idle")
        target = status.get("target_name") or "the capture"
        if state == "recording":
            clock = _local_clock(status.get("started_at"))
            base = (
                f"Recording since {clock} for {target}."
                if clock
                else f"Recording for {target}."
            )
        elif state == "resolving":
            base = (
                f"Resolving the capture for {target}; the result appears in "
                "about 2-3 minutes."
            )
        else:
            base = "No capture is running."
        if self._capture_result:
            return f"{base}\n\n{self._capture_result}"
        return base

    async def _refusal_text(self, err: ServiceValidationError) -> str:
        """Render a capture refusal in the user's own language (never waiting).

        Returns the message *unescaped*. A caller that puts the result straight
        into a markdown-rendered description placeholder escapes it with
        ``_escape_markdown`` (WR-03: the messages embed LLM-derived candidate
        names in ``{choices}``); the ``_capture_result`` path stores it raw and is
        escaped once when the Capture menu renders it.

        HA's ``async_get_exception_message`` is English-only (it resolves against
        the ``"en"`` cache) and HA always caches English as the fallback base, so
        the message is read here from the configured locale with an English
        fallback. A cold cache or a locale without the key falls back to the raw
        translation key, exactly as the helper did.
        """
        key = str(err.translation_key or "")
        if not key:
            return str(err)
        placeholders = dict(err.translation_placeholders or {})
        localize_key = f"component.{DOMAIN}.exceptions.{key}.message"
        for language in dict.fromkeys((self.hass.config.language or "en", "en")):
            try:
                # Warm the exceptions category for this language, then read the
                # message the helper would have returned (minus its English bias).
                translations = await async_get_translations(
                    self.hass, language, "exceptions", [DOMAIN]
                )
            except Exception:  # noqa: BLE001 - a translation failure must not break the menu
                continue
            message = translations.get(localize_key)
            if not message:
                continue
            message = message.rstrip(".")
            if not placeholders:
                return message
            try:
                return message.format(**placeholders)
            except (KeyError, IndexError, ValueError):
                return message
        return key

    def _capture_target_choice(
        self, hub: Any, target_value: str, name: str
    ) -> tuple[str, str, str] | None:
        """Resolve a Capture form's target to ``(target_type, target, label)``.

        A ``new_device`` needs a valid name (``ValueError`` -> ``name_invalid``);
        a candidate or device stores the target's *user-supplied* name, else the
        signature id, as the label (D-05 owner revision: the decorated option
        text, an LLM label or a placeholder is never stored as a label). Returns
        ``None`` when a rendered candidate/device option no longer resolves.
        """
        if target_value == _CAPTURE_NEW_DEVICE:
            norm = normalise_name(name)
            return ("new_device", norm, norm)
        kind, ident = self._parse_session_target(target_value)
        if kind is None:
            return None
        if kind == "candidate":
            dec = hub.learning.review.decisions.get(ident) or {}
            label = dec.get("name")
        else:
            rec = hub.devices.record(ident)
            label = (rec or {}).get("name")
        return (kind, ident, str(label) if label else ident)

    def _capture_target_options(self) -> list[selector.SelectOptionDict]:
        """The Capture target choices: New device, then candidates and devices."""
        options = [
            selector.SelectOptionDict(
                value=_CAPTURE_NEW_DEVICE, label="New device (name below)"
            )
        ]
        options.extend(self._session_targets())
        return options

    def _capture_target_selector(self) -> Any:
        return selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=self._capture_target_options(),
                mode=selector.SelectSelectorMode.LIST,
            )
        )

    def _capture_entry_lines(self, entry: Any) -> list[str]:
        """The markdown lines for one capture's list_view entry (D-10..D-12)."""
        result = entry.get("result")
        if not isinstance(result, Mapping):
            result = {}
        target = entry.get("target")
        name = (target or {}).get("name") or "a target that no longer exists"
        when = _local_stamp((entry.get("window") or {}).get("start_local"))
        state = str(entry.get("state") or "")
        verdict = result.get("verdict")
        # A capture that is not done (recording/resolving/interrupted) carries no
        # result, so render a friendly placeholder instead of interpolating None.
        verdict_text = str(verdict) if verdict is not None else "no verdict"
        lines = [
            "- "
            + f"{when} — {name} — {state}, {verdict_text}: "
            + f"{_capture_w(result.get('step_w'))}, "
            + f"{_capture_min(result.get('dur_s'))}, "
            + f"{_capture_wh(result.get('energy_wh'))} — "
            + _ATTACHED_PHRASES.get(
                str(entry.get("attached_as") or ""),
                str(entry.get("attached_as") or ""),
            )
        ]
        similar = result.get("similar")
        if not isinstance(similar, Mapping):
            similar = {}
        twin = similar.get("twin")
        if not isinstance(twin, Mapping):
            twin = {}
        twin_id = str(twin.get("id")) if twin.get("id") else None
        rendered: set[str] = set()
        for item in (similar.get("top") or [])[:3]:
            if not isinstance(item, Mapping):
                continue
            item_id = str(item.get("id"))
            rendered.add(item_id)
            note = ", near twin" if twin_id is not None and item_id == twin_id else ""
            lines.append(f"  Most similar: {item.get('name')} ({item.get('score')}){note}")
        if similar.get("near_twin") and twin_id is not None and twin_id not in rendered:
            lines.append(f"  Near twin: {twin.get('name')}")
        earlier = entry.get("earlier_runs")
        if isinstance(earlier, Mapping) and earlier.get("n") is not None:
            lines.append(
                f"  Earlier runs matched: {earlier.get('n')} ({earlier.get('basis')})"
            )
        llm = entry.get("llm_verdict")
        if llm:
            lines.append(f"  LLM: {llm}")
        # D-10: a non-clean device capture never trains the device.
        if (
            entry.get("target_kind") == "device"
            and entry.get("target_state") == "ok"
            and verdict != "clean"
        ):
            lines.append("  Not used to retrain the device (only clean captures train)")
        replay = entry.get("replay")
        if isinstance(replay, Mapping):
            if replay.get("would_claim"):
                lines.append(
                    "  Replay check (evidence only): the active version alone "
                    "would have claimed this run"
                )
            else:
                lines.append(
                    "  Replay check (evidence only): the active version alone "
                    "would not have claimed this run"
                )
        return lines

    def _capture_lines(self, hub: Any, limit: int = 10) -> list[str]:
        """The newest ``limit`` captures, newest first, as markdown lines (D-11)."""
        entries = list(reversed(hub.captures.list_view()))[:limit]
        lines: list[str] = []
        for entry in entries:
            lines.extend(self._capture_entry_lines(entry))
        return lines

    async def async_step_capture(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """The Capture menu: Start, Stop (while recording), Retro and Recent (D-04)."""
        hub = getattr(self.config_entry, "runtime_data", None)
        captures = getattr(hub, "captures", None) if hub is not None else None
        if captures is None:
            return self.async_abort(reason="not_loaded")

        menu_opts = ["capture_start"]
        if captures.status_view().get("state") == "recording":
            menu_opts.append("capture_stop")
        menu_opts.extend(["capture_retro", "capture_recent"])
        status_text = _escape_markdown(self._capture_status_text(hub))
        # D-04: the last action result ("Recording started for X."/"Stopped. …")
        # belongs to the menu return that immediately follows the action; clear it
        # once shown so a later Capture visit does not keep re-rendering it.
        self._capture_result = ""
        return self.async_show_menu(
            step_id="capture",
            menu_options=menu_opts,
            description_placeholders={"status": status_text},
        )

    async def async_step_capture_start(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Start a live capture and return at once, never waiting (D-04)."""
        hub = getattr(self.config_entry, "runtime_data", None)
        captures = getattr(hub, "captures", None) if hub is not None else None
        if captures is None:
            return self.async_abort(reason="not_loaded")

        errors: dict[str, str] = {}
        refusal = ""
        if user_input is not None:
            try:
                choice = self._capture_target_choice(
                    hub,
                    str(user_input.get("target") or ""),
                    str(user_input.get("name") or ""),
                )
            except ValueError:
                errors["base"] = "name_invalid"
            else:
                if choice is None:
                    errors["base"] = "target_gone"
                else:
                    target_type, target, label = choice
                    try:
                        await captures.async_start(
                            label=label, target_type=target_type, target=target
                        )
                    except ServiceValidationError as err:
                        errors["base"] = "capture_refused"
                        refusal = _escape_markdown(await self._refusal_text(err))
                    else:
                        self._capture_result = f"Recording started for {label}."
                        return await self.async_step_capture()

        schema = vol.Schema(
            {
                vol.Required("target"): self._capture_target_selector(),
                vol.Optional("name"): selector.TextSelector(),
            }
        )
        return self.async_show_form(
            step_id="capture_start",
            data_schema=schema,
            errors=errors,
            description_placeholders={"refusal": refusal},
        )

    async def async_step_capture_stop(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Stop the live capture and return to the menu at once (D-04)."""
        hub = getattr(self.config_entry, "runtime_data", None)
        captures = getattr(hub, "captures", None) if hub is not None else None
        if captures is None:
            return self.async_abort(reason="not_loaded")

        try:
            await captures.async_stop()
        except ServiceValidationError as err:
            self._capture_result = await self._refusal_text(err)
        else:
            self._capture_result = (
                "Stopped. The result appears under Recent captures and on the "
                "Capture status sensor in about 2-3 minutes."
            )
        return await self.async_step_capture()

    async def async_step_capture_recent(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """List the newest captures with their insight (read-only, D-11)."""
        hub = getattr(self.config_entry, "runtime_data", None)
        captures = getattr(hub, "captures", None) if hub is not None else None
        if captures is None:
            return self.async_abort(reason="not_loaded")

        if user_input is not None:
            return await self.async_step_capture()

        lines = self._capture_lines(hub)
        text = "\n".join(lines) if lines else "There are no captures yet."
        return self.async_show_form(
            step_id="capture_recent",
            data_schema=vol.Schema({}),
            description_placeholders={"captures": _escape_markdown(text)},
        )

    async def async_step_capture_retro(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Capture a past run from the menu; Submit commits and returns (D-04)."""
        hub = getattr(self.config_entry, "runtime_data", None)
        captures = getattr(hub, "captures", None) if hub is not None else None
        if captures is None:
            return self.async_abort(reason="not_loaded")

        errors: dict[str, str] = {}
        refusal = ""
        if user_input is not None:
            # The DateTimeSelector rejects malformed values before the handler
            # runs, but a submitted value outside it can still fail to parse: that
            # is an invalid window, not an invalid name, so report it on the field
            # rather than as name_invalid.
            try:
                start = cv.datetime(user_input.get("start"))
            except vol.Invalid:
                errors["start"] = "invalid_datetime"
            try:
                end = cv.datetime(user_input.get("end"))
            except vol.Invalid:
                errors["end"] = "invalid_datetime"

            if not errors:
                try:
                    choice = self._capture_target_choice(
                        hub,
                        str(user_input.get("target") or ""),
                        str(user_input.get("name") or ""),
                    )
                except ValueError:
                    errors["base"] = "name_invalid"
                else:
                    if choice is None:
                        errors["base"] = "target_gone"
                    else:
                        target_type, target, label = choice
                        try:
                            entry = await captures.async_capture_window(
                                start=start,
                                end=end,
                                label=label,
                                target_type=target_type,
                                target=target,
                            )
                        except ServiceValidationError as err:
                            errors["base"] = "capture_refused"
                            refusal = _escape_markdown(await self._refusal_text(err))
                        else:
                            self._capture_result = "\n".join(
                                self._capture_entry_lines(entry)
                            )
                            return await self.async_step_capture()

        schema = vol.Schema(
            {
                vol.Required("start"): selector.DateTimeSelector(),
                vol.Required("end"): selector.DateTimeSelector(),
                vol.Required("target"): self._capture_target_selector(),
                vol.Optional("name"): selector.TextSelector(),
            }
        )
        return self.async_show_form(
            step_id="capture_retro",
            data_schema=schema,
            errors=errors,
            description_placeholders={"refusal": refusal},
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

        captures = getattr(hub, "captures", None)
        orphan_count = len(captures.orphans()) if captures is not None else 0
        entries = hub.learning.review_entries()
        if not entries:
            # D-06 (owner Q6): an orphan-only state aborts with the count and the
            # locked wording, instead of the plain no_candidates abort.
            if orphan_count > 0:
                return self.async_abort(
                    reason="orphan_captures",
                    description_placeholders={"count": str(orphan_count)},
                )
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
        captures_notice = (
            ""
            if orphan_count == 0
            else (
                "\n\n**Captured runs without a target:** "
                f"{orphan_count} captured run(s): target no longer exists: reassign or delete them with the PowerGuesser capture actions."
            )
        )
        return self.async_show_form(
            step_id="review",
            data_schema=schema,
            errors=errors,
            description_placeholders={"captures_notice": captures_notice},
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

    def _proposal_text(self, hub: Any, uuid: str) -> str:
        """The pending-proposal block of the device menu (D-09, D-10).

        Names where a pending update comes from — the user's captures (with the
        captured-run count) or a learning run — and the gate's validation-day
        scores with the no-regression basis. Below it: how many captured runs of
        this device do not train because they are not clean (D-10), and the
        Phase 8 replay check as an evidence-only line (D-10). Empty when nothing
        applies; otherwise markdown-escaped and prefixed with two newlines so it
        follows the evidence block.
        """
        devices = getattr(hub, "devices", None)
        captures = getattr(hub, "captures", None)
        rec = devices.record(uuid) if devices is not None else None
        lines: list[str] = []
        pending = rec.get("pending_proposal") if isinstance(rec, dict) else None
        if isinstance(pending, dict):
            if pending.get("source") == "capture":
                n_runs = len(pending.get("capture_ids") or [])
                noun = "run" if n_runs == 1 else "runs"
                lines.append(f"Update available from your captures ({n_runs} captured {noun})")
            else:
                lines.append("Update available from a learning run")
            artefact = pending.get("artefact")
            gate = artefact.get("proposal_gate") if isinstance(artefact, dict) else None
            if isinstance(gate, dict):
                lines.append(
                    "Validation-day score: proposed "
                    f"{_format_score(gate.get('proposed_S'))} vs active "
                    f"{_format_score(gate.get('active_S'))} — {gate.get('basis')}"
                )
        counts = (
            captures.device_capture_counts(uuid) if captures is not None else None
        )
        if counts is not None:
            if counts.get("not_clean"):
                lines.append(
                    f"{counts['not_clean']} captured runs on this device are not "
                    "used for retraining (only clean captures train)."
                )
            if counts.get("replay_checked"):
                lines.append(
                    "Replay check (evidence only): the active version alone would have claimed "
                    f"{counts['replay_claimed']} of {counts['replay_checked']} "
                    "checked captured runs; it does not decide whether an update is proposed."
                )
        if not lines:
            return ""
        return "\n\n" + _escape_markdown("\n\n".join(lines))

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

        menu_opts: list[str] = []
        if entry["update_available"]:
            menu_opts.append("accept_update")
        if entry["versions_count"] >= 2:
            menu_opts.append("rollback")
        if self._device_merge_sources(hub, self._device_uuid):
            menu_opts.append("merge")
        menu_opts.append("rename")
        evidence, state = review_actions.evidence_view(
            hub, dev_uuid=self._device_uuid
        )
        return self.async_show_menu(
            step_id="device",
            menu_options=menu_opts,
            description_placeholders={
                "candidate": _escape_markdown(hub.learning.candidate_label(entry)),
                # D-07a: the same block as a candidate, replayed from the
                # device's active artefact (or the "n/a" block without one).
                # D-11/D-12 (Plan 09-07): the stored capture insight follows it,
                # advisory only, before the {proposal} block.
                "evidence": self._append_insight(
                    review_actions.evidence_text(evidence, state),
                    review_actions.capture_insight_text(
                        hub, "device", str(self._device_uuid or "")
                    ),
                ),
                # D-09/D-10 (Plan 09-06): where a pending update comes from, how
                # it scored, which captures do not train and the replay evidence.
                "proposal": self._proposal_text(hub, self._device_uuid),
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
        evidence, state = review_actions.evidence_view(hub, cand=cand)
        return self.async_show_menu(
            step_id="candidate",
            menu_options=menu_opts,
            description_placeholders={
                "candidate": _escape_markdown(label),
                # Plan 07-05 (D-06/D-07a/D-12): the full evidence block below the
                # unchanged candidate label; a row that comes from an unconfirmed
                # device record is read as a device, never as a pending candidate.
                # Plan 09-07: the stored capture insight follows it, advisory only.
                "evidence": self._append_insight(
                    review_actions.evidence_text(evidence, state),
                    review_actions.capture_insight_text(
                        hub, "candidate", str(self._candidate_id or "")
                    ),
                ),
            },
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

        # One preview rule and one gate rule, shared with the backend (D-03):
        # the form cannot disagree with review_actions.async_confirm_candidate.
        preview = review_actions.candidate_is_preview(hub, cand)
        gate = review_actions.capture_gate(cand)
        label = hub.learning.candidate_label(cand)

        errors: dict[str, str] = {}
        if user_input is not None:
            raw_name = str(user_input.get("name", ""))
            try:
                await review_actions.async_confirm_candidate(
                    hub,
                    cand_id,
                    raw_name,
                    override=bool(user_input.get("confirm_anyway")),
                )
            except review_actions.InvalidName:
                errors["base"] = "name_invalid"
            except review_actions.PreviewNeedsRuns:
                errors["base"] = "preview_needs_runs"
            except CandidateGone:
                self._review_errors = {"base": "candidate_gone"}
                return await self.async_step_review()
            else:
                return await self.async_step_review()

        dec = hub.learning.review.decisions.get(cand_id)
        current_name = (
            (dec.get("name") if dec else None)
            or cand.get("label")
            or ""
        )
        if gate is not None:
            n, required = gate
            warning = (
                "\n\n**Preview from captures:** based on "
                f"{review_actions.captured_runs_phrase(n)}; confirming needs at "
                f"least {required} or the Confirm anyway option."
            )
        elif preview:
            warning = (
                "\n\n**Unvalidated:** this candidate is based on less than 21 days "
                "of data. Its estimates are provisional."
            )
        else:
            warning = ""
        schema_fields: dict[Any, Any] = {
            vol.Required("name"): selector.TextSelector(),
        }
        suggested: dict[str, Any] = {"name": current_name}
        if gate is not None:
            schema_fields[
                vol.Optional("confirm_anyway", default=False)
            ] = selector.BooleanSelector()
            suggested["confirm_anyway"] = False
        schema = vol.Schema(schema_fields)
        return self.async_show_form(
            step_id="confirm",
            data_schema=self.add_suggested_values_to_schema(schema, suggested),
            errors=errors,
            description_placeholders={
                "candidate": _escape_markdown(label),
                # D-11 (Plan 09-07): the stored capture insight joins the warning,
                # advisory only: it never blocks Confirm.
                "warning": self._append_insight(
                    warning,
                    review_actions.capture_insight_text(
                        hub, "candidate", str(cand_id or "")
                    ),
                ),
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
                await review_actions.async_rename_candidate(hub, cand_id, raw_name)
            except review_actions.InvalidName:
                errors["base"] = "name_invalid"
            except CandidateGone:
                self._review_errors = {"base": "candidate_gone"}
                return await self.async_step_review()
            else:
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

        errors: dict[str, str] = {}
        if user_input is not None:
            raw_name = str(user_input.get("name", ""))
            try:
                await review_actions.async_rename_device(
                    self.hass, self.config_entry, hub, dev_uuid, raw_name
                )
            except review_actions.InvalidName:
                errors["base"] = "name_invalid"
            except CandidateGone:
                self._review_errors = {"base": "candidate_gone"}
                return await self.async_step_review()
            else:
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
            try:
                await review_actions.async_merge_candidate(
                    hub, cand_id, user_input.get("sources")
                )
            except CandidateGone:
                self._review_errors = {"base": "candidate_gone"}
                return await self.async_step_review()
            else:
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
            try:
                await review_actions.async_merge_device(
                    hub, self._device_uuid, user_input.get("sources")
                )
            except review_actions.InvalidSource:
                errors["base"] = "candidate_gone"
            except (CandidateGone, ValueError):
                self._review_errors = {"base": "candidate_gone"}
                return await self.async_step_review()
            else:
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
                await review_actions.async_dismiss_candidate(hub, cand_id)
            except CandidateGone:
                self._review_errors = {"base": "candidate_gone"}
                return await self.async_step_review()

        return await self.async_step_review()
