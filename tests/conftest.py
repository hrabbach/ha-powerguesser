"""Fixtures for PowerGuesser tests."""

from typing import Any
import pytest

from homeassistant.components.energy import async_get_manager
from homeassistant.components.energy.data import EnergyManager
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component

from custom_components.powerguesser.const import (
    BUFFER_DIR,
    BUFFER_FILE,
    CANDIDATES_STORE_KEY,
    DOMAIN,
    LEARNING_STORES_VERSION,
    VALIDATION_STORE_KEY,
    VALIDATION_STORES_VERSION,
)
from custom_components.powerguesser.engine.reader import load_gross_layout, open_readonly


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "bootstrap_enabled: run the recorder bootstrap in this test",
    )


@pytest.fixture(autouse=True)
def disable_bootstrap_schedule(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disable PowerGuesser bootstrap scheduling unless marked with bootstrap_enabled."""
    if request.node.get_closest_marker("bootstrap_enabled") is None:
        monkeypatch.setattr(
            "custom_components.powerguesser.hub.PowerGuesserHub.async_schedule_bootstrap",
            lambda self: None,
        )


@pytest.fixture
def hass_config_dir(hass_tmp_config_dir: str) -> str:
    """Return a temporary directory for HA configuration."""
    return hass_tmp_config_dir


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable custom integrations for all tests."""


@pytest.fixture
def mock_recorder_before_hass(recorder_db_url: str) -> None:
    """Ensure recorder_db_url is evaluated before hass."""


@pytest.fixture
async def energy_manager(recorder_mock: Any, hass: HomeAssistant) -> EnergyManager:
    """Fixture that initializes the energy component and returns EnergyManager."""
    assert await async_setup_component(hass, "energy", {"energy": {}})
    manager = await async_get_manager(hass)
    assert manager is not None
    manager.data = EnergyManager.default_preferences()
    return manager


@pytest.fixture
def std_prefs() -> dict[str, Any]:
    """Standard preferences fixture for Energy dashboard."""
    return {
        "energy_sources": [
            {
                "type": "grid",
                "stat_rate": "sensor.grid_power",
            },
            {
                "type": "solar",
                "stat_rate": "sensor.solar_power",
            },
            {
                "type": "battery",
                "stat_rate": "sensor.battery_power",
            },
        ],
        "device_consumption": [
            {
                "stat_consumption": "sensor.washer_energy",
                "stat_rate": "sensor.washer_power",
            }
        ],
    }


def async_set_power(
    hass: HomeAssistant,
    entity_id: str,
    value: Any,
    unit: str = "W",
) -> None:
    """Helper to set a power entity state with proper attributes."""
    if value in ("unavailable", "unknown"):
        hass.states.async_set(entity_id, value, {})
        return

    attrs: dict[str, Any] = {
        "device_class": "power",
        "unit_of_measurement": unit,
    }
    hass.states.async_set(entity_id, str(value), attrs)


async def async_setup_via_flow(hass: HomeAssistant) -> ConfigEntry:
    """Helper to run the PowerGuesser config flow and set up the entry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": "user"}
    )
    assert result["type"] == "form"
    result2 = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    assert result2["type"] == "create_entry"
    await hass.async_block_till_done()
    return result2["result"]


async def async_flush_buffer(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Flush pending buffer writer rows to SQLite."""
    await entry.runtime_data.writer.async_flush()
    await hass.async_block_till_done()


async def async_read_buffer(
    hass: HomeAssistant,
) -> tuple[dict[tuple[str, str], list[tuple[int, int | None]]], list[tuple[str, str, tuple[int, ...]]]]:
    """Read all samples and gross layout from the buffer database."""

    def _read() -> tuple[dict[tuple[str, str], list[tuple[int, int | None]]], list[tuple[str, str, tuple[int, ...]]]]:
        path = hass.config.path(BUFFER_DIR, BUFFER_FILE)
        conn = open_readonly(path)
        try:
            cur = conn.execute("SELECT id, entity_id, role FROM channels")
            channels = cur.fetchall()
            ch_map = {row[0]: (row[1], row[2]) for row in channels}

            data: dict[tuple[str, str], list[tuple[int, int | None]]] = {
                key: [] for key in ch_map.values()
            }
            for ch_id, key in ch_map.items():
                cur = conn.execute(
                    "SELECT ts, w FROM samples WHERE ch = ? ORDER BY ts ASC",
                    (ch_id,),
                )
                data[key] = cur.fetchall()

            layout = load_gross_layout(conn)
            return data, layout
        finally:
            conn.close()

    return await hass.async_add_executor_job(_read)


async def async_open_options_step(
    hass: HomeAssistant, entry: ConfigEntry, step_id: str
) -> dict[str, Any]:
    """Open options flow and advance from the menu to step_id."""
    from homeassistant.data_entry_flow import FlowResultType

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.MENU
    result2 = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": step_id}
    )
    return result2


def load_llm_cassette(file: str, name: str) -> dict[str, Any]:
    """Read named fixture from tests/fixtures/llm/<file>.json."""
    import json
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "llm" / f"{file}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[name]


def mock_llm(aioclient_mock: Any, base_url: str, responder: Any) -> None:
    """Register POST mock for {base_url}/chat/completions."""
    import inspect
    import aiohttp
    from pytest_homeassistant.test_util.aiohttp import AiohttpClientMockResponse

    async def _side_effect(method: str, url: Any, data: Any) -> AiohttpClientMockResponse:
        res = await responder(data) if inspect.iscoroutinefunction(responder) else responder(data)
        status = res.get("status", 200)
        json_data = res.get("json")
        text = res.get("text")
        headers = res.get("headers")
        exc_type = res.get("exc")
        exc = None
        if exc_type == "timeout":
            exc = TimeoutError()
        elif exc_type == "connect":
            exc = aiohttp.ClientConnectionError("Connection error")
        return AiohttpClientMockResponse(
            method,
            url,
            status=status,
            json=json_data,
            text=text,
            headers=headers,
            exc=exc,
        )

    aioclient_mock.post(f"{base_url}/chat/completions", side_effect=_side_effect)


@pytest.fixture
def inprocess_worker(monkeypatch: pytest.MonkeyPatch, hass: HomeAssistant) -> None:
    """Monkeypatch LearningSupervisor._async_run_stage to run worker in executor.

    The stage result goes through the same output contract as the real subprocess
    (worker.stage_outputs, then a JSON round trip as on stdout), so the supervisor sees
    exactly what `_async_run_stage` returns in production (WR-15).
    """
    import json

    from custom_components.powerguesser.engine.learning.worker import (
        run_stage_from_file,
        stage_outputs,
    )
    from custom_components.powerguesser.learning import LearningSupervisor

    def _run(stage: str, job_path: Any) -> dict[str, Any]:
        outputs = stage_outputs(stage, run_stage_from_file(stage, job_path))
        return json.loads(json.dumps(outputs))

    async def _mock_run_stage(self: LearningSupervisor, stage: str, job_path: Any) -> dict[str, Any]:
        outputs = await hass.async_add_executor_job(_run, stage, job_path)
        return {**outputs, "stats": {"wall_s": 0.0, "max_rss_kb": 0}}

    monkeypatch.setattr(LearningSupervisor, "_async_run_stage", _mock_run_stage)


async def async_fill_synth_buffer(
    hass: HomeAssistant,
    entry: ConfigEntry,
    spec: Any = None,
    end_ms: int | None = None,
) -> Any:
    """Fill buffer with synthetic data using channel IDs from writer.db."""
    from tests.fixtures.learning.synth import SynthSpec, fill_buffer
    import homeassistant.util.dt as dt_util

    if spec is None:
        spec = SynthSpec(days=22, washer=True)
    if end_ms is None:
        now = dt_util.now()
        today_midnight = dt_util.start_of_local_day(now)
        end_ms = int(round(today_midnight.timestamp() * 1000))

    def _fill() -> Any:
        db = entry.runtime_data.writer.db
        rows = db.channel_rows()
        channel_ids = {entity_id: ch for ch, entity_id, role, kind in rows}
        res = fill_buffer(db, channel_ids, spec, end_ms)
        return res

    res = await hass.async_add_executor_job(_fill)
    if getattr(entry.runtime_data, "learning", None) is not None:
        entry.runtime_data.learning.invalidate_usable_days_cache()
    return res


async def async_seed_llm(
    hass: HomeAssistant,
    entry: ConfigEntry,
    base_url: str = "http://llm.test/v1",
    api_key: str = "sk-test-secret",
    model: str = "m1",
    probe: dict[str, Any] | None = None,
) -> None:
    """Seed entry.data with LLM configuration and capability probe."""
    import homeassistant.util.dt as dt_util

    if probe is None:
        probe = {
            "strategy": "json_schema",
            "token_param": "max_completion_tokens",
            "send_temperature": True,
            "model_returned": model,
            "probed_at": dt_util.utcnow().isoformat(),
        }
    new_data = {
        **entry.data,
        "llm_base_url": base_url,
        "llm_api_key": api_key,
        "llm_model": model,
        "llm_probe": probe,
    }
    hass.config_entries.async_update_entry(entry, data=new_data)
    await hass.async_block_till_done()


async def async_seed_consent(
    hass: HomeAssistant,
    host: str = "llm.test",
    names: bool = False,
    base_url: str | None = None,
) -> None:
    """Grant consent in supervisor's consent store.

    Consent is keyed on scheme + host + port (WR-11); without base_url the grant is for
    http://<host>/v1, the shape async_seed_llm uses by default.
    """
    entries = hass.config_entries.async_loaded_entries(DOMAIN)
    assert entries, "No PowerGuesser entry loaded"
    supervisor = entries[0].runtime_data.learning
    await supervisor.consent_store.async_grant(base_url or f"http://{host}/v1", names)


@pytest.fixture
def kettle_plug_prefs(std_prefs: dict[str, Any]) -> dict[str, Any]:
    """Standard preferences plus a kettle plug in device_consumption."""
    import copy

    prefs = copy.deepcopy(std_prefs)
    prefs["device_consumption"].append(
        {
            "stat_consumption": "sensor.kettle_energy",
            "stat_rate": "sensor.kettle_power",
        }
    )
    return prefs


async def async_seed_shadow_set(
    hass: HomeAssistant,
    entry: ConfigEntry,
    artefacts: list[dict[str, Any]],
    links: list[dict[str, Any]],
    *,
    train_mean_w: float = 12.5,
    created: str | None = None,
) -> dict[str, Any]:
    """Build a v1 shadow set (powerguesser.shadow contract) and store it.

    Channel ids, grid_s and enabled_event_types come from the live buffer DB
    and profile (executor); train_cutoff_ms is the start of the 7th-last local
    day; created is the set's ISO UTC creation time (utcnow when None). Plans
    05-02 to 05-06 reuse this helper.
    """
    from datetime import timedelta
    from pathlib import Path

    import homeassistant.util.dt as dt_util

    from custom_components.powerguesser.engine.buffer import KIND_METERED
    from custom_components.powerguesser.engine.validation.constants import (
        SHADOW_HOLDOUT_DAYS,
        SHADOW_SCHEMA_VERSION,
    )

    def _build() -> dict[str, Any]:
        job = entry.runtime_data.learning.build_window_job(Path("seed-shadow"))
        channel_rows = entry.runtime_data.writer.db.channel_rows()
        day_bounds_ms = job["day_bounds_ms"]
        metered = [
            (ch, entity_id)
            for ch, entity_id, _role, kind in channel_rows
            if kind == KIND_METERED
        ]
        eligible = sorted(entity_id for _ch, entity_id in metered)
        ch_of = {entity_id: ch for ch, entity_id in metered}
        n_days = len(day_bounds_ms) - 1
        first_holdout = max(0, n_days - SHADOW_HOLDOUT_DAYS)
        linked_entities = {link["entity_id"] for link in links}
        not_discovered = [
            {
                "entity_id": entity_id,
                "channel": ch_of[entity_id],
                "n_train_sessions": 0,
            }
            for entity_id in eligible
            if entity_id not in linked_entities
        ]
        full_links = [
            {**link, "channel": ch_of.get(link["entity_id"], 0)} for link in links
        ]
        created_iso = created or dt_util.utcnow().isoformat()
        return {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "created": created_iso,
            "run_id": "test-shadow",
            "grid_s": job["grid_s"],
            "enabled_event_types": list(job["enabled_event_types"]),
            "eligible": eligible,
            "train_cutoff_ms": int(day_bounds_ms[first_holdout]),
            "holdout_days": SHADOW_HOLDOUT_DAYS,
            "artefacts": [dict(a) for a in artefacts],
            "links": full_links,
            "not_discovered": not_discovered,
            "plugs": {
                entity_id: {"train_mean_w": train_mean_w} for entity_id in eligible
            },
        }

    shadow = await hass.async_add_executor_job(_build)
    await entry.runtime_data.validation.async_store_shadow(shadow)
    return shadow


def seed_validation_report(
    hass_storage: dict[str, Any],
    report: dict[str, Any] | None,
    plug_classes: dict[str, str] | None = None,
) -> None:
    """Seed the powerguesser.validation Store in its v1 shape (Plan 05-04).

    report is the stored latest report (ok or refused) or None for "never
    run"; plug_classes is the stored class map. Plans 05-04..05-06 reuse this
    helper.
    """
    hass_storage[VALIDATION_STORE_KEY] = {
        "version": VALIDATION_STORES_VERSION,
        "minor_version": 1,
        "key": VALIDATION_STORE_KEY,
        "data": {"report": report, "plug_classes": plug_classes or {}},
    }


def make_switched_candidate(
    sig_id: str,
    label: str,
    rise_w: float = 2000.0,
    train_share: float = 0.05,
    val_share: float = 0.045,
    validated: bool = True,
    kind: str = "switched",
) -> dict[str, Any]:
    """Build a candidate dict (artefact + scores) for device-flow tests."""
    box = {"lo": rise_w * 0.9, "p50": rise_w, "hi": rise_w * 1.1}
    dur_box = {"lo": 60, "p50": 120, "hi": 300}
    state = {
        "name": "on",
        "rise_w": box,
        "fall_w": box,
        "dur_s": dur_box,
        "plateau_w": box,
        "phase_share": None,
    }
    return {
        "schema_version": 1,
        "signature_id": sig_id,
        "kind": kind,
        "label": label,
        "category": "appliance",
        "extraction": {"grid_s": 30, "edge_threshold_w": 120.0, "merge_samples": 2, "pair_tol": 0.25},
        "states": [state],
        "session": None,
        "time_prior": {"hour": [1 / 24] * 24, "dow": [1 / 7] * 7},
        "constraints": {"max_concurrent": 1},
        "provenance": {},
        "scores": {
            "train": {"share": train_share, "n": 20},
            "val": {"share": val_share, "n": 10} if val_share is not None else None,
            "validated": validated,
            "conflicts": 0,
        },
        "fingerprint": {
            "kind": kind,
            "states": [state],
        },
    }


def seed_candidates(
    hass_storage: dict[str, Any],
    candidates: list[dict[str, Any]],
    validated: bool = True,
) -> None:
    """Seed the candidates Store with the given candidates."""
    hass_storage[CANDIDATES_STORE_KEY] = {
        "version": LEARNING_STORES_VERSION,
        "minor_version": 1,
        "key": CANDIDATES_STORE_KEY,
        "data": {
            "run_id": "run-test-devices",
            "created": "2026-11-16T12:00:00Z",
            "window_end_ms": 1700000000000,
            "validated": validated,
            "candidates": candidates,
        },
    }


async def async_confirm_candidate(
    hass: HomeAssistant,
    entry: ConfigEntry,
    sig_id: str,
    name: str,
) -> dict[str, Any]:
    """Open Review, pick the candidate, choose Confirm and submit the name."""
    from homeassistant.data_entry_flow import FlowResultType

    res_review = await async_open_options_step(hass, entry, "review")
    assert res_review["type"] is FlowResultType.FORM
    res_cand = await hass.config_entries.options.async_configure(
        res_review["flow_id"], {"candidate": sig_id}
    )
    assert res_cand["type"] is FlowResultType.MENU
    assert res_cand["step_id"] == "candidate"
    res_form = await hass.config_entries.options.async_configure(
        res_cand["flow_id"], {"next_step_id": "confirm"}
    )
    assert res_form["type"] is FlowResultType.FORM
    assert res_form["step_id"] == "confirm"
    return await hass.config_entries.options.async_configure(
        res_form["flow_id"], {"name": name}
    )


def switched_cluster_ids(request_data: Any) -> list[str]:
    """Return the switched cluster ids of the digest in an LLM request body, in digest order.

    Session clusters sit between switched ones since CR-03, so tests that propose
    switched devices pick their clusters from this list instead of hard-coding c01/c02.
    """
    import json
    import re

    if isinstance(request_data, (bytes, bytearray)):
        request_data = request_data.decode("utf-8")
    req = json.loads(request_data) if isinstance(request_data, str) else request_data
    ids: list[str] = []
    for m in req.get("messages", []):
        content = m.get("content", "")
        if isinstance(content, str):
            for match in re.finditer(r"\bid=(c\d{2}) kind=switched\b", content):
                if match.group(1) not in ids:
                    ids.append(match.group(1))
    return ids


def proposal_responder(n_devices: int = 2) -> Any:
    """Return a mock_llm responder answering with a valid proposal for clusters in request."""
    import json
    import re
    from tests.fixtures.learning.fake_llm import chat_completion

    def _responder(request_data: Any) -> dict[str, Any]:
        if isinstance(request_data, (bytes, bytearray)):
            request_data = request_data.decode("utf-8")
        if isinstance(request_data, str):
            req_dict = json.loads(request_data)
        else:
            req_dict = request_data

        cluster_ids: list[str] = []
        cluster_kinds: dict[str, str] = {}
        messages = req_dict.get("messages", [])
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                for match in re.finditer(r"\bid=(c\d{2}) kind=(\w+)", content):
                    cluster_kinds[match.group(1)] = match.group(2)
                for match in re.finditer(r"\b(c\d{2})\b", content):
                    cid = match.group(1)
                    if cid not in cluster_ids:
                        cluster_ids.append(cid)

        dev_list = []
        for i, cid in enumerate(cluster_ids[:n_devices]):
            dev_list.append(
                {
                    "ref": f"d{i+1}",
                    "label": f"Device {i+1}",
                    "category": "unknown",
                    "tracker_kind": "program" if cluster_kinds.get(cid) == "session" else "switched",
                    "member_clusters": [cid],
                    "states": [
                        {
                            "name": "on",
                            "cluster_ref": cid,
                            "role": "main",
                        }
                    ],
                    "phase_hint": "single",
                    "confidence": "medium",
                    "rationale": f"Device from cluster {cid}.",
                    "ambiguous_with": [],
                    "reference_plug": "",
                }
            )

        proposal = {
            "analysis": "Generated proposal.",
            "schema_version": 1,
            "devices": dev_list,
            "rejected_clusters": [cid for cid in cluster_ids[n_devices:]],
            "merge_suggestions": [],
        }

        if "tools" in req_dict:
            return {"status": 200, "json": chat_completion(tool_arguments=json.dumps(proposal))}
        return {"status": 200, "json": chat_completion(content=json.dumps(proposal))}

    return _responder

