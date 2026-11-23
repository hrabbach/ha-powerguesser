"""Tests for learning run service, subprocess execution, lifecycle, and cleanup."""

import asyncio
import json
import logging
from pathlib import Path
import pytest

from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError

from custom_components.powerguesser.const import (
    BUFFER_DIR,
    DOMAIN,
    LEARNING_DIR,
    LEARNING_STORE_KEY,
    CANDIDATES_STORE_KEY,
    CONSENT_STORE_KEY,
    REVIEW_STORE_KEY,
    LEARNING_STORES_VERSION,
)
from custom_components.powerguesser import learning
from homeassistant.helpers.storage import Store


def _set_std_powers(hass: HomeAssistant) -> None:
    from tests.conftest import async_set_power
    async_set_power(hass, "sensor.grid_power", 500)
    async_set_power(hass, "sensor.solar_power", 1200)
    async_set_power(hass, "sensor.battery_power", -300)
    async_set_power(hass, "sensor.washer_power", 200)


@pytest.mark.slow
async def test_learn_service_runs_real_worker_to_candidates(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test full learning run through real worker subprocess to ranked candidates."""
    from tests.conftest import (
        async_fill_synth_buffer,
        async_seed_consent,
        async_seed_llm,
        async_setup_via_flow,
        mock_llm,
        proposal_responder,
    )

    caplog.set_level(logging.INFO)
    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)

    mock_llm(aioclient_mock, "http://llm.test/v1", proposal_responder(2))

    await hass.services.async_call(DOMAIN, "learn", blocking=True)
    supervisor = entry.runtime_data.learning
    assert supervisor.run_task is not None
    await supervisor.run_task

    data = supervisor.coordinator.data
    assert data["status"] == "idle"
    assert data["last_error"] is None
    assert len(supervisor.candidates) >= 1
    assert data["pending_candidates"] == len(supervisor.candidates)

    last_run = data["last_run"]
    assert last_run is not None
    assert "days" in last_run
    assert "train" in last_run["days"]
    assert "val" in last_run["days"]
    assert "dropped" in last_run["days"]

    # Check stage files
    run_id = last_run["run_id"]
    stage_dir = Path(hass.config.path(BUFFER_DIR, LEARNING_DIR, run_id))
    assert (stage_dir / "digest.json").exists()
    assert (stage_dir / "request_1.json").exists()
    assert (stage_dir / "proposal_1.json").exists()
    assert (stage_dir / "library_1.json").exists()

    # Verify request body matched HTTP call
    request_data = json.loads((stage_dir / "request_1.json").read_text(encoding="utf-8"))
    assert len(aioclient_mock.mock_calls) >= 1
    sent_body = aioclient_mock.mock_calls[0][2]
    if isinstance(sent_body, (bytes, bytearray)):
        sent_body = sent_body.decode("utf-8")
    if isinstance(sent_body, str):
        sent_body = json.loads(sent_body)
    assert sent_body == request_data

    # Check INFO logging for both stages
    assert "Learning stage extract finished in" in caplog.text
    assert "Learning stage compile_score finished in" in caplog.text


async def test_learn_raises_when_not_loaded(hass: HomeAssistant) -> None:
    """Test calling powerguesser.learn raises when no entry is loaded."""
    from custom_components.powerguesser.const import SERVICE_LEARN

    # Register services via async_setup
    from custom_components.powerguesser import async_setup
    assert await async_setup(hass, {})

    with pytest.raises(ServiceValidationError) as exc_info:
        await hass.services.async_call(DOMAIN, SERVICE_LEARN, blocking=True)
    assert exc_info.value.translation_key == "not_loaded"


async def test_second_trigger_while_running_raises(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    aioclient_mock: object,
    inprocess_worker: None,
) -> None:
    """Test triggering a run while another is active raises already_running."""
    from tests.conftest import (
        async_fill_synth_buffer,
        async_seed_consent,
        async_seed_llm,
        async_setup_via_flow,
        mock_llm,
        proposal_responder,
    )

    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)

    unblock_event = asyncio.Event()
    base_responder = proposal_responder(2)

    async def _waiting_responder(data: object) -> dict:
        await unblock_event.wait()
        return base_responder(data)

    mock_llm(aioclient_mock, "http://llm.test/v1", _waiting_responder)

    # Trigger first run
    await hass.services.async_call(DOMAIN, "learn", blocking=True)
    supervisor = entry.runtime_data.learning
    assert supervisor.run_task is not None
    assert not supervisor.run_task.done()

    # Trigger second run
    with pytest.raises(ServiceValidationError) as exc_info:
        await hass.services.async_call(DOMAIN, "learn", blocking=True)
    assert exc_info.value.translation_key == "already_running"

    # Unblock first run and let it finish
    unblock_event.set()
    await supervisor.run_task
    assert supervisor.coordinator.data["status"] == "idle"


async def test_unload_kills_running_worker(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test unloading integration kills running worker subprocess."""
    from tests.conftest import (
        async_fill_synth_buffer,
        async_seed_consent,
        async_seed_llm,
        async_setup_via_flow,
    )

    sleep_script = tmp_path / "fake_worker.py"
    sleep_script.write_text(
        "import sys, time\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(learning, "WORKER_ENTRY", sleep_script)

    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)

    await hass.services.async_call(DOMAIN, "learn", blocking=True)
    supervisor = entry.runtime_data.learning

    # Wait for process to spawn
    for _ in range(50):
        if supervisor._proc is not None:
            break
        await asyncio.sleep(0.1)

    proc = supervisor._proc
    assert proc is not None
    assert proc.returncode is None

    # Unload entry
    assert await hass.config_entries.async_unload(entry.entry_id)

    # Process must be killed within 5 s
    for _ in range(50):
        if proc.returncode is not None:
            break
        await asyncio.sleep(0.1)

    assert proc.returncode is not None


async def test_stage_timeout_names_stage_and_stderr(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test worker timeout names stage, timeout seconds, and stderr tail."""
    from tests.conftest import (
        async_fill_synth_buffer,
        async_seed_consent,
        async_seed_llm,
        async_setup_via_flow,
    )

    timeout_script = tmp_path / "timeout_worker.py"
    timeout_script.write_text(
        "import sys, time\n"
        "sys.stderr.write('worker-stderr-marker\\n')\n"
        "sys.stderr.flush()\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(learning, "WORKER_ENTRY", timeout_script)
    monkeypatch.setattr(learning, "STAGE_TIMEOUT_S", 1)

    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)
    await async_fill_synth_buffer(hass, entry)
    await async_seed_llm(hass, entry)
    await async_seed_consent(hass, host="llm.test", names=False)

    await hass.services.async_call(DOMAIN, "learn", blocking=True)
    supervisor = entry.runtime_data.learning
    await supervisor.run_task

    coord_data = supervisor.coordinator.data
    assert coord_data["status"] == "failed"
    last_error = coord_data["last_error"]
    assert last_error is not None
    assert "extract" in last_error
    assert "timed out after 1 s" in last_error
    assert "worker-stderr-marker" in last_error

    proc = supervisor._proc
    assert proc is not None
    assert proc.returncode is not None


async def test_remove_entry_deletes_learning_data(
    hass: HomeAssistant,
    energy_manager: object,
    std_prefs: dict,
) -> None:
    """Test removing config entry deletes learning directories and stores."""
    from tests.conftest import async_setup_via_flow

    energy_manager.data = std_prefs
    _set_std_powers(hass)
    entry = await async_setup_via_flow(hass)

    learning_dir = Path(hass.config.path(BUFFER_DIR, LEARNING_DIR))
    dummy_file = learning_dir / "test_run" / "test.json"
    dummy_file.parent.mkdir(parents=True, exist_ok=True)
    dummy_file.write_text("{}", encoding="utf-8")

    store = Store(hass, LEARNING_STORES_VERSION, LEARNING_STORE_KEY)
    await store.async_save({"test": True})

    assert dummy_file.exists()

    await hass.config_entries.async_remove(entry.entry_id)

    assert not learning_dir.exists()
