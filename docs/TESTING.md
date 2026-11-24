<!-- generated-by: gsd-doc-writer -->
# Testing

## Test framework and setup

The suite uses **pytest** (asyncio mode `auto`, per-loop fixture scope `function`)
driven by `pytest-homeassistant==2026.11.1`, the Home Assistant test harness that
provides the `hass` core, `MockConfigEntry`, `async_fire_time_changed`, recorder
mocks, and the `enable_custom_integrations` fixture. Snapshot assertions use
**syrupy** (amber `.ambr` files), and CI sharding uses **pytest-split**. All of this
is configured in `pyproject.toml` — there is no separate pytest config file, and
`pythonpath = ["."]` plus `testpaths = ["tests"]` make `pytest` from the repository
root "just work".

Setup requirements:

- **Python >= 3.14.2** (matching Home Assistant 2026.11.1's `requires-python`;
  `ci/pytest.sh` fails fast on anything older).
- `pytest-homeassistant==2026.11.1` and `pytest-split`:
  `python3 -m pip install "pytest-homeassistant==2026.11.1" pytest-split`
- The integration's manifest requirements (`numpy`, `scipy>=1.16.2,<2`) installed
  under the Home Assistant constraint file.

Note that `tests/conftest.py` imports `homeassistant.*`, so **every** test — even
the HA-free engine tests under `tests/engine/` — needs a working Home Assistant
environment to collect. The engine itself is kept HA-free by an AST guard test,
not by a separate venv.

All LLM behavior is tested from **cassettes** (redacted JSON fixtures under
`tests/fixtures/llm/`, replayed via `load_llm_cassette`/`mock_llm` in
`tests/conftest.py`). No test makes a live LLM call, and `ci/pytest.sh` hard-fails
if any `LLM_*` variable is set in the environment.

## Suites

The suite (~100 test files) is split into three directories by concern:

- **`tests/engine/`** — pure engine tests. They import only
  `custom_components/powerguesser/engine/` (the HA-free computation layer) and
  shared helpers; nothing under this tree should touch `homeassistant.*`.
  Sub-packages mirror the engine layout:
  - `detect/` — attribution, joint/online detection, replay/stream parity, tick budgets
  - `learning/` — episodes, digests, proposals, identity, refits, worker subprocess
  - `validation/` — classes, drift, linking, metrics, overlap, reports, stage validation
  - Top-level files cover the SQLite buffer, intervals, prefs model, self-check,
    settle logic, and the bootstrap rows.
- **`tests/ha/`** — real Home Assistant flows through `pytest-homeassistant`:
  config/options flows, consent, buffer writing, learning schedule/caps, device
  detection/drift/merge/proposals/sensors, validation service, diagnostics,
  repairs issues, LLM client and replay, plug sessions, options screens. Includes
  the syrupy characterization snapshot
  `tests/ha/__snapshots__/test_llm_rounds_characterization.ambr`.
- **`tests/scripts/`** — tests for the developer tooling in `scripts/`
  (`llm_spike.py`, `tune_offline.py`, `render_accuracy_table.py`) and for
  `ci/export_github.sh`. They load the scripts via `importlib` from their real
  paths rather than importing the integration.

`tests/fixtures/` holds shared data: a reference day CSV, preference variants,
learning fixtures, LLM cassettes, and trace/program generators.

## Markers

Two markers matter:

- **`slow`** — registered in `pyproject.toml`; used on tests that spawn the real
  learning-worker subprocess (seconds per test). Deselect them for a fast local
  loop: `pytest -m 'not slow'`. Used in `tests/engine/learning/test_worker_subprocess.py`,
  `tests/ha/test_learning_run.py`, `tests/ha/test_learning_worker_contract.py`,
  `tests/ha/test_device_detection.py`, and `tests/scripts/test_llm_spike.py`.
- **`bootstrap_enabled`** — registered in `tests/conftest.py`; `bootstrap` is the
  only code path that schedules the recorder bootstrap, and tests opt in with
  this marker instead of paying its cost by default.

## Running tests

```bash
# Full suite (uses the committed shard timings for reporting)
pytest

# Fast local loop — skip the slow worker-subprocess tests
pytest -m 'not slow'

# One suite
pytest tests/engine

# One file or one test
pytest tests/ha/test_sensor.py
pytest tests/ha/test_sensor.py::test_remainder_clamping_and_over_attribution_attribute

# Regenerate a syrupy snapshot after an intentional output change
pytest --snapshot-update
```

`syrupy` fails the session when a snapshot belongs to a deselected test, so any
**sharded** run (CI or local experimentation with `--splits`) must pass
`--snapshot-warn-unused`. The committed `ci/.test_durations` file feeds
pytest-split's `least_duration` algorithm; the convenient wrapper is:

```bash
# One CI-equivalent shard locally (normally driven by CI_NODE_TOTAL/CI_NODE_INDEX)
sh ci/pytest.sh
```

## Writing new tests

- File naming: `test_*.py` colocated with the concern — engine logic under
  `tests/engine/<area>/`, HA-touching behavior under `tests/ha/`, script tooling
  under `tests/scripts/`.
- Async tests need no marker: `asyncio_mode = "auto"` in `pyproject.toml` handles
  `async def` tests automatically.
- Shared fixtures live in `tests/conftest.py` — `energy_manager` (sets up the
  Energy component with default preferences), `std_prefs`, `hass_config_dir`,
  `async_set_power` (sets a power entity state with the right attributes), and
  `async_setup_via_flow` (runs the real config flow). The autouse
  `disable_bootstrap_schedule` and `auto_enable_custom_integrations` fixtures
  keep tests cheap; opt into bootstrap only via `@pytest.mark.bootstrap_enabled`.
- Keep the engine boundary: anything importing `homeassistant.*` belongs in
  `tests/ha/`, and new engine imports are caught by
  `tests/engine/test_no_ha_imports.py` (AST scan of the whole engine package).
- LLM interactions must be cassette-based — extend `tests/fixtures/llm/` and the
  `load_llm_cassette`/`mock_llm` helpers; never add a live-endpoint test.

## Coverage requirements

No coverage plugin or threshold is configured — there is no `--cov`, no
`pytest-cov` dependency, and no coverage gate in CI. Correctness is enforced by
characterization snapshots and the boundary/guard tests instead.

## CI integration

CI is **GitLab** (`.gitlab-ci.yml`). Pipelines run only for merge requests,
version tags (`x.y.z`), and manual/web- or schedule-triggered runs — plain branch
pushes create no pipeline.

The `pytest` job (stage `validate`, image `python:3.14`) runs `sh ci/pytest.sh`
with **`parallel: 4`**: GitLab sets `CI_NODE_TOTAL=4` / `CI_NODE_INDEX=1..4` and
pytest-split executes that shard using the committed `ci/.test_durations`
timings and `--snapshot-warn-unused`. A single failed shard can be retried on its
own. The job has a 20-minute timeout; with current timings each shard takes
roughly 9–10.5 minutes including the ~1.5-minute install.

After any large change to the suite's shape, run the manual `pytest-durations`
job (`STORE_DURATIONS=1`, 45-minute timeout): it does one full unsplit pass,
writes a fresh `ci/.test_durations` artifact (kept for one week), which you then
download and commit so the four shards stay balanced.

Two sibling validation jobs also gate merges: `hassfest` (HA integration lint)
and `musl-gate` (verifies the manifest stays limited to `numpy`/`scipy` and that
both resolve for aarch64 musl). The release pipeline additionally requires
`export-dry-run` and `release-check` before tagging.
