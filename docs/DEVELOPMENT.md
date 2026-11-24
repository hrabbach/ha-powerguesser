<!-- generated-by: gsd-doc-writer -->
# Development

How to set up a working development environment, run the tests, and land a change.
PowerGuesser is a pure-Python Home Assistant custom integration — there is no build step,
no package to install, and no bundler. The "build" is the test suite plus a set of CI
gate scripts.

## Repository layout

```
custom_components/powerguesser/   The integration (the only runtime artifact)
  ├─ *.py                         Home Assistant layer: config_flow, hub, sensor,
  │                               devices, learning, validation, llm_client, ...
  ├─ engine/                      HA-free computation package (never imports
  │   │                           homeassistant; also runs in the worker process)
  │   ├─ detect/                  Online detector (residual attribution)
  │   ├─ learning/                Learning pipeline stages + worker.py
  │   └─ validation/              Holdout validation, drift, metrics, reports
  ├─ manifest.json                Integration metadata; requirements: numpy, scipy
  ├─ strings.json + translations/ UI text
  └─ brand/                       Icons (checked by the release gate)
tests/
  ├─ conftest.py                  Shared fixtures and helpers (see "Test helpers")
  ├─ engine/                      Engine tests — plain pytest, no hass fixture
  ├─ ha/                          Integration tests — pytest-homeassistant hass fixture
  ├─ scripts/                     Tests for the dev scripts in scripts/
  └─ fixtures/                    Synthetic data, Energy prefs variants, LLM cassettes
scripts/                          Dev-only tools (never imported by the integration):
                                  llm_spike.py, tune_offline.py, render_accuracy_table.py
ci/                               CI shell scripts + .test_durations (shard timings)
.githooks/                        Version-controlled git hooks (pre-commit)
.github/workflows/                Workflows for the public GitHub mirror
docs/                             Documentation, release notes (releases/), validation report
pyproject.toml                    pytest configuration only (not a build file)
hacs.json                         Marketplace/HACS metadata (min HA 2026.11.0)
```

The hard architectural boundary is `engine/`: it must stay free of `homeassistant`
imports so it can run in a separate low-priority worker process and be tested without a
running Home Assistant. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full picture.

## Local setup

Prerequisites:

- **Python >= 3.14.2** (Home Assistant 2026.11.1's `requires-python`; `ci/pytest.sh`
  refuses anything older).
- **Linux or WSL** for the test runner. The maintainer runs tests in WSL
  (AlmaLinux-10), not native Windows — `pytest-homeassistant` needs Python 3.14 and the
  shell-based CI gates assume a POSIX environment.
- **Docker** (optional) for running the CI gate scripts locally exactly as CI does.

1. Clone the repository and create a venv with the HA test package (exact pin — it tracks
   HA core releases):

   ```bash
   uv venv ~/.venvs/powerguesser --python 3.14
   ~/.venvs/powerguesser/bin/python -m pip install pytest-homeassistant==2026.11.1 freezegun
   ```

2. Install the integration's runtime requirements (`numpy`, `scipy>=1.16.2,<2` from
   `manifest.json`) under Home Assistant's own constraint file, mirroring what CI does:

   ```bash
   CONSTRAINT_FILE=$(~/.venvs/powerguesser/bin/python -c \
     "import homeassistant.requirements as hr; print(hr.pip_kwargs()['constraints'])")
   ~/.venvs/powerguesser/bin/python -m pip install --constraint "$CONSTRAINT_FILE" \
     numpy "scipy>=1.16.2,<2"
   ```

3. Enable the version-controlled git hooks (see "Branch conventions"):

   ```bash
   git config core.hooksPath .githooks
   ```

4. Sanity check — run the fast half of the suite:

   ```bash
   ~/.venvs/powerguesser/bin/pytest -q -m 'not slow'
   ```

No environment variables are needed for development, and **no LLM endpoint is required**:
every LLM HTTP call in the test suite is mocked. In fact `ci/pytest.sh` fails immediately
if `LLM_BASE_URL`, `LLM_API_KEY`, or `LLM_MODEL` are set — unset them before running the
CI script locally.

**Line endings:** `.gitattributes` forces LF for `*.sh`. If your Windows checkout uses
`core.autocrlf`, strip carriage returns before running the CI shell scripts (the local
proof commands below do this with `sed 's/\r$//'`).

## Running tests

pytest is configured in `pyproject.toml` (`asyncio_mode = "auto"`, `testpaths = ["tests"]`,
`pythonpath = ["."]`, `addopts = "-p no:respx"`).

| Command | What it runs |
| --- | --- |
| `pytest -q` | Full suite (~35 min unsharded; CI shards it 4 ways) |
| `pytest -q -m 'not slow'` | Fast local loop — deselects the `slow` marker (tests that spawn the real worker subprocess, seconds each) |
| `pytest tests/engine -q` | Engine-only tests — plain pytest, no `hass` fixture needed |
| `pytest tests/ha/test_setup.py -q` | A single test file |
| `sh ci/pytest.sh` | The CI entry point (installs deps, then runs pytest; unsharded when `CI_NODE_TOTAL` is unset) |

Test layout conventions:

- **`tests/engine/`** mirrors `engine/` and never touches Home Assistant.
  `tests/engine/test_no_ha_imports.py` AST-scans the engine package and fails on any
  `homeassistant` import or any import of the integration package — keep it green.
- **`tests/ha/`** uses the `hass` fixture from `pytest-homeassistant` for config-flow,
  sensor, learning, and validation behaviour. Snapshot tests use syrupy
  (`tests/ha/__snapshots__/*.ambr`).
- **`tests/scripts/`** covers the dev tools in `scripts/`.
- **LLM traffic is always mocked.** Use `mock_llm(aioclient_mock, base_url, responder)`
  with canned responses from `tests/fixtures/llm/*.json` (`load_llm_cassette`), or the
  `proposal_responder` helper that answers with a valid device proposal for whatever
  clusters the request contains.
- The `slow` marker is declared in `pyproject.toml`; CI runs slow tests too (in the
  shards), so only deselect them locally.

Test helpers in `tests/conftest.py` worth knowing before writing a new test:

- `energy_manager` / `std_prefs` — set up the Energy component with a standard
  grid + solar + battery + one metered device preference set.
- `async_setup_via_flow(hass)` — run the config flow and return the loaded `ConfigEntry`.
- `async_fill_synth_buffer(hass, entry, spec)` — fill the SQLite buffer with synthetic
  days of data from `tests/fixtures/learning/synth.py` (`SynthSpec`).
- `inprocess_worker` — run learning worker stages in-process (through the same output
  contract as the real subprocess) so most tests never spawn a process.
- `async_seed_llm` / `async_seed_consent` / `seed_candidates` / `seed_validation_report` —
  seed the LLM config, consent record, and the learning/validation stores.

## Build and check commands

There is no compile step. The commands below are the project's "build": the gates CI
runs, all of which can be run locally.

| Command | Description |
| --- | --- |
| `pytest -q` (venv) | Full test suite |
| `sh ci/musl_gate.sh` | Dependency gate: manifest requirements limited to numpy/scipy, real amd64 install under HA constraints, and an `aarch64-unknown-linux-musl` dry-run resolve (HAOS has no compiler, so only pure wheels are allowed) |
| `sh ci/export_github.sh dry-run <dir>` | Build the public tree from the export allowlist, leak-scan it, and print the file list |
| `sh ci/release_check.sh <tag>` | Release gate: tag == `manifest.version`, `hacs.json`, brand icons, README accuracy markers, release notes exist (plus the accuracy-table drift check below) |
| `python3 -I -S scripts/render_accuracy_table.py --check` | Fail if the README accuracy block drifted from `docs/validation/accuracy-report.json` (`--write` regenerates it) |
| `python3 scripts/tune_offline.py <subcommand>` | Offline tuning harness — opens a copy of a real buffer read-only and prints measurements; dev-only, assigns no engine constants |
| `python3 scripts/llm_spike.py <subcommand>` | Probe an OpenAI-compatible endpoint / build preview payloads without Home Assistant; dev-only |

Local proof of the two Docker-based CI gates (from Git Bash or WSL, strips CR line
endings from the Windows checkout):

```bash
# musl gate, in the real HA container image
MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/repo" -w /repo \
  --entrypoint sh ghcr.io/home-assistant/home-assistant:2026.11.1 ci/musl_gate.sh

# CI pytest job, in the same image CI uses
MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W):/repo" -w /repo \
  python:3.14 sh -c "sed 's/\r$//' ci/pytest.sh > /tmp/pytest.sh && sh /tmp/pytest.sh"
```

## Code style and conventions

- **Home Assistant conventions are the baseline**: fully async, no blocking I/O in the
  event loop (file/SQLite work goes through the executor), config flow only (no YAML
  configuration), UI text in `strings.json` + `translations/`.
- **Type annotations and docstrings throughout** — every module and helper carries both;
  match the surrounding file when editing.
- **The engine stays HA-free.** No `homeassistant` imports, no imports of the
  integration package, relative-import levels kept inside the package. Enforced by
  `tests/engine/test_no_ha_imports.py`.
- **Runtime dependencies are frozen to numpy + scipy.** HAOS containers have no C
  compiler and no musllinux wheels exist for scikit-learn/onnxruntime/etc., so a new
  runtime dependency would break installs. Enforced by `ci/musl_gate.sh`. Dev-only
  tooling (scripts/, tests/) may use more, but never under `custom_components/`.
- **No network in tests.** LLM calls go through `aioclient_mock` and recorded cassettes;
  CI runs with no LLM variables set and fails if any are present.
- **Lint/format:** no linter or formatter config is committed and CI has no lint job.
  Ruff and mypy are used locally (unpinned; `.ruff_cache/` is gitignored) with Home
  Assistant's own style as the reference. The merge-request code review (see below) is
  the de facto style gate.

## Branch conventions

- **`main` is the base branch.** All work lands on it through merge requests.
- **Direct code commits to `main` are blocked by a pre-commit hook**
  (`.githooks/pre-commit`, enabled with `git config core.hooksPath .githooks`). Allowed
  directly on `main`: documentation-only changes — anything under `docs/`, any `*.md`
  file, top-level `README`/`LICENSE`/`NOTICE`, and the internal planning tree. Merge
  commits are always allowed. One-time override: `ALLOW_MAIN_COMMIT=1 git commit ...`
  (or `git commit --no-verify`).
- **No branch naming convention is documented.** Name branches however you like; the MR
  targets `main`.
- Plain branch pushes — including direct commits to `main` — create **no CI pipeline**.
  CI runs on merge requests, version tags, and manual/web/scheduled runs, so land work
  through an MR to get a pipeline.

## Merge request process

1. Push a branch and open a merge request targeting `main`.
2. The MR pipeline runs automatically (see "CI pipeline" below). The project merge rule
   is **"Pipelines must succeed"** — a green pipeline is required to merge.
3. An **advisory AI code review** job also runs on MRs that touch code paths
   (`custom_components/`, `tests/`, `scripts/`, `ci/`, `.githooks/`, CI config). It posts
   review comments on the MR; it is `allow_failure: true` and never blocks a merge. For
   docs-only MRs it is available as a manual job.
4. Address review feedback, keep the pipeline green, and merge. Squash merges work, but
   note the pre-commit hook: `git merge --squash` writes no `MERGE_HEAD`, so committing
   the squashed result locally on `main` needs the override above.

Development happens on the project's GitLab. The public GitHub repository
(`hrabbach/ha-powerguesser`) is a **read-only export mirror**: on each version tag, CI
publishes the allowlisted public tree there (see `ci/export_github.sh`). File issues on
GitHub; expect code review and merges on GitLab.

## CI pipeline

CI is GitLab (`.gitlab-ci.yml`), with two small GitHub Actions workflows on the public
mirror. Pipelines run for merge requests, version tags (`x.y.z`), and web/scheduled
triggers — nothing else. All jobs are interruptible except the two release jobs, so a
newer pipeline on the same ref auto-cancels a superseded one.

Stages: `review` → `validate` → `release`.

| Job | Stage | What it does |
| --- | --- | --- |
| `code-review` | review | MR-only, advisory (`allow_failure: true`). Runs a pinned, checksum-verified LLM review tool over the MR diff and posts the comments back. Credentials come from masked CI variables; every failure path is loud and posts nothing. |
| `hassfest` | validate | Official `ghcr.io/home-assistant/hassfest` image — Home Assistant's integration validator. |
| `musl-gate` | validate | `ci/musl_gate.sh` inside the HA 2026.11.1 container image: manifest requirements limited to numpy/scipy, real amd64 install under HA's constraint file, aarch64-musl dry-run resolve. |
| `pytest` | validate | `python:3.14` image, `parallel: 4` shards via pytest-split (`ci/pytest.sh`), 20 min timeout per shard. Shard balance comes from the committed `ci/.test_durations`. |
| `pytest-durations` | validate | Manual job that reruns the suite unsplit and uploads a fresh `ci/.test_durations` artifact. Run it after a large change in the suite's shape, then download and commit the file. |
| `export-dry-run` | validate | `ci/export_github.sh dry-run` — builds and leak-scans the public tree on every pipeline, so a leak is caught on the branch, not at release time. |
| `release-check` | validate | Tag pipelines only (`interruptible: false`). `ci/release_check.sh "$CI_COMMIT_TAG"` — the release gate. |
| `export-github` | release | Tag pipelines only (`interruptible: false`), after all validate jobs. `ci/export_github.sh publish` pushes the public tree and tag to the GitHub mirror. |

On the GitHub mirror, `.github/workflows/validate.yml` runs `hacs/action` and `hassfest`
on pushes to `main`, PRs, a nightly schedule, and manual dispatch; `release.yml` runs on
version tags — `hacs/action` + `hassfest`, then the release gate, then creates the
GitHub Release from `docs/releases/<tag>.md`.

## Releasing

Releasing is tag-driven and gated end to end; the full runbook (preconditions, the
go-ahead gate, tag push, and GitHub-side confirmation) lives in
[docs/releases/RELEASING.md](releases/RELEASING.md). In short: bump
`custom_components/powerguesser/manifest.json`, add `docs/releases/<tag>.md`, land the
change through a green-pipeline MR, then tag. The tag pipeline re-runs every validate
job plus `release-check` before `export-github` publishes to the mirror, where the
GitHub Release workflow takes over.

## Next steps

- [ARCHITECTURE.md](ARCHITECTURE.md) — system overview, component diagram, and the
  engine/HA-layer boundary you must respect when writing code.
- [CONFIGURATION.md](CONFIGURATION.md) — how the integration is configured (config flow,
  options flow, stores) — useful context for config-flow and learning tests.
- [docs/releases/RELEASING.md](releases/RELEASING.md) — the release runbook.
