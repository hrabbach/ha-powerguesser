<!-- generated-by: gsd-doc-writer -->
# Contributing to PowerGuesser

Thanks for your interest in improving PowerGuesser, an open-source (Apache-2.0)
Home Assistant custom integration for NILM. This guide explains how work lands,
how to set up a development environment, and which checks your change must pass.

## How contributions land

Development happens on the maintainer's GitLab instance. The public GitHub
repository ([hrabbach/ha-powerguesser](https://github.com/hrabbach/ha-powerguesser))
is a **filtered export** of the public tree, rebuilt and pushed by CI on every
release tag (`ci/export_github.sh`), so the GitHub repo is not the landing path
for code — pushes to it are replaced by the next export.

- **Code changes land through GitLab merge requests (MRs).** A version-controlled
  pre-commit hook (`.githooks/pre-commit`) keeps direct commits to `main`
  docs-only.
- **CI runs on the merge request and must be green to merge.** Plain branch
  pushes — including direct commits to `main` — create no pipeline at all, so a
  change that skips the MR has no CI to gate on.
<!-- VERIFY: whether and how external contributors without GitLab access can submit code (e.g. patch attached to a GitHub issue) -->

## Development setup

1. Clone the repository and `cd` into it.
2. Enable the version-controlled git hooks (once per clone):

   ```bash
   git config core.hooksPath .githooks
   ```

3. Use Python **>= 3.14.2** (the minimum required by Home Assistant 2026.11.x).
4. Install the test dependencies the same way CI does:

   ```bash
   python3 -m pip install "pytest-homeassistant==2026.11.1" pytest-split
   ```

   The integration's own manifest requirements are installed under Home
   Assistant's constraint file — `sh ci/pytest.sh` does this for you before
   running the suite.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the integration is put
together and [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for its configuration
options.

## Running the checks locally

CI's validate stage runs on every MR. All of it can be reproduced locally:

| Check | What it does | Run locally |
| --- | --- | --- |
| `pytest` | Full test suite (sharded 4-way in CI via pytest-split) | `sh ci/pytest.sh` or `python3 -m pytest` |
| `hassfest` | Home Assistant integration validation | hassfest docker image (`ghcr.io/home-assistant/hassfest`) |
| `musl-gate` | Fails if manifest requirements go beyond `numpy`/`scipy` (Alpine/musl wheels) | `sh ci/musl_gate.sh` |
| `export-dry-run` | Builds and leak-scans the public tree | `sh ci/export_github.sh dry-run <dir>` |

Useful pytest invocations:

```bash
# Run the whole suite
python3 -m pytest

# Fast local loop: skip the slow subprocess tests
python3 -m pytest -m "not slow"
```

**Never set `LLM_BASE_URL`, `LLM_API_KEY` or `LLM_MODEL` when running the suite** —
tests replay recorded cassettes only, and CI fails the job outright if any `LLM_*`
variable is set.

Test layout: tests live in `tests/` (`tests/engine`, `tests/ha`, `tests/scripts`)
with shared fixtures in `tests/conftest.py` and `tests/fixtures`. Pytest
configuration (asyncio auto mode, markers, test paths) is in `pyproject.toml`.
Snapshot tests use syrupy; mark long-running tests `@pytest.mark.slow`.

## Coding standards

- **No auto-formatter or linter is configured.** Match the style of the
  surrounding code and Home Assistant integration conventions (async patterns,
  type hints).
- CI enforces correctness instead: hassfest validates the integration metadata,
  musl-gate keeps the dependency set installable on HA's Alpine base, and the
  pytest suite must pass.
- Keep the public tree clean: every pipeline dry-runs the GitHub export, which
  scans for tokens, private hostnames and internal-only paths.

## Merge request guidelines

- **Target branch:** `main`. No branch-naming convention is documented.
- Open an MR early — the pipeline runs on the MR, and the merge rule is
  **"Pipelines must succeed"**. A failed pytest shard can be retried on its own
  from the pipeline's job list.
- An **advisory AI code review** (the `code-review` job) may post review comments
  on your MR. It is allowed to fail and never blocks a merge, but treat its
  findings like a human reviewer's.
- **Direct commits to `main` are docs-only** — anything under `docs/` or
  `.planning/`, a top-level README/LICENSE/NOTICE, or any `*.md`. The
  `.githooks/pre-commit` hook refuses everything else. `ALLOW_MAIN_COMMIT=1` (or
  `git commit --no-verify`) is a deliberate one-time escape hatch, not the normal
  flow; merge commits landing an MR are always allowed.
- Write commit messages that explain *why* the change is needed; no formal
  conventional-commits requirement is documented.

## Issue reporting

<!-- VERIFY: GitHub Issues on hrabbach/ha-powerguesser is the intake channel for bug reports and feature requests -->
Please report bugs and request features via
[GitHub Issues](https://github.com/hrabbach/ha-powerguesser/issues). There are no
issue templates, so please include:

- PowerGuesser version and Home Assistant version.
- Steps to reproduce, what you expected, and what actually happened.
- Relevant log excerpts (**Settings → System → Logs**). For learning-phase
  problems, the output of the `powerguesser.preview_learning_payload` service
  helps — it shows the exact request body without the API key.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License, Version 2.0](LICENSE), the same license as the project (see also
[NOTICE](NOTICE)).
