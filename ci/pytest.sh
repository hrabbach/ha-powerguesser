#!/bin/sh
set -eu

# (a) Guard: exit 1 when any LLM variable is set (D-21: cassettes only)
if [ -n "${LLM_BASE_URL:-}" ] || [ -n "${LLM_API_KEY:-}" ] || [ -n "${LLM_MODEL:-}" ]; then
    echo "ERROR: LLM_* variables must not be set in CI (D-21: cassettes only)" >&2
    exit 1
fi

# (b) Python version check: fail unless python3 is at least 3.14.2 (HA 2026.11.1 requires-python)
python3 -c '
import sys
if sys.version_info < (3, 14, 2):
    print(f"ERROR: Python >= 3.14.2 required, found {sys.version}", file=sys.stderr)
    sys.exit(1)
'

# (c) Environment configuration for pip
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_ROOT_USER_ACTION=ignore
export PYTHONDONTWRITEBYTECODE=1

# (d) Install pytest-homeassistant==2026.11.1 and pytest-split (CI sharding)
python3 -m pip install --quiet "pytest-homeassistant==2026.11.1" "pytest-split"

# (e) Resolve HA constraint file and install manifest requirements under it
CONSTRAINT_FILE=$(python3 -c "import homeassistant.requirements as hr; print(hr.pip_kwargs()['constraints'])")
if [ ! -f "$CONSTRAINT_FILE" ]; then
    echo "ERROR: Constraint file $CONSTRAINT_FILE does not exist" >&2
    exit 1
fi

MANIFEST_REQS=$(python3 -c 'import json; print(" ".join(json.load(open("custom_components/powerguesser/manifest.json"))["requirements"]))')
python3 -m pip install --quiet --constraint "$CONSTRAINT_FILE" $MANIFEST_REQS

# (f) Print versions
python3 -V
python3 -c '
import importlib.metadata as im
for pkg in ["homeassistant", "pytest-homeassistant", "pytest-split", "numpy", "scipy"]:
    print(f"{pkg}: {im.version(pkg)}")
'

# (g) Run the suite, preserving the exit code (WR-15).
#
# CI runs this job as `parallel: 4`, so GitLab sets CI_NODE_TOTAL=4 and
# CI_NODE_INDEX=1..4; pytest-split then selects this shard's tests. Running with
# STORE_DURATIONS=1 (the manual pytest-durations job) instead does one full,
# unsplit pass and writes ci/.test_durations for balanced sharding.
#
# least_duration spreads heavy tests across shards instead of clustering them in
# a trailing contiguous chunk. --snapshot-warn-unused is required when sharding:
# syrupy fails the session when a snapshot belongs to a deselected test (every
# shard would otherwise exit 1 with all its tests passing). Off CI (no CI_NODE_*),
# SPLITS=1 runs everything.
SPLITS="${CI_NODE_TOTAL:-1}"
GROUP="${CI_NODE_INDEX:-1}"
DURATIONS_PATH="${PYTEST_DURATIONS_PATH:-ci/.test_durations}"

if [ "${STORE_DURATIONS:-0}" = "1" ]; then
    exec python3 -m pytest -q -p no:cacheprovider \
        --store-durations --durations-path "$DURATIONS_PATH"
fi

exec python3 -m pytest -q -p no:cacheprovider \
    --splits "$SPLITS" --group "$GROUP" \
    --splitting-algorithm least_duration \
    --snapshot-warn-unused \
    --durations-path "$DURATIONS_PATH"
