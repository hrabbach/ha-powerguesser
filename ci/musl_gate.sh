#!/bin/sh
set -eu

# 1. Read manifest requirements with python3 and fail unless every requirement is numpy or scipy
python3 - << 'EOF'
import json, sys, re

with open("custom_components/powerguesser/manifest.json", "r", encoding="utf-8") as f:
    manifest = json.load(f)

reqs = manifest.get("requirements", [])
allowed = {"numpy", "scipy"}

for req in reqs:
    m = re.match(r"^([a-zA-Z0-9_\-\.]+)", req)
    if not m:
        print(f"ERROR: Cannot parse requirement '{req}'", file=sys.stderr)
        sys.exit(1)
    name = m.group(1).lower().replace("_", "-")
    if name not in allowed:
        print(f"ERROR: Forbidden runtime requirement '{name}' in manifest.json (DIST-02: only numpy and scipy allowed)", file=sys.stderr)
        sys.exit(1)

print(f"Verified manifest requirements: {reqs}")
EOF

# 2. Resolve HA's constraint file path
CONSTRAINT_FILE=$(python3 -c "import homeassistant.requirements as hr; print(hr.pip_kwargs()['constraints'])")
if [ ! -f "$CONSTRAINT_FILE" ]; then
    echo "ERROR: Constraint file $CONSTRAINT_FILE does not exist" >&2
    exit 1
fi
echo "Resolved HA constraints file: $CONSTRAINT_FILE"

# 3. Amd64 real install (D-24 part 1)
TARGET_DIR=$(mktemp -d /tmp/fresh_install_XXXXXX)
python3 -m uv pip install --quiet --index-strategy unsafe-first-match --upgrade --constraint "$CONSTRAINT_FILE" --target "$TARGET_DIR" $(python3 -c 'import json; print(" ".join(json.load(open("custom_components/powerguesser/manifest.json"))["requirements"]))')

PYTHONPATH="$TARGET_DIR" python3 -c 'import numpy, scipy, scipy.optimize, scipy.signal; print("Installed versions:", numpy.__version__, scipy.__version__)'
rm -rf "$TARGET_DIR"

# 4. Aarch64 resolve (D-24 part 2)
DRY_RUN_TARGET=$(mktemp -d /tmp/dry_run_XXXXXX)
AARCH64_OUTPUT=$(python3 -m uv pip install --dry-run --target "$DRY_RUN_TARGET" --python-platform aarch64-unknown-linux-musl --python-version 3.14 --only-binary :all: --index-strategy unsafe-first-match --constraint "$CONSTRAINT_FILE" $(python3 -c 'import json; print(" ".join(json.load(open("custom_components/powerguesser/manifest.json"))["requirements"]))') 2>&1)
rm -rf "$DRY_RUN_TARGET"

echo "$AARCH64_OUTPUT" | grep -qi "numpy" || { echo "ERROR: numpy not resolved in aarch64 output" >&2; exit 1; }
echo "$AARCH64_OUTPUT" | grep -qi "scipy" || { echo "ERROR: scipy not resolved in aarch64 output" >&2; exit 1; }

# 5. Final success confirmation
echo "MUSL GATE OK"
