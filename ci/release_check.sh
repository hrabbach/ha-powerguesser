#!/bin/sh
set -eu

# Release gate (Plan 06-02, D-17): fail unless the tree is a consistent release.
# Usage: sh ci/release_check.sh <tag>
#
# (a) the tag equals manifest.version exactly
# (b) hacs.json parses, name is a non-empty string, homeassistant is the minimum HA version
# (c) manifest documentation/issue_tracker are public GitHub URLs
# (d) LICENSE/NOTICE/README exist, the README carries the accuracy markers and an image
# (e) both brand icons have the right size
# (f) the release notes file docs/releases/<tag>.md exists
python3 - "$1" << 'EOF'
import json
import struct
import sys
from pathlib import Path
from urllib.parse import urlsplit

MANIFEST = Path("custom_components/powerguesser/manifest.json")
BRAND = Path("custom_components/powerguesser/brand")
HACS = Path("hacs.json")
README = Path("README.md")
LICENSE = Path("LICENSE")
NOTICE = Path("NOTICE")


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"{path} does not exist")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        fail(f"{path} is not valid JSON ({exc})")


def png_size(path):
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        fail(f"{path} does not exist")
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        fail(f"{path} does not start with the PNG signature")
    if data[12:16] != b"IHDR":
        fail(f"{path} has no IHDR chunk")
    return struct.unpack(">II", data[16:24])


tag = sys.argv[1] if len(sys.argv) > 1 else ""

# (a) the tag equals manifest.version exactly.
manifest = load_json(MANIFEST)
version = manifest.get("version")
if not isinstance(version, str) or not version:
    fail(f"{MANIFEST} has no version string")
if tag != version:
    fail(f"tag {tag!r} != manifest.version {version!r}")

# (b) hacs.json parses and carries name + the minimum HA version.
hacs = load_json(HACS)
if not isinstance(hacs.get("name"), str) or not hacs["name"]:
    fail(f"{HACS} name must be a non-empty string")
if hacs.get("homeassistant") != "2026.11.0":
    fail(f"{HACS} homeassistant must be 2026.11.0 (got {hacs.get('homeassistant')!r})")

# (c) the manifest links point at the public GitHub repository.
for key in ("documentation", "issue_tracker"):
    url = manifest.get(key)
    if not isinstance(url, str) or not url:
        fail(f"{MANIFEST} {key} is missing")
    hostname = urlsplit(url).hostname
    if hostname != "github.com":
        fail(f"{MANIFEST} {key} hostname must be github.com (got {hostname!r})")

# (d) licence, notice and the README accuracy block.
try:
    license_lines = LICENSE.read_text(encoding="utf-8").splitlines()
except FileNotFoundError:
    fail(f"{LICENSE} does not exist")
first_line = next((line.strip() for line in license_lines if line.strip()), "")
if first_line != "Apache License":
    fail(f"{LICENSE} first non-blank line must be Apache License (got {first_line!r})")
if not NOTICE.is_file():
    fail(f"{NOTICE} does not exist")
try:
    readme = README.read_text(encoding="utf-8")
except FileNotFoundError:
    fail(f"{README} does not exist")
if readme.count("<!-- accuracy-table:start -->") != 1:
    fail(f"{README} must contain exactly one accuracy-table:start marker")
if readme.count("<!-- accuracy-table:end -->") != 1:
    fail(f"{README} must contain exactly one accuracy-table:end marker")
if not any(line.startswith("![") for line in readme.splitlines()):
    fail(f"{README} must contain at least one Markdown image")

# (e) brand icons carry the PNG signature and the expected IHDR size.
for icon, expected in (("icon.png", (256, 256)), ("icon@2x.png", (512, 512))):
    size = png_size(BRAND / icon)
    if size != expected:
        fail(f"{BRAND / icon} must be {expected[0]}x{expected[1]} (got {size[0]}x{size[1]})")

# (f) the release notes for this tag exist (the GitHub release workflow reads them).
notes = Path("docs") / "releases" / f"{tag}.md"
if not notes.is_file():
    fail(f"{notes} does not exist")

print(f"release_check OK for tag {tag}")
EOF

# The README accuracy block must match the committed report (D-20, T-06-03).
python3 -I -S scripts/render_accuracy_table.py --check
