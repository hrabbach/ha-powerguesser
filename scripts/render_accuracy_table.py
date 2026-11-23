#!/usr/bin/env python3
"""Render the README accuracy block from the committed validation report (Plan 06-02, D-20).

CLAIMS RULE (D-20, SC4): the README states a figure for a device class only
when that class passes the P5 D-08 bar in the committed report. Every other
vocabulary row reads "not validated". Nothing here is hand-written prose: the
block is generated, and ci/release_check.sh runs --check so a hand-edited claim
fails the release (T-06-03).

Stdlib only. The script inserts custom_components/powerguesser on sys.path and
imports the class vocabulary from the HA-free engine package, so it runs under
``python3 -I -S`` with no site-packages (an accidental numpy or homeassistant
import fails the release gate, review concern L64@8b3be9c).

Usage:
    render_accuracy_table.py                          # print the block
    render_accuracy_table.py --write                  # rewrite README between the markers
    render_accuracy_table.py --check                  # exit 1 on drift
    render_accuracy_table.py --redact-from raw.json --out docs/validation/accuracy-report.json
"""

from collections.abc import Mapping, Sequence
import argparse
import json
from pathlib import Path
import sys

# --- engine vocabulary (stdlib + the HA-free engine package only) ---
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENGINE_PARENT = _REPO_ROOT / "custom_components" / "powerguesser"
if str(_ENGINE_PARENT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_PARENT))

from engine.validation.classes import BANDS  # noqa: E402
from engine.validation.constants import REPORT_SCHEMA_VERSION  # noqa: E402

# D-06 vocabularies. KINDS is module-level so a test can compute the row count
# from it and BANDS together, never from a typed literal (review L68@8b3be9c).
KINDS = ("switched", "program")
LEVELS = ("1_level", "multi_level")

KIND_LABELS = {"switched": "Switched load", "program": "Program appliance"}
BAND_LABELS = {
    "under_0.5kw": "under 0.5 kW",
    "0.5_1.5kw": "0.5-1.5 kW",
    "1.5_3kw": "1.5-3 kW",
    "over_3kw": "over 3 kW",
}
LEVEL_LABELS = {"1_level": "single level", "multi_level": "multi level"}

COLUMNS = (
    "Class",
    "Description",
    "Status",
    "Passing peers",
    "Median energy error %",
    "Max energy error %",
    "Holdout events",
)

START_MARKER = "<!-- accuracy-table:start -->"
END_MARKER = "<!-- accuracy-table:end -->"

DEFAULT_REPORT = "docs/validation/accuracy-report.json"
DEFAULT_README = "README.md"

# The fixed caveat below the table (D-20): one household, the D-08 "two passing
# peers" rule, and the per-device unvalidated flag.
CAVEAT = (
    "Figures come from one household's holdout validation (plug-masked shadow "
    "models replayed on the most recent 7 days). A class reads validated only "
    "when at least two plug devices of that class pass; other installations may "
    "differ, and every unvalidated device carries `unvalidated: true`."
)

# The redaction allowlist (D-20, T-06-04): class-level fields and header
# metadata only. No entity id, device name, hostname or peer list survives.
_HEADER_KEYS = ("schema_version", "status", "reason", "generated_at")
_HOLDOUT_KEYS = ("start", "end", "start_ms", "end_ms")
_CLASS_KEYS = (
    "class",
    "peers",
    "passing",
    "median_error_pct",
    "max_error_pct",
    "n_events",
    "passes",
)


class MarkerError(Exception):
    """The README does not carry both accuracy-table markers."""


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def load_report(path: str | Path) -> object | None:
    """Parse the report JSON, or None when it is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def _usable(report: object) -> bool:
    """True only for an ok report under the schema version this release reads."""
    return (
        isinstance(report, Mapping)
        and report.get("status") == "ok"
        and report.get("schema_version") == REPORT_SCHEMA_VERSION
    )


def reason_line(report: object) -> str | None:
    """A one-line reason why every class reads not validated, or None when usable."""
    if not isinstance(report, Mapping):
        return (
            "The accuracy report is missing or unreadable, so every class reads "
            "not validated."
        )
    status = report.get("status")
    if status != "ok":
        detail = report.get("reason")
        base = f"The accuracy report status is {status!r}, so every class reads not validated"
        return f"{base} ({detail})." if detail else f"{base}."
    if report.get("schema_version") != REPORT_SCHEMA_VERSION:
        return (
            "The accuracy report carries schema_version "
            f"{report.get('schema_version')!r} which this release does not read "
            f"(expected {REPORT_SCHEMA_VERSION}), so every class reads not validated."
        )
    return None


def _validated_entries(report: object) -> dict[str, Mapping]:
    """The report's class entries with passes true, keyed by class id (D-08, D-20).

    Only an ok report at REPORT_SCHEMA_VERSION can validate anything: a refused,
    absent or foreign-schema report validates no class.
    """
    if not _usable(report):
        return {}
    classes = report.get("classes")
    if not _is_sequence(classes):
        return {}
    out: dict[str, Mapping] = {}
    for entry in classes:
        if not isinstance(entry, Mapping) or entry.get("passes") is not True:
            continue
        class_id = entry.get("class")
        if isinstance(class_id, str) and class_id:
            out[class_id] = entry
    return out


def _cell(value: object) -> str:
    return "" if value is None else str(value)


def build_rows(report: object) -> list[dict[str, object]]:
    """One row per kind x band x level, with figures only for a validated class."""
    validated = _validated_entries(report)
    rows: list[dict[str, object]] = []
    for kind in KINDS:
        for band in BANDS:
            for levels in LEVELS:
                class_id = f"{kind}_{band}_{levels}"
                entry = validated.get(class_id)
                rows.append(
                    {
                        "class": class_id,
                        "description": (
                            f"{KIND_LABELS[kind]}, {BAND_LABELS[band]}, {LEVEL_LABELS[levels]}"
                        ),
                        "status": "validated" if entry is not None else "not validated",
                        "passing_peers": (
                            f"{_cell(entry.get('passing'))}/{_cell(entry.get('peers'))}"
                            if entry is not None
                            else ""
                        ),
                        "median_error_pct": (entry.get("median_error_pct") if entry else None),
                        "max_error_pct": (entry.get("max_error_pct") if entry else None),
                        "n_events": (entry.get("n_events") if entry else None),
                    }
                )
    return rows


def render_table(report: object) -> str:
    """The Markdown table: header, separator, then every vocabulary row."""
    lines = [
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    for row in build_rows(report):
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["class"]),
                    str(row["description"]),
                    str(row["status"]),
                    str(row["passing_peers"]),
                    _cell(row["median_error_pct"]),
                    _cell(row["max_error_pct"]),
                    _cell(row["n_events"]),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def render_block(report: object) -> str:
    """The full block that goes between the README markers: reason, table, caveat."""
    parts: list[str] = []
    reason = reason_line(report)
    if reason:
        parts.append(reason)
        parts.append("")
    parts.append(render_table(report))
    parts.append("")
    parts.append(CAVEAT)
    return "\n".join(parts)


def _marker_span(text: str) -> tuple[int, int]:
    start = text.find(START_MARKER)
    if start < 0:
        raise MarkerError(f"{START_MARKER} not found")
    end = text.find(END_MARKER, start + len(START_MARKER))
    if end < 0:
        raise MarkerError(f"{END_MARKER} not found")
    return start + len(START_MARKER), end


def current_block(text: str) -> str:
    """The text currently between the two markers."""
    inner_start, end = _marker_span(text)
    return text[inner_start:end]


def apply_block(text: str, block: str) -> str:
    """Replace only the text between the markers; everything else is untouched."""
    inner_start, end = _marker_span(text)
    return text[:inner_start] + "\n" + block + "\n" + text[end:]


def redact_report(raw: object) -> dict[str, object]:
    """Keep only the allowlisted header and class-level fields (T-06-04).

    Keeps schema_version, status, reason, generated_at, the holdout dates,
    thresholds, and per class only class, peers, passing, median_error_pct,
    max_error_pct, n_events and passes. Entity ids, device names and peer
    lists are dropped.
    """
    out: dict[str, object] = {}
    if not isinstance(raw, Mapping):
        return out
    for key in _HEADER_KEYS:
        if key in raw:
            out[key] = raw[key]

    holdout = raw.get("holdout")
    if isinstance(holdout, Mapping):
        out["holdout"] = {k: holdout[k] for k in _HOLDOUT_KEYS if k in holdout}

    thresholds = raw.get("thresholds")
    if isinstance(thresholds, Mapping):
        out["thresholds"] = dict(thresholds)

    classes: list[dict[str, object]] = []
    raw_classes = raw.get("classes")
    if _is_sequence(raw_classes):
        for entry in raw_classes:
            if isinstance(entry, Mapping):
                classes.append({k: entry[k] for k in _CLASS_KEYS if k in entry})
    out["classes"] = classes
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the README accuracy block from the validation report (D-20)."
    )
    parser.add_argument("--report", default=DEFAULT_REPORT, help="validation report JSON")
    parser.add_argument("--readme", default=DEFAULT_README, help="README to rewrite or check")
    parser.add_argument("--write", action="store_true", help="rewrite the README block")
    parser.add_argument("--check", action="store_true", help="exit 1 when the README block drifts")
    parser.add_argument("--redact-from", metavar="RAW", help="redact a raw report JSON")
    parser.add_argument("--out", metavar="PATH", help="destination for --redact-from")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.redact_from:
        if not args.out:
            sys.stderr.write("ERROR: --redact-from requires --out\n")
            return 1
        raw = load_report(args.redact_from)
        if raw is None:
            sys.stderr.write(f"ERROR: cannot read {args.redact_from}\n")
            return 1
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(redact_report(raw), indent=2) + "\n", encoding="utf-8"
        )
        return 0

    if args.write and args.check:
        sys.stderr.write("ERROR: --write and --check are mutually exclusive\n")
        return 1

    block = render_block(load_report(args.report))

    if args.write or args.check:
        path = Path(args.readme)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            sys.stderr.write(f"ERROR: cannot read {path} ({exc})\n")
            return 1
        if args.write:
            try:
                updated = apply_block(text, block)
            except MarkerError as exc:
                sys.stderr.write(f"ERROR: {path}: {exc}\n")
                return 1
            path.write_text(updated, encoding="utf-8")
            return 0
        try:
            present = current_block(text)
        except MarkerError as exc:
            sys.stderr.write(f"ERROR: {path}: {exc}\n")
            return 1
        if present.strip() != block.strip():
            sys.stderr.write(
                f"ERROR: {path} accuracy block is out of date with "
                f"{args.report}; run scripts/render_accuracy_table.py --write\n"
            )
            return 1
        return 0

    sys.stdout.write(block + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
