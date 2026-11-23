"""Proposal validation, parsing, sanitisation, and salvage for LLM output."""

from collections.abc import Mapping, Sequence
import re
import unicodedata

from ..llm_wire import extract_json
from .prompts import CATEGORIES, REJECT_REASONS, TRACKER_KINDS

MAX_DEVICES = 40
LABEL_MAX = 40
ANALYSIS_MAX = 1200
RATIONALE_MAX = 300
REASON_MAX = 200

_FORBIDDEN_LABEL_CHARS = frozenset("`*_[]()<>#|~!\\")
_PHASE_HINTS = frozenset({"single", "multi_balanced", "unknown"})
_CONFIDENCES = frozenset({"high", "medium", "low"})
_ROLES = frozenset({"main", "auxiliary", "transient"})
# Digest cluster kind each tracker kind is compiled and replayed from (CR-03)
_KIND_FOR_TRACKER: dict[str, str] = {"switched": "switched", "program": "session"}

# Over-long human-readable text is cut locally instead of costing a repair turn
# (03-SPIKE.md follow-up 7: the provider ignores the caps even under strict json_schema).
# Only these free-text fields are trimmed; ids, enums, numbers and structure never are.
TRIM_MARK = "…"
_ZWJ = "‍"
_ROOT_FREE_TEXT: dict[str, int] = {"analysis": ANALYSIS_MAX}
_ITEM_FREE_TEXT: dict[str, dict[str, int]] = {
    "devices": {"label": LABEL_MAX, "rationale": RATIONALE_MAX},
    "merge_suggestions": {"reason": REASON_MAX},
}


def _extends_previous(ch: str) -> bool:
    """True when ch belongs to the preceding character (combining mark, joiner, selector, skin tone)."""
    return (
        unicodedata.category(ch) in ("Mn", "Mc", "Me")
        or ch == _ZWJ
        or "︀" <= ch <= "️"
        or "\U0001f3fb" <= ch <= "\U0001f3ff"
    )


def trim_text(s: str, cap: int) -> str:
    """Cut s to at most cap code points, ending in TRIM_MARK, without splitting a character sequence.

    Text within the cap is returned unchanged. Otherwise the cut moves back so a base
    character is never separated from its combining marks or a ZWJ emoji sequence,
    trailing whitespace is removed, and TRIM_MARK is appended inside the cap.
    """
    if len(s) <= cap:
        return s
    if cap < len(TRIM_MARK):
        return ""
    end = cap - len(TRIM_MARK)
    while end > 0 and (_extends_previous(s[end]) or s[end - 1] == _ZWJ):
        end -= 1
    return s[:end].rstrip() + TRIM_MARK


def trim_free_text(obj: object) -> tuple[object, list[str]]:
    """Return a copy of a proposal with over-long free-text fields trimmed, plus the sorted trimmed paths.

    Trimmed: /analysis, /devices/N/label, /devices/N/rationale, /merge_suggestions/N/reason.
    Anything that is not a string in the expected place is left for validate_proposal to report.
    The input is not mutated.
    """
    if not isinstance(obj, dict):
        return obj, []

    out = dict(obj)
    trimmed: list[str] = []

    for key, cap in _ROOT_FREE_TEXT.items():
        value = out.get(key)
        if isinstance(value, str) and len(value) > cap:
            out[key] = trim_text(value, cap)
            trimmed.append(f"/{key}")

    for list_key, fields in _ITEM_FREE_TEXT.items():
        items = out.get(list_key)
        if not isinstance(items, list):
            continue
        new_items: list[object] = []
        for idx, item in enumerate(items):
            if isinstance(item, dict):
                new_item = item
                for field, cap in fields.items():
                    value = item.get(field)
                    if isinstance(value, str) and len(value) > cap:
                        if new_item is item:
                            new_item = dict(item)
                        new_item[field] = trim_text(value, cap)
                        trimmed.append(f"/{list_key}/{idx}/{field}")
                new_items.append(new_item)
            else:
                new_items.append(item)
        out[list_key] = new_items

    return out, sorted(trimmed)


def sanitize_label(s: str) -> str:
    """NFC-normalise, strip control chars and formatting symbols, collapse whitespace, cut to 40 code points."""
    if not isinstance(s, str):
        return "unknown"

    norm = unicodedata.normalize("NFC", s)
    filtered: list[str] = []
    for ch in norm:
        if unicodedata.category(ch).startswith("C"):
            continue
        if ch in _FORBIDDEN_LABEL_CHARS:
            continue
        filtered.append(ch)

    cleaned = "".join(filtered)
    collapsed = re.sub(r"\s+", " ", cleaned).strip()
    result = collapsed[:LABEL_MAX]
    return result if result else "unknown"


def validate_proposal(
    obj: object,
    *,
    cluster_ids: Sequence[str],
    plug_ids: Sequence[str],
    cluster_kinds: Mapping[str, str] | None = None,
) -> tuple[dict | None, list[str]]:
    """Validate proposal object strictly against schema, length caps, and semantic references.

    cluster_kinds maps a digest cluster id to its kind ("switched" or "session"); when given,
    a "program" device may use only session clusters and a "switched" device only switched
    clusters, because compile and replay treat the two episode kinds separately.
    """
    errors: list[tuple[str, str]] = []

    if not isinstance(obj, dict):
        return None, ["/: expected JSON object"]

    # Check root keys
    expected_root_keys = {
        "analysis",
        "schema_version",
        "devices",
        "rejected_clusters",
        "merge_suggestions",
    }
    actual_root_keys = set(obj.keys())
    missing_root = expected_root_keys - actual_root_keys
    for k in sorted(missing_root):
        errors.append((f"/{k}", "missing required property"))
    extra_root = actual_root_keys - expected_root_keys
    for k in sorted(extra_root):
        errors.append((f"/{k}", "additional property not allowed"))

    # schema_version
    if "schema_version" in obj:
        sv = obj["schema_version"]
        if not isinstance(sv, int) or isinstance(sv, bool) or sv != 1:
            errors.append(("/schema_version", "expected integer 1"))

    # analysis
    if "analysis" in obj:
        ana = obj["analysis"]
        if not isinstance(ana, str):
            errors.append(("/analysis", "expected string"))
        elif len(ana) > ANALYSIS_MAX:
            errors.append(("/analysis", f"exceeds maximum length of {ANALYSIS_MAX} characters"))

    cluster_id_set = set(cluster_ids)
    plug_id_set = set(plug_ids)

    # Track devices and cluster usages
    device_refs: set[str] = set()
    cluster_to_devices: dict[str, list[tuple[int, str, list[str]]]] = {}
    used_device_clusters: set[str] = set()

    devices = obj.get("devices")
    if devices is not None:
        if not isinstance(devices, list):
            errors.append(("/devices", "expected array"))
        else:
            if len(devices) > MAX_DEVICES:
                errors.append(("/devices", f"exceeds maximum of {MAX_DEVICES} devices"))

            expected_dev_keys = {
                "ref",
                "label",
                "category",
                "tracker_kind",
                "member_clusters",
                "states",
                "phase_hint",
                "confidence",
                "rationale",
                "ambiguous_with",
                "reference_plug",
            }

            for idx, dev in enumerate(devices):
                dev_path = f"/devices/{idx}"
                if not isinstance(dev, dict):
                    errors.append((dev_path, "expected object"))
                    continue

                actual_dev_keys = set(dev.keys())
                for k in sorted(expected_dev_keys - actual_dev_keys):
                    errors.append((f"{dev_path}/{k}", "missing required property"))
                for k in sorted(actual_dev_keys - expected_dev_keys):
                    errors.append((f"{dev_path}/{k}", "additional property not allowed"))

                # ref
                d_ref = dev.get("ref")
                if not isinstance(d_ref, str) or not d_ref:
                    errors.append((f"{dev_path}/ref", "expected non-empty string"))
                else:
                    if d_ref in device_refs:
                        errors.append((f"{dev_path}/ref", f"duplicate device ref '{d_ref}'"))
                    device_refs.add(d_ref)

                # label
                d_label = dev.get("label")
                if not isinstance(d_label, str):
                    errors.append((f"{dev_path}/label", "expected string"))
                elif len(d_label) > LABEL_MAX:
                    errors.append((f"{dev_path}/label", f"exceeds maximum length of {LABEL_MAX}"))

                # category
                d_cat = dev.get("category")
                if d_cat not in CATEGORIES:
                    errors.append((f"{dev_path}/category", f"must be one of {list(CATEGORIES)}"))

                # tracker_kind
                d_kind = dev.get("tracker_kind")
                if d_kind not in TRACKER_KINDS:
                    errors.append((f"{dev_path}/tracker_kind", f"must be one of {list(TRACKER_KINDS)}"))

                # member_clusters
                d_members = dev.get("member_clusters")
                member_set: set[str] = set()
                if not isinstance(d_members, list):
                    errors.append((f"{dev_path}/member_clusters", "expected array of strings"))
                else:
                    for m_idx, c_ref in enumerate(d_members):
                        m_path = f"{dev_path}/member_clusters/{m_idx}"
                        if not isinstance(c_ref, str):
                            errors.append((m_path, "expected string"))
                        elif c_ref not in cluster_id_set:
                            errors.append((m_path, f"cluster '{c_ref}' not in input digest"))
                        else:
                            member_set.add(c_ref)
                            used_device_clusters.add(c_ref)
                            c_kind = (cluster_kinds or {}).get(c_ref)
                            if c_kind is not None and d_kind in _KIND_FOR_TRACKER:
                                want = _KIND_FOR_TRACKER[d_kind]
                                if c_kind != want:
                                    errors.append((
                                        m_path,
                                        f"cluster '{c_ref}' is a {c_kind} cluster; "
                                        f"tracker_kind '{d_kind}' needs {want} clusters",
                                    ))

                # states
                d_states = dev.get("states")
                if not isinstance(d_states, list):
                    errors.append((f"{dev_path}/states", "expected array of state objects"))
                else:
                    expected_state_keys = {"name", "cluster_ref", "role"}
                    for s_idx, st in enumerate(d_states):
                        s_path = f"{dev_path}/states/{s_idx}"
                        if not isinstance(st, dict):
                            errors.append((s_path, "expected object"))
                            continue
                        actual_st_keys = set(st.keys())
                        for sk in sorted(expected_state_keys - actual_st_keys):
                            errors.append((f"{s_path}/{sk}", "missing required property"))
                        for sk in sorted(actual_st_keys - expected_state_keys):
                            errors.append((f"{s_path}/{sk}", "additional property not allowed"))

                        if not isinstance(st.get("name"), str):
                            errors.append((f"{s_path}/name", "expected string"))

                        st_cref = st.get("cluster_ref")
                        if not isinstance(st_cref, str):
                            errors.append((f"{s_path}/cluster_ref", "expected string"))
                        else:
                            if st_cref not in cluster_id_set:
                                errors.append((f"{s_path}/cluster_ref", f"cluster '{st_cref}' not in input digest"))
                            if isinstance(d_members, list) and st_cref not in member_set:
                                errors.append((f"{s_path}/cluster_ref", f"cluster '{st_cref}' not in device member_clusters"))

                        st_role = st.get("role")
                        if st_role not in _ROLES:
                            errors.append((f"{s_path}/role", f"must be one of {sorted(_ROLES)}"))

                # phase_hint
                if dev.get("phase_hint") not in _PHASE_HINTS:
                    errors.append((f"{dev_path}/phase_hint", f"must be one of {sorted(_PHASE_HINTS)}"))

                # confidence
                if dev.get("confidence") not in _CONFIDENCES:
                    errors.append((f"{dev_path}/confidence", f"must be one of {sorted(_CONFIDENCES)}"))

                # rationale
                d_rat = dev.get("rationale")
                if not isinstance(d_rat, str):
                    errors.append((f"{dev_path}/rationale", "expected string"))
                elif len(d_rat) > RATIONALE_MAX:
                    errors.append((f"{dev_path}/rationale", f"exceeds maximum length of {RATIONALE_MAX}"))

                # ambiguous_with
                d_ambig = dev.get("ambiguous_with")
                ambig_list: list[str] = []
                if not isinstance(d_ambig, list):
                    errors.append((f"{dev_path}/ambiguous_with", "expected array of strings"))
                else:
                    for a_idx, a_ref in enumerate(d_ambig):
                        a_path = f"{dev_path}/ambiguous_with/{a_idx}"
                        if not isinstance(a_ref, str):
                            errors.append((a_path, "expected string"))
                        elif a_ref not in cluster_id_set:
                            errors.append((a_path, f"cluster '{a_ref}' not in input digest"))
                        else:
                            ambig_list.append(a_ref)

                # reference_plug
                d_plug = dev.get("reference_plug")
                if not isinstance(d_plug, str):
                    errors.append((f"{dev_path}/reference_plug", "expected string"))
                elif d_plug != "" and d_plug not in plug_id_set:
                    errors.append((f"{dev_path}/reference_plug", f"plug '{d_plug}' not in input digest"))

                # Record cluster usages for ambiguous_with checking
                if isinstance(d_members, list):
                    for c_ref in member_set:
                        cluster_to_devices.setdefault(c_ref, []).append((idx, str(d_ref), ambig_list))

    # Cross-device cluster ambiguity check
    for c_ref, usages in cluster_to_devices.items():
        if len(usages) > 1:
            # Check if every device that uses c_ref lists all other devices in ambiguous_with
            # Wait, rule: "a cluster listed in two devices is valid only when each device lists it in ambiguous_with;
            # otherwise both devices get an error"
            for dev_idx, dev_ref, ambig in usages:
                # Is c_ref in ambiguous_with of this device?
                # "each device lists it in ambiguous_with"
                if c_ref not in ambig:
                    errors.append((
                        f"/devices/{dev_idx}/ambiguous_with",
                        f"shared cluster '{c_ref}' must be listed in ambiguous_with",
                    ))

    # rejected_clusters
    rejected_set: set[str] = set()
    rejected = obj.get("rejected_clusters")
    if rejected is not None:
        if not isinstance(rejected, list):
            errors.append(("/rejected_clusters", "expected array"))
        else:
            expected_rej_keys = {"cluster_ref", "reason"}
            for r_idx, rej in enumerate(rejected):
                r_path = f"/rejected_clusters/{r_idx}"
                if not isinstance(rej, dict):
                    errors.append((r_path, "expected object"))
                    continue
                actual_rej_keys = set(rej.keys())
                for k in sorted(expected_rej_keys - actual_rej_keys):
                    errors.append((f"{r_path}/{k}", "missing required property"))
                for k in sorted(actual_rej_keys - expected_rej_keys):
                    errors.append((f"{r_path}/{k}", "additional property not allowed"))

                c_ref = rej.get("cluster_ref")
                if not isinstance(c_ref, str):
                    errors.append((f"{r_path}/cluster_ref", "expected string"))
                elif c_ref not in cluster_id_set:
                    errors.append((f"{r_path}/cluster_ref", f"cluster '{c_ref}' not in input digest"))
                else:
                    if c_ref in rejected_set:
                        errors.append((f"{r_path}/cluster_ref", f"duplicate rejected cluster '{c_ref}'"))
                    rejected_set.add(c_ref)
                    if c_ref in used_device_clusters:
                        errors.append((f"{r_path}/cluster_ref", f"cluster '{c_ref}' is also used by a device"))

                reason = rej.get("reason")
                if reason not in REJECT_REASONS:
                    errors.append((f"{r_path}/reason", f"must be one of {list(REJECT_REASONS)}"))

    # merge_suggestions
    merges = obj.get("merge_suggestions")
    if merges is not None:
        if not isinstance(merges, list):
            errors.append(("/merge_suggestions", "expected array"))
        else:
            expected_merge_keys = {"clusters", "reason"}
            for m_idx, mg in enumerate(merges):
                m_path = f"/merge_suggestions/{m_idx}"
                if not isinstance(mg, dict):
                    errors.append((m_path, "expected object"))
                    continue
                actual_mg_keys = set(mg.keys())
                for k in sorted(expected_merge_keys - actual_mg_keys):
                    errors.append((f"{m_path}/{k}", "missing required property"))
                for k in sorted(actual_mg_keys - expected_merge_keys):
                    errors.append((f"{m_path}/{k}", "additional property not allowed"))

                m_clusters = mg.get("clusters")
                if not isinstance(m_clusters, list):
                    errors.append((f"{m_path}/clusters", "expected array of strings"))
                else:
                    for c_idx, c_ref in enumerate(m_clusters):
                        c_path = f"{m_path}/clusters/{c_idx}"
                        if not isinstance(c_ref, str):
                            errors.append((c_path, "expected string"))
                        elif c_ref not in cluster_id_set:
                            errors.append((c_path, f"cluster '{c_ref}' not in input digest"))

                m_reason = mg.get("reason")
                if not isinstance(m_reason, str):
                    errors.append((f"{m_path}/reason", "expected string"))
                elif len(m_reason) > REASON_MAX:
                    errors.append((f"{m_path}/reason", f"exceeds maximum length of {REASON_MAX}"))

    # Sort errors by (path, message)
    errors.sort(key=lambda item: (item[0], item[1]))
    formatted_errors = [f"{path}: {msg}" for path, msg in errors]

    if formatted_errors:
        return None, formatted_errors
    return obj, []


def parse_proposal(
    message: object,
    *,
    cluster_ids: Sequence[str],
    plug_ids: Sequence[str],
    cluster_kinds: Mapping[str, str] | None = None,
) -> tuple[dict | None, list[str], list[str]]:
    """Extract JSON from message, trim over-long free text, and validate against digest cluster/plug IDs.

    Returns (proposal or None, errors, trimmed paths). Trimming is a soft note, not an
    error; every other violation is returned as an error for the repair turn.
    """
    obj, _ = extract_json(message)
    if obj is None:
        return None, ["/: no JSON object found"], []
    obj, trimmed = trim_free_text(obj)
    valid, errors = validate_proposal(
        obj, cluster_ids=cluster_ids, plug_ids=plug_ids, cluster_kinds=cluster_kinds
    )
    return valid, errors, trimmed


def salvage(
    obj: object,
    errors: Sequence[str],
    *,
    cluster_ids: Sequence[str],
    plug_ids: Sequence[str],
    cluster_kinds: Mapping[str, str] | None = None,
) -> tuple[dict, list[str]] | None:
    """Drop offending devices/rejections/merges pointed to by error paths, re-validate, and return surviving proposal.

    Over-long free text is trimmed first, so a length error alone never costs an entry.
    """
    obj, trimmed = trim_free_text(obj)
    if not isinstance(obj, dict):
        return None
    trimmed_paths = set(trimmed)
    errors = [e for e in errors if e.split(":", 1)[0].strip() not in trimmed_paths]

    bad_device_indices: set[int] = set()
    bad_rejected_indices: set[int] = set()
    bad_merge_indices: set[int] = set()

    for err in errors:
        path = err.split(":", 1)[0].strip()
        m_dev = re.match(r"^/devices/(\d+)", path)
        if m_dev:
            bad_device_indices.add(int(m_dev.group(1)))
        m_rej = re.match(r"^/rejected_clusters/(\d+)", path)
        if m_rej:
            bad_rejected_indices.add(int(m_rej.group(1)))
        m_mrg = re.match(r"^/merge_suggestions/(\d+)", path)
        if m_mrg:
            bad_merge_indices.add(int(m_mrg.group(1)))

    devices = obj.get("devices", [])
    if not isinstance(devices, list):
        return None

    surviving_devices: list[dict] = []
    unresolved_clusters: list[str] = []

    for idx, dev in enumerate(devices):
        if idx in bad_device_indices:
            if isinstance(dev, dict) and isinstance(dev.get("member_clusters"), list):
                for c in dev["member_clusters"]:
                    if isinstance(c, str):
                        unresolved_clusters.append(c)
        else:
            surviving_devices.append(dev)

    if not surviving_devices:
        return None

    rejected = obj.get("rejected_clusters", [])
    surviving_rejected = [
        rej for idx, rej in enumerate(rejected)
        if idx not in bad_rejected_indices
    ] if isinstance(rejected, list) else []

    merges = obj.get("merge_suggestions", [])
    surviving_merges = [
        mg for idx, mg in enumerate(merges)
        if idx not in bad_merge_indices
    ] if isinstance(merges, list) else []

    candidate = {
        "analysis": obj.get("analysis", ""),
        "schema_version": obj.get("schema_version", 1),
        "devices": surviving_devices,
        "rejected_clusters": surviving_rejected,
        "merge_suggestions": surviving_merges,
    }

    valid_prop, re_errors = validate_proposal(
        candidate, cluster_ids=cluster_ids, plug_ids=plug_ids, cluster_kinds=cluster_kinds
    )
    if valid_prop is None:
        return None

    # Deduplicate unresolved clusters preserving order
    seen: set[str] = set()
    dedup_unresolved: list[str] = []
    for c in unresolved_clusters:
        if c not in seen:
            seen.add(c)
            dedup_unresolved.append(c)

    return valid_prop, dedup_unresolved
