"""Tests for LLM proposal validation, parsing, sanitisation, and salvage."""

import json
import pytest

from custom_components.powerguesser.engine.learning.proposal import (
    ANALYSIS_MAX,
    EXAMPLE_VERDICTS,
    LABEL_MAX,
    MAX_DEVICES,
    RATIONALE_MAX,
    REASON_MAX,
    TRIM_MARK,
    parse_proposal,
    pop_example_verdicts,
    salvage,
    sanitize_label,
    trim_free_text,
    trim_text,
    validate_proposal,
)
from tests.fixtures.learning.fake_llm import chat_completion, proposal_for


@pytest.fixture
def sample_digest():
    return {
        "header": {
            "grid_s": 30,
            "days": 14,
            "residual_kwh": 120.0,
            "clustered_share": 0.85,
            "edge_threshold_w": 120,
            "phases_mapped": False,
            "plugs": 1,
        },
        "clusters": [
            {
                "id": "c01",
                "kind": "switched",
                "n": 25,
                "rise_w": {"p10": 1950, "p50": 2000, "p90": 2050},
                "fall_w": {"p10": -2050, "p50": -2000, "p90": -1950},
                "dur_min": {"p10": 1.5, "p50": 2.5, "p90": 3.5},
                "hour": [0] * 24,
                "dow": [0] * 7,
                "energy_share": 0.25,
            },
            {
                "id": "c02",
                "kind": "switched",
                "n": 15,
                "rise_w": {"p10": 1400, "p50": 1500, "p90": 1600},
                "fall_w": {"p10": -1600, "p50": -1500, "p90": -1400},
                "dur_min": {"p10": 5.0, "p50": 10.0, "p90": 15.0},
                "hour": [0] * 24,
                "dow": [0] * 7,
                "energy_share": 0.15,
            },
        ],
        "plugs": [
            {
                "id": "p1",
                "n": 10,
                "energy_kwh_per_day": 0.8,
                "dur_min": {"p10": 30.0, "p50": 45.0, "p90": 60.0},
                "peak_w": 2200,
                "hour": [0] * 24,
                "dow": [0] * 7,
            }
        ],
    }


def test_valid_proposal_accepted(sample_digest):
    """A structurally and semantically correct proposal passes validation."""
    prop = proposal_for(sample_digest)
    cluster_ids = [c["id"] for c in sample_digest["clusters"]]
    plug_ids = [p["id"] for p in sample_digest["plugs"]]

    valid, errors = validate_proposal(prop, cluster_ids=cluster_ids, plug_ids=plug_ids)
    assert valid is not None
    assert errors == []


def test_empty_devices_array_is_valid(sample_digest):
    """An empty devices array is explicitly valid (LLM finds no devices)."""
    prop = proposal_for(sample_digest, devices=[])
    cluster_ids = ["c01", "c02"]
    plug_ids = ["p1"]

    valid, errors = validate_proposal(prop, cluster_ids=cluster_ids, plug_ids=plug_ids)
    assert valid is not None
    assert errors == []
    assert valid["devices"] == []


def test_unknown_cluster_and_plug_references_rejected(sample_digest):
    """Referencing unknown cluster or plug ids produces error paths."""
    prop = proposal_for(sample_digest)
    prop["devices"][0]["member_clusters"] = ["c99"]
    prop["devices"][0]["states"][0]["cluster_ref"] = "c99"
    prop["devices"][0]["reference_plug"] = "p99"

    valid, errors = validate_proposal(
        prop,
        cluster_ids=["c01", "c02"],
        plug_ids=["p1"],
    )
    assert valid is None
    paths = [err.split(":")[0] for err in errors]
    assert "/devices/0/member_clusters/0" in paths
    assert "/devices/0/states/0/cluster_ref" in paths
    assert "/devices/0/reference_plug" in paths


def test_extra_and_missing_keys_rejected(sample_digest):
    """Additional properties and missing required properties are flagged with exact paths."""
    prop = proposal_for(sample_digest)
    prop["extra_root"] = "forbidden"
    del prop["analysis"]

    valid, errors = validate_proposal(
        prop,
        cluster_ids=["c01", "c02"],
        plug_ids=["p1"],
    )
    assert valid is None
    paths = [err.split(":")[0] for err in errors]
    assert "/extra_root" in paths
    assert "/analysis" in paths


def test_bad_enums_rejected(sample_digest):
    """Invalid enum values for category, tracker_kind, role, confidence, and phase_hint are caught."""
    prop = proposal_for(sample_digest)
    dev = prop["devices"][0]
    dev["category"] = "nuclear_reactor"
    dev["tracker_kind"] = "continuous_v2"
    dev["states"][0]["role"] = "super_main"
    dev["confidence"] = "ultra_high"
    dev["phase_hint"] = "quad_phase"

    valid, errors = validate_proposal(
        prop,
        cluster_ids=["c01", "c02"],
        plug_ids=["p1"],
    )
    assert valid is None
    paths = [err.split(":")[0] for err in errors]
    assert "/devices/0/category" in paths
    assert "/devices/0/tracker_kind" in paths
    assert "/devices/0/states/0/role" in paths
    assert "/devices/0/confidence" in paths
    assert "/devices/0/phase_hint" in paths


def test_state_cluster_must_be_in_device_member_clusters(sample_digest):
    """A state cluster_ref that is in the digest but NOT in member_clusters is rejected."""
    prop = proposal_for(sample_digest)
    # c02 is in digest, but device 0 only has member_clusters = ['c01']
    prop["devices"][0]["member_clusters"] = ["c01"]
    prop["devices"][0]["states"][0]["cluster_ref"] = "c02"

    valid, errors = validate_proposal(
        prop,
        cluster_ids=["c01", "c02"],
        plug_ids=["p1"],
    )
    assert valid is None
    assert any("/devices/0/states/0/cluster_ref" in err for err in errors)


def test_shared_cluster_requires_ambiguous_with(sample_digest):
    """A cluster listed in two devices is valid ONLY when each device lists it in ambiguous_with."""
    prop = proposal_for(sample_digest)
    dev1 = dict(prop["devices"][0])
    dev2 = dict(prop["devices"][0])
    dev1["ref"] = "d1"
    dev2["ref"] = "d2"
    dev1["member_clusters"] = ["c01"]
    dev2["member_clusters"] = ["c01"]
    dev1["states"] = [{"name": "on1", "cluster_ref": "c01", "role": "main"}]
    dev2["states"] = [{"name": "on2", "cluster_ref": "c01", "role": "main"}]

    # 1. Neither lists ambiguous_with -> error for both
    dev1["ambiguous_with"] = []
    dev2["ambiguous_with"] = []
    prop["devices"] = [dev1, dev2]

    valid, errors = validate_proposal(prop, cluster_ids=["c01", "c02"], plug_ids=["p1"])
    assert valid is None
    assert any("/devices/0/ambiguous_with" in err for err in errors)
    assert any("/devices/1/ambiguous_with" in err for err in errors)

    # 2. Both list c01 in ambiguous_with -> passes
    dev1["ambiguous_with"] = ["c01"]
    dev2["ambiguous_with"] = ["c01"]
    prop["devices"] = [dev1, dev2]

    valid, errors = validate_proposal(prop, cluster_ids=["c01", "c02"], plug_ids=["p1"])
    assert valid is not None
    assert errors == []


def test_rejected_cluster_used_by_device_error(sample_digest):
    """A cluster listed in rejected_clusters that is also used in a device produces an error."""
    prop = proposal_for(sample_digest)
    prop["rejected_clusters"] = [{"cluster_ref": "c01", "reason": "noise"}]

    valid, errors = validate_proposal(prop, cluster_ids=["c01", "c02"], plug_ids=["p1"])
    assert valid is None
    assert any("/rejected_clusters/0/cluster_ref" in err for err in errors)


def test_duplicate_device_refs_and_max_devices(sample_digest):
    """Duplicate device refs and exceeding MAX_DEVICES produce errors."""
    prop = proposal_for(sample_digest)
    d0 = prop["devices"][0]

    # Duplicate ref
    prop["devices"] = [dict(d0), dict(d0)]
    valid, errors = validate_proposal(prop, cluster_ids=["c01", "c02"], plug_ids=["p1"])
    assert valid is None
    assert any("/devices/1/ref" in err and "duplicate" in err for err in errors)

    # More than MAX_DEVICES (40)
    devs = []
    for i in range(MAX_DEVICES + 1):
        dev_copy = dict(d0)
        dev_copy["ref"] = f"d{i}"
        dev_copy["member_clusters"] = []
        dev_copy["states"] = []
        devs.append(dev_copy)
    prop["devices"] = devs

    valid, errors = validate_proposal(prop, cluster_ids=["c01", "c02"], plug_ids=["p1"])
    assert valid is None
    assert any("/devices" in err and "exceeds maximum of 40" in err for err in errors)


def test_string_length_limits(sample_digest):
    """Strings exceeding caps (analysis > 1200, rationale > 300, merge reason > 200, label > 40) fail."""
    prop = proposal_for(sample_digest)
    prop["analysis"] = "x" * (ANALYSIS_MAX + 1)
    prop["devices"][0]["rationale"] = "y" * (RATIONALE_MAX + 1)
    prop["devices"][0]["label"] = "z" * (LABEL_MAX + 1)
    prop["merge_suggestions"] = [{"clusters": ["c01", "c02"], "reason": "w" * (REASON_MAX + 1)}]

    valid, errors = validate_proposal(prop, cluster_ids=["c01", "c02"], plug_ids=["p1"])
    assert valid is None
    paths = [err.split(":")[0] for err in errors]
    assert "/analysis" in paths
    assert "/devices/0/rationale" in paths
    assert "/devices/0/label" in paths
    assert "/merge_suggestions/0/reason" in paths


def test_errors_sorted_deterministically(sample_digest):
    """Errors are sorted by (path, message) and identical input gives identical list."""
    prop = proposal_for(sample_digest)
    prop["extra_z"] = 1
    prop["extra_a"] = 2
    prop["devices"][0]["label"] = "x" * 50

    _, errs1 = validate_proposal(prop, cluster_ids=["c01", "c02"], plug_ids=["p1"])
    _, errs2 = validate_proposal(prop, cluster_ids=["c01", "c02"], plug_ids=["p1"])
    assert errs1 == errs2
    assert errs1 == sorted(errs1)


def test_sanitize_label_rules():
    """sanitize_label normalises NFC, drops Unicode category C, removes forbidden chars, collapses space, cuts to 40."""
    # Control character \x00, format char \u200b, forbidden characters ` * _ [ ] ( ) < > # | ~ ! \
    raw = "  Kitchen  `*_[Oven]#|~!\\  \x00\u200b(Main) <1>  "
    sanitized = sanitize_label(raw)
    assert sanitized == "Kitchen Oven Main 1"

    # Empty string or purely forbidden chars becomes "unknown"
    assert sanitize_label("") == "unknown"
    assert sanitize_label("   `*#   ") == "unknown"
    assert sanitize_label(None) == "unknown"

    # Cut to 40 code points
    long_label = "A" * 60
    assert len(sanitize_label(long_label)) == 40


def test_parse_proposal_handles_messages(sample_digest):
    """parse_proposal extracts JSON from OpenAI message shapes and validates."""
    prop = proposal_for(sample_digest)
    comp = chat_completion(json.dumps(prop))
    message = comp["choices"][0]["message"]

    valid, errors, trimmed = parse_proposal(message, cluster_ids=["c01", "c02"], plug_ids=["p1"])
    assert valid is not None
    assert errors == []
    assert trimmed == []

    # Unparseable
    bad_msg = {"role": "assistant", "content": "not json at all"}
    valid_bad, errors_bad, trimmed_bad = parse_proposal(
        bad_msg, cluster_ids=["c01", "c02"], plug_ids=["p1"]
    )
    assert valid_bad is None
    assert errors_bad == ["/: no JSON object found"]
    assert trimmed_bad == []


def test_salvage_drops_error_devices_and_keeps_valid_entries(sample_digest):
    """Salvage preserves valid devices, returns unresolved member clusters of dropped devices, and keeps valid rejections."""
    prop = proposal_for(sample_digest)
    d0 = dict(prop["devices"][0])
    d0["ref"] = "d1"
    d0["member_clusters"] = ["c01"]
    d0["states"] = [{"name": "s1", "cluster_ref": "c01", "role": "main"}]

    # Second device has invalid cluster
    d1 = dict(prop["devices"][0])
    d1["ref"] = "d2"
    d1["member_clusters"] = ["c99"]
    d1["states"] = [{"name": "s2", "cluster_ref": "c99", "role": "main"}]

    prop["devices"] = [d0, d1]
    prop["rejected_clusters"] = [
        {"cluster_ref": "c02", "reason": "noise"},
        {"cluster_ref": "c88", "reason": "noise"},  # invalid
    ]
    prop["merge_suggestions"] = [
        {"clusters": ["c01", "c02"], "reason": "valid merge"},
        {"clusters": ["c77"], "reason": "invalid"},  # invalid
    ]

    _, errors = validate_proposal(prop, cluster_ids=["c01", "c02"], plug_ids=["p1"])
    assert errors

    salvaged = salvage(prop, errors, cluster_ids=["c01", "c02"], plug_ids=["p1"])
    assert salvaged is not None
    salvaged_prop, unresolved = salvaged

    assert len(salvaged_prop["devices"]) == 1
    assert salvaged_prop["devices"][0]["ref"] == "d1"
    assert unresolved == ["c99"]

    # Valid rejection c02 survived, c88 dropped
    assert len(salvaged_prop["rejected_clusters"]) == 1
    assert salvaged_prop["rejected_clusters"][0]["cluster_ref"] == "c02"

    # Valid merge survived, c77 dropped
    assert len(salvaged_prop["merge_suggestions"]) == 1
    assert salvaged_prop["merge_suggestions"][0]["clusters"] == ["c01", "c02"]


def test_tracker_kind_must_match_cluster_kind(sample_digest):
    """A program device needs session clusters and a switched device switched clusters (CR-03)."""
    kinds = {"c01": "switched", "c02": "session"}
    prop = proposal_for(sample_digest, devices=[])
    base = proposal_for(sample_digest)["devices"][0]
    good_sw = {**base, "ref": "d1", "tracker_kind": "switched", "member_clusters": ["c01"],
               "states": [{"name": "on", "cluster_ref": "c01", "role": "main"}]}
    good_prog = {**base, "ref": "d2", "tracker_kind": "program", "member_clusters": ["c02"],
                 "states": [{"name": "run", "cluster_ref": "c02", "role": "main"}]}
    prop["devices"] = [good_sw, good_prog]
    valid, errors = validate_proposal(
        prop, cluster_ids=["c01", "c02"], plug_ids=["p1"], cluster_kinds=kinds
    )
    assert errors == []
    assert valid is not None

    bad_prog = {**good_prog, "member_clusters": ["c01"],
                "states": [{"name": "run", "cluster_ref": "c01", "role": "main"}]}
    bad_sw = {**good_sw, "ref": "d3", "member_clusters": ["c02"],
              "states": [{"name": "on", "cluster_ref": "c02", "role": "main"}]}
    prop["devices"] = [bad_prog, bad_sw]
    valid, errors = validate_proposal(
        prop, cluster_ids=["c01", "c02"], plug_ids=["p1"], cluster_kinds=kinds
    )
    assert valid is None
    assert any(e.startswith("/devices/0/member_clusters/0:") and "session" in e for e in errors)
    assert any(e.startswith("/devices/1/member_clusters/0:") and "switched" in e for e in errors)

    # Salvage drops the mismatched device and keeps the matching one
    prop["devices"] = [good_sw, bad_sw]
    _, errors = validate_proposal(
        prop, cluster_ids=["c01", "c02"], plug_ids=["p1"], cluster_kinds=kinds
    )
    salvaged = salvage(
        prop, errors, cluster_ids=["c01", "c02"], plug_ids=["p1"], cluster_kinds=kinds
    )
    assert salvaged is not None
    assert [d["ref"] for d in salvaged[0]["devices"]] == ["d1"]

    # parse_proposal forwards the kinds as well
    msg = chat_completion(content=json.dumps(prop))["choices"][0]["message"]
    parsed, p_errors, _ = parse_proposal(
        msg, cluster_ids=["c01", "c02"], plug_ids=["p1"], cluster_kinds=kinds
    )
    assert parsed is None
    assert any(e.startswith("/devices/1/member_clusters/0:") for e in p_errors)


def test_salvage_returns_none_when_no_device_survives(sample_digest):
    """Salvage returns None when all devices had errors."""
    prop = proposal_for(sample_digest)
    prop["devices"] = [prop["devices"][0]]
    prop["devices"][0]["member_clusters"] = ["c99"]
    prop["devices"][0]["states"][0]["cluster_ref"] = "c99"

    _, errors = validate_proposal(prop, cluster_ids=["c01", "c02"], plug_ids=["p1"])
    assert salvage(prop, errors, cluster_ids=["c01", "c02"], plug_ids=["p1"]) is None


def test_trim_text_cuts_to_cap_with_mark():
    """Over-long text is cut to the cap, trailing whitespace removed, and the mark appended within the cap."""
    assert TRIM_MARK == "…"
    assert trim_text("short", 40) == "short"
    assert trim_text("x" * 40, 40) == "x" * 40

    out = trim_text("y" * 41, 40)
    assert len(out) == 40
    assert out == "y" * 39 + TRIM_MARK

    # Trailing whitespace before the mark is removed
    out = trim_text("Heat pump" + " " * 3 + "z" * 40, 12)
    assert out == "Heat pump" + TRIM_MARK

    # Degenerate caps never exceed the cap
    assert trim_text("abc", 1) == TRIM_MARK
    assert trim_text("abc", 0) == ""


def test_trim_text_is_unicode_safe():
    """The cut never separates a base character from its combining marks or a joined emoji sequence."""
    # "e" + COMBINING ACUTE: cutting between them would leave a bare "e"
    decomposed = "Caf" + "é" * 5
    out = trim_text(decomposed, 6)
    assert len(out) <= 6
    assert out.endswith(TRIM_MARK)
    body = out[: -len(TRIM_MARK)]
    assert body == "Caf" or body.endswith("́")

    # Family emoji joined by ZWJ is kept whole or dropped whole
    family = "\U0001F468‍\U0001F469‍\U0001F467"
    out = trim_text("ab" + family + "cdefgh", 6)
    assert len(out) <= 6
    body = out[: -len(TRIM_MARK)]
    assert not body.endswith("‍")
    assert body in ("ab", "ab" + family)

    # Astral characters count as one code point and are never split
    out = trim_text("\U0001F50C" * 10, 5)
    assert out == "\U0001F50C" * 4 + TRIM_MARK


def test_trim_free_text_trims_only_capped_free_text(sample_digest):
    """analysis, device label/rationale and merge reason are trimmed; ids, enums and structure are not."""
    prop = proposal_for(sample_digest)
    prop["analysis"] = "a" * (ANALYSIS_MAX + 5)
    prop["devices"][0]["label"] = "Washing machine with very long model name"  # 41
    prop["devices"][0]["rationale"] = "r" * (RATIONALE_MAX + 1)
    prop["devices"][0]["ref"] = "d" * 200
    prop["devices"][0]["states"][0]["name"] = "n" * 500
    prop["devices"][0]["category"] = "not-an-enum-value-" + "x" * 100
    prop["merge_suggestions"] = [{"clusters": ["c01", "c02"], "reason": "m" * (REASON_MAX + 1)}]
    prop["rejected_clusters"] = [{"cluster_ref": "c02", "reason": "q" * 300}]
    original = json.loads(json.dumps(prop))

    out, trimmed = trim_free_text(prop)

    assert trimmed == [
        "/analysis",
        "/devices/0/label",
        "/devices/0/rationale",
        "/merge_suggestions/0/reason",
    ]
    assert len(out["analysis"]) == ANALYSIS_MAX
    assert len(out["devices"][0]["label"]) <= LABEL_MAX
    assert out["devices"][0]["label"].endswith(TRIM_MARK)
    assert len(out["devices"][0]["rationale"]) == RATIONALE_MAX
    assert len(out["merge_suggestions"][0]["reason"]) == REASON_MAX
    # Untouched: ids, uncapped names, enums (valid or not), enum-typed rejection reason
    assert out["devices"][0]["ref"] == "d" * 200
    assert out["devices"][0]["states"][0]["name"] == "n" * 500
    assert out["devices"][0]["category"] == original["devices"][0]["category"]
    assert out["rejected_clusters"] == original["rejected_clusters"]
    # Input is not mutated
    assert prop == original


def test_trim_free_text_tolerates_malformed_shapes():
    """Non-dict roots, non-list arrays, non-dict items and non-string fields pass through untouched."""
    assert trim_free_text(["x"]) == (["x"], [])
    obj = {"analysis": 5, "devices": "nope", "merge_suggestions": [1, {"reason": None}]}
    out, trimmed = trim_free_text(obj)
    assert trimmed == []
    assert out == obj


def test_parse_proposal_trims_instead_of_rejecting(sample_digest):
    """Over-length free text validates after trimming and is reported; other violations still fail."""
    prop = proposal_for(sample_digest)
    prop["devices"][0]["label"] = "L" * (LABEL_MAX + 1)
    prop["devices"][0]["rationale"] = "R" * (RATIONALE_MAX + 50)
    message = chat_completion(json.dumps(prop))["choices"][0]["message"]

    valid, errors, trimmed = parse_proposal(message, cluster_ids=["c01", "c02"], plug_ids=["p1"])
    assert errors == []
    assert valid is not None
    assert trimmed == ["/devices/0/label", "/devices/0/rationale"]
    assert len(valid["devices"][0]["label"]) == LABEL_MAX
    # The label still goes through the label rules downstream
    assert sanitize_label(valid["devices"][0]["label"]) == valid["devices"][0]["label"]

    # A non-length violation in the same reply still fails validation (repair turn)
    prop["devices"][0]["member_clusters"] = ["c99"]
    message = chat_completion(json.dumps(prop))["choices"][0]["message"]
    valid, errors, trimmed = parse_proposal(message, cluster_ids=["c01", "c02"], plug_ids=["p1"])
    assert valid is None
    assert any("c99" in e for e in errors)
    assert not any("exceeds maximum length" in e for e in errors)
    assert trimmed == ["/devices/0/label", "/devices/0/rationale"]


def test_salvage_keeps_over_length_devices_after_trimming(sample_digest):
    """Salvage trims free text first, so an over-long label does not cost the device."""
    prop = proposal_for(sample_digest)
    d0 = dict(prop["devices"][0])
    d0["label"] = "L" * (LABEL_MAX + 10)
    d1 = dict(prop["devices"][0])
    d1["ref"] = "d2"
    d1["member_clusters"] = ["c99"]
    d1["states"] = [{"name": "s2", "cluster_ref": "c99", "role": "main"}]
    prop["devices"] = [d0, d1]

    _, errors = validate_proposal(prop, cluster_ids=["c01", "c02"], plug_ids=["p1"])
    salvaged = salvage(prop, errors, cluster_ids=["c01", "c02"], plug_ids=["p1"])
    assert salvaged is not None
    salvaged_prop, unresolved = salvaged
    assert [d["ref"] for d in salvaged_prop["devices"]] == ["d1"]
    assert len(salvaged_prop["devices"][0]["label"]) == LABEL_MAX
    assert unresolved == ["c99"]


def _with_verdicts(sample_digest, verdicts):
    prop = proposal_for(sample_digest)
    prop["example_verdicts"] = verdicts
    return prop


def test_verdicts_optional_and_validated(sample_digest):
    """With example ids, example_verdicts is optional but each entry is validated (D-07)."""
    clusters, plugs = ["c01", "c02"], ["p1"]

    # Absent is valid.
    base = proposal_for(sample_digest)
    valid, errors = validate_proposal(
        base, cluster_ids=clusters, plug_ids=plugs, example_ids=("e1", "e2")
    )
    assert valid is not None
    assert errors == []

    # A valid list is accepted.
    prop = _with_verdicts(
        sample_digest,
        [
            {"example": "e1", "verdict": "consistent"},
            {"example": "e2", "verdict": "cannot_tell"},
        ],
    )
    valid, errors = validate_proposal(
        prop, cluster_ids=clusters, plug_ids=plugs, example_ids=("e1", "e2")
    )
    assert valid is not None
    assert errors == []
    assert set(EXAMPLE_VERDICTS) == {"consistent", "inconsistent", "cannot_tell"}

    # Unknown example id.
    prop = _with_verdicts(sample_digest, [{"example": "e9", "verdict": "consistent"}])
    valid, errors = validate_proposal(
        prop, cluster_ids=clusters, plug_ids=plugs, example_ids=("e1", "e2")
    )
    assert valid is None
    assert any(e.startswith("/example_verdicts/0:") for e in errors)

    # Unknown verdict.
    prop = _with_verdicts(sample_digest, [{"example": "e1", "verdict": "maybe"}])
    _, errors = validate_proposal(
        prop, cluster_ids=clusters, plug_ids=plugs, example_ids=("e1", "e2")
    )
    assert any(e.startswith("/example_verdicts/0:") for e in errors)

    # Extra key.
    prop = _with_verdicts(sample_digest, [{"example": "e1", "verdict": "consistent", "x": 1}])
    _, errors = validate_proposal(
        prop, cluster_ids=clusters, plug_ids=plugs, example_ids=("e1", "e2")
    )
    assert any(e.startswith("/example_verdicts/0:") for e in errors)

    # Duplicate example id -> the second entry errors.
    prop = _with_verdicts(
        sample_digest,
        [
            {"example": "e1", "verdict": "consistent"},
            {"example": "e1", "verdict": "inconsistent"},
        ],
    )
    _, errors = validate_proposal(
        prop, cluster_ids=clusters, plug_ids=plugs, example_ids=("e1", "e2")
    )
    assert any(e.startswith("/example_verdicts/1:") for e in errors)

    # Without example ids the key stays an unknown root key (capture-free unchanged).
    prop = _with_verdicts(sample_digest, [{"example": "e1", "verdict": "consistent"}])
    _, errors = validate_proposal(prop, cluster_ids=clusters, plug_ids=plugs)
    assert any(e.startswith("/example_verdicts:") for e in errors)


def test_parse_proposal_takes_example_ids(sample_digest):
    """parse_proposal forwards example_ids and reports the same example_verdicts paths."""
    clusters, plugs = ["c01", "c02"], ["p1"]

    prop = _with_verdicts(sample_digest, [{"example": "e1", "verdict": "consistent"}])
    msg = chat_completion(json.dumps(prop))["choices"][0]["message"]
    valid, errors, trimmed = parse_proposal(
        msg, cluster_ids=clusters, plug_ids=plugs, example_ids=("e1",)
    )
    assert valid is not None
    assert errors == []

    prop = _with_verdicts(sample_digest, [{"example": "e9", "verdict": "consistent"}])
    msg = chat_completion(json.dumps(prop))["choices"][0]["message"]
    valid, errors, _ = parse_proposal(
        msg, cluster_ids=clusters, plug_ids=plugs, example_ids=("e1",)
    )
    assert valid is None
    assert any(e.startswith("/example_verdicts/0:") for e in errors)


def test_salvage_drops_bad_verdicts_only(sample_digest):
    """Salvage keeps the devices and drops only the bad verdict entries (D-07)."""
    clusters, plugs = ["c01", "c02"], ["p1"]

    prop = _with_verdicts(
        sample_digest,
        [
            {"example": "e1", "verdict": "consistent"},
            {"example": "e9", "verdict": "consistent"},  # bad
            {"example": "e2", "verdict": "cannot_tell"},
        ],
    )
    _, errors = validate_proposal(
        prop, cluster_ids=clusters, plug_ids=plugs, example_ids=("e1", "e2")
    )
    assert any("/example_verdicts/1:" in e for e in errors)

    salvaged = salvage(
        prop, errors, cluster_ids=clusters, plug_ids=plugs, example_ids=("e1", "e2")
    )
    assert salvaged is not None
    salvaged_prop, _ = salvaged
    assert salvaged_prop["devices"] == prop["devices"]
    assert salvaged_prop["example_verdicts"] == [
        {"example": "e1", "verdict": "consistent"},
        {"example": "e2", "verdict": "cannot_tell"},
    ]


def test_pop_example_verdicts():
    """pop_example_verdicts returns the proposal without the key plus the verdicts; input untouched."""
    prop = {
        "analysis": "a",
        "devices": [],
        "example_verdicts": [{"example": "e1", "verdict": "consistent"}],
    }
    without, verdicts = pop_example_verdicts(prop)
    assert without == {"analysis": "a", "devices": []}
    assert verdicts == [{"example": "e1", "verdict": "consistent"}]
    assert "example_verdicts" in prop  # not mutated

    without2, verdicts2 = pop_example_verdicts({"analysis": "a"})
    assert without2 == {"analysis": "a"}
    assert verdicts2 == []

    # A non-list value yields an empty list.
    _, verdicts3 = pop_example_verdicts({"analysis": "a", "example_verdicts": "nope"})
    assert verdicts3 == []
