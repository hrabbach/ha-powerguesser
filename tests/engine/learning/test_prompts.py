"""Tests for prompt generation, repair request formatting, names opt-in, and privacy guarantees."""

import ast
import json
from pathlib import Path
import re
import pytest

from custom_components.powerguesser.engine.learning.prompts import (
    PROPOSAL_SCHEMA,
    STATIC_PREFIX,
    build_refine_request,
    build_repair_request,
    build_request,
    digest_payload_bytes,
    payload_bytes,
    render_digest,
)
from custom_components.powerguesser.engine.llm_wire import Strategy


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
                "conc_with": [],
            }
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


def test_build_repair_request_minimal_structure():
    """build_repair_request contains cluster/plug IDs, cut output, errors, system prefix, and no digest lines."""
    profile = {"strategy": Strategy.JSON_SCHEMA.value, "token_param": "max_completion_tokens", "send_temperature": True}
    errors = ["/devices/0/label: exceeds maximum length", "/analysis: missing required property"]
    long_output = "{" + "a" * 20000 + "}"

    body = build_repair_request(
        model="test-model",
        profile=profile,
        max_tokens=4000,
        cluster_ids=["c01", "c02"],
        plug_ids=["p1"],
        invalid_output=long_output,
        errors=errors,
    )

    messages = body["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == STATIC_PREFIX

    user_msg = messages[1]["content"]
    assert "Valid cluster ids: c01, c02" in user_msg
    assert "Valid plug ids: p1" in user_msg
    assert "- /devices/0/label: exceeds maximum length" in user_msg
    assert "- /analysis: missing required property" in user_msg
    # Cut to 16384 characters
    assert len(long_output[:16384]) == 16384
    assert f"<output>\n{long_output[:16384]}\n</output>" in user_msg
    # No rendered digest lines in repair turn
    assert "header grid_s=" not in user_msg


def test_names_opt_in_in_data_block_only(sample_digest):
    """build_request with names puts 'p1: Name' inside <data> only, system prompt has no names, sorted by pN."""
    profile = {"strategy": Strategy.JSON_SCHEMA.value, "token_param": "max_completion_tokens", "send_temperature": True}
    names = {"p1": "Basement `*Laundry*"}

    body = build_request(
        sample_digest,
        model="test-model",
        profile=profile,
        max_tokens=4000,
        names=names,
    )

    sys_msg = body["messages"][0]["content"]
    assert "Basement" not in sys_msg
    assert "Laundry" not in sys_msg

    user_msg = body["messages"][1]["content"]
    assert "<names>\np1: Basement Laundry\n</names>" in user_msg


def test_empty_or_none_names_produce_byte_identical_bodies(sample_digest):
    """names=None and names={} produce byte-identical request bodies."""
    profile = {"strategy": Strategy.JSON_SCHEMA.value, "token_param": "max_completion_tokens", "send_temperature": True}

    body_none = build_request(
        sample_digest,
        model="test-model",
        profile=profile,
        max_tokens=4000,
        names=None,
    )
    body_empty = build_request(
        sample_digest,
        model="test-model",
        profile=profile,
        max_tokens=4000,
        names={},
    )

    bytes_none = json.dumps(body_none, sort_keys=True)
    bytes_empty = json.dumps(body_empty, sort_keys=True)
    assert bytes_none == bytes_empty
    assert "<names>" not in bytes_none


def test_digest_without_plugs_gives_no_names_block(sample_digest):
    """A digest without plug lines yields no names block even when names are provided."""
    profile = {"strategy": Strategy.JSON_SCHEMA.value, "token_param": "max_completion_tokens", "send_temperature": True}
    digest_no_plugs = dict(sample_digest)
    digest_no_plugs["plugs"] = []

    body = build_request(
        digest_no_plugs,
        model="test-model",
        profile=profile,
        max_tokens=4000,
        names={"p1": "UniqueApplianceXYZ"},
    )
    raw = json.dumps(body)
    assert "<names>" not in raw
    assert "UniqueApplianceXYZ" not in raw


def test_digest_payload_bytes_matches_digest_inside_payload(sample_digest):
    """digest_payload_bytes is the UTF-8 size of the escaped digest inside payload_bytes(body)."""
    profile = {"strategy": Strategy.JSON_SCHEMA.value, "token_param": "max_completion_tokens", "send_temperature": True}
    names = {"p1": "Küche Wäschetrockner"}  # multi-byte UTF-8 in the digest
    rendered = render_digest(sample_digest, names)
    escaped = json.dumps(rendered, ensure_ascii=False)[1:-1]
    assert "\\n" in escaped  # digest lines are escaped newlines in the payload
    assert len(escaped.encode("utf-8")) > len(escaped)  # bytes, not characters

    bodies = (
        build_request(sample_digest, model="m", profile=profile, max_tokens=4000, names=names),
        build_refine_request(
            model="m",
            profile=profile,
            max_tokens=4000,
            digest=sample_digest,
            previous={"analysis": "a \"quoted\"\nline", "devices": []},
            scorecard={"devices": []},
            names=names,
        ),
    )
    for body in bodies:
        compact = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
        assert escaped in compact
        assert digest_payload_bytes(sample_digest, names) == len(escaped.encode("utf-8"))
        assert digest_payload_bytes(sample_digest, names) < payload_bytes(body)

    # names=None renders no names block, so the digest is smaller
    assert digest_payload_bytes(sample_digest) < digest_payload_bytes(sample_digest, names)


def test_request_bodies_carry_no_timestamps_or_entity_ids(sample_digest):
    """All 4 strategies with names on and off produce request bodies with zero timestamps, ISO dates, or sensor IDs."""
    int_pattern = re.compile(r"\b\d{12,}\b")
    iso_pattern = re.compile(r"\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}")

    for strat in (Strategy.JSON_SCHEMA, Strategy.TOOL_CALL, Strategy.JSON_OBJECT, Strategy.FENCED):
        for names in (None, {"p1": "Smart Plug Dryer"}):
            profile = {"strategy": strat.value, "token_param": "max_completion_tokens", "send_temperature": True}
            body = build_request(
                sample_digest,
                model="test-model",
                profile=profile,
                max_tokens=4000,
                names=names,
            )
            raw = json.dumps(body)

            assert not int_pattern.findall(raw), f"12+ digit int found for {strat}"
            assert "sensor." not in raw, f"sensor. found for {strat}"
            assert not iso_pattern.findall(raw), f"ISO date found for {strat}"


def test_ha_side_learning_modules_avoid_numpy_scipy():
    """prompts.py, proposal.py, artefact.py and engine/llm_wire.py import neither numpy nor scipy."""
    base_dir = Path(__file__).resolve().parent.parent.parent.parent / "custom_components" / "powerguesser" / "engine"
    files_to_check = [
        base_dir / "learning" / "prompts.py",
        base_dir / "learning" / "proposal.py",
        base_dir / "learning" / "artefact.py",
        base_dir / "learning" / "budget.py",
        base_dir / "llm_wire.py",
    ]

    for fpath in files_to_check:
        assert fpath.exists(), f"File {fpath} does not exist"
        tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "numpy" not in alias.name, f"{fpath} imports numpy via 'import {alias.name}'"
                    assert "scipy" not in alias.name, f"{fpath} imports scipy via 'import {alias.name}'"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "numpy" not in mod, f"{fpath} imports from numpy via 'from {mod}'"
                assert "scipy" not in mod, f"{fpath} imports from scipy via 'from {mod}'"
