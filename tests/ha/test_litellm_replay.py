"""Replay of the owner's LiteLLM proxy responses recorded by the 03-09 live spike (D-15).

tests/fixtures/llm/litellm.json holds the redacted cassettes written by
scripts/llm_spike.py (request bodies only as sha256, response headers reduced to
x-litellm-response-cost and x-litellm-model-id). The request byte counts below come
from the spike's rung table (03-SPIKE.md "## Rung table"), because the cassettes do
not store request bodies.
"""

from typing import Any

import pytest

from homeassistant.core import HomeAssistant

from custom_components.powerguesser import const
from custom_components.powerguesser.engine.learning.budget import (
    estimate_tokens,
    truncation_suspected,
)
from custom_components.powerguesser.engine.learning.proposal import parse_proposal
from custom_components.powerguesser.engine.llm_wire import (
    PROBE_EXPECTED,
    extract_json,
    model_matches,
)
from custom_components.powerguesser.llm_client import LlmClient
from tests.conftest import load_llm_cassette, mock_llm

BASE_URL = "http://litellm.test:4000/v1"
MODEL = "kimi-for-coding"

# Request sizes from the spike's rung table (bytes of the JSON request body).
PAYLOAD_BYTES = 5514
TRUNCATION_BYTES = 7488
# Digest share of those bodies (03-SPIKE.md): 983 bytes in the payload; the truncation
# request repeats the <data> block three times (+1974 bytes).
PAYLOAD_DIGEST_BYTES = 983
TRUNCATION_DIGEST_BYTES = 983 + 1974

# Digest ids of the synthetic preview body the payload was built from.
CLUSTER_IDS = ["c01", "c02", "c03"]
PLUG_IDS = ["p1"]


@pytest.fixture(autouse=True)
def zero_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable retry backoff delays during replay tests."""
    monkeypatch.setattr(const, "RETRY_BACKOFF_S", 0.0)


def _cassette(name: str) -> dict[str, Any]:
    return load_llm_cassette("litellm", name)


def _message(name: str) -> dict[str, Any]:
    return _cassette(name)["json"]["choices"][0]["message"]


def _rung_of(data: dict[str, Any]) -> str:
    """Recognise the probe rung and parameter variant from a request body."""
    if "tools" in data:
        rung = "tool_call"
    else:
        fmt = (data.get("response_format") or {}).get("type")
        rung = {"json_schema": "json_schema", "json_object": "json_object"}.get(
            fmt, "fenced"
        )
    if rung != "json_schema":
        return rung
    if "temperature" in data:
        return "json_schema_temperature_0" if data["temperature"] == 0 else (
            "json_schema_temperature_1"
        )
    if "max_tokens" in data:
        return "json_schema_max_tokens"
    return "json_schema"


async def test_litellm_probe_replay_selects_live_strategy(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """The probe replayed against the proxy's answers selects json_schema, like the live run.

    The stored profile reflects deviation A2: max_completion_tokens and no temperature.
    """
    client = LlmClient(hass, BASE_URL, "sk-replay-not-a-secret")
    sent: list[dict[str, Any]] = []

    def _responder(data: dict[str, Any]) -> dict[str, Any]:
        sent.append(data)
        return _cassette(_rung_of(data))

    mock_llm(aioclient_mock, BASE_URL, _responder)

    profile = await client.async_probe(MODEL)

    assert profile["strategy"] == "json_schema"
    assert profile["token_param"] == "max_completion_tokens"
    assert profile["send_temperature"] is False
    assert profile["model_returned"] == MODEL
    assert profile["model_mismatch"] is False

    # One request, on the first rung, in the default profile shape.
    assert len(sent) == 1
    assert "temperature" not in sent[0]
    assert "max_completion_tokens" in sent[0]
    assert "max_tokens" not in sent[0]


def test_litellm_router_fallback_detected_by_model_matches() -> None:
    """The rungs the router answered with its fallback model are flagged, the others are not.

    Live finding: the json_schema_temperature_0 and tool_call requests for
    kimi-for-coding were answered by xiaomi/mimo-v2.6-flash with status 200.
    """
    mismatched = {"json_schema_temperature_0", "tool_call"}
    names = [
        "json_schema",
        "tool_call",
        "json_object",
        "fenced",
        "json_schema_max_tokens",
        "json_schema_temperature_0",
        "json_schema_temperature_1",
        "payload",
        "truncation",
    ]
    for name in names:
        returned = _cassette(name)["json"]["model"]
        assert model_matches(MODEL, returned) is (name not in mismatched), (
            name,
            returned,
        )


async def test_litellm_probe_surfaces_router_fallback(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """When the probe lands on a rung the router answered with another model, it says so."""
    client = LlmClient(hass, BASE_URL, "sk-replay-not-a-secret")

    def _responder(data: dict[str, Any]) -> dict[str, Any]:
        rung = _rung_of(data)
        if rung == "json_schema":
            return load_llm_cassette("probe", "unsupported_response_format")
        return _cassette(rung)

    mock_llm(aioclient_mock, BASE_URL, _responder)

    profile = await client.async_probe(MODEL)

    assert profile["strategy"] == "tool_call"
    assert profile["model_returned"] == "xiaomi/mimo-v2.6-flash"
    assert profile["model_mismatch"] is True


def test_litellm_payload_reply_location() -> None:
    """Every recorded reply is found where its strategy predicts it."""
    expected = {
        "json_schema": "content",
        "tool_call": "tool_calls",
        "json_object": "content",
        "fenced": "fenced",
        "json_schema_max_tokens": "content",
        "json_schema_temperature_0": "content",
        "json_schema_temperature_1": "content",
    }
    for name, location in expected.items():
        obj, found = extract_json(_message(name))
        assert found == location, name
        assert obj == PROBE_EXPECTED, name

    # The payload was sent with the probe-selected json_schema strategy.
    for name in ("payload", "truncation"):
        obj, found = extract_json(_message(name))
        assert found == "content", name
        assert isinstance(obj, dict), name
        assert {"analysis", "devices", "rejected_clusters", "merge_suggestions"} <= set(
            obj
        ), name


def test_litellm_replies_validate_first_turn_after_trimming() -> None:
    """Recorded finding: strict json_schema did not make the proxy honour the length caps.

    The payload reply has a 41-character label and a 503-character rationale; the
    truncation reply has a 1385-character analysis, two rationales over 300 and a
    338-character merge reason (03-SPIKE.md follow-up 7). Over-long free text is now
    trimmed locally (B1), so both replies validate on the first turn without a repair
    request, and the trimmed fields are reported.
    """
    expected_trimmed = {
        "payload": ["/devices/1/label", "/devices/1/rationale"],
        "truncation": [
            "/analysis",
            "/devices/1/rationale",
            "/devices/2/rationale",
            "/merge_suggestions/0/reason",
        ],
    }
    for name, paths in expected_trimmed.items():
        valid, errors, trimmed = parse_proposal(
            _message(name), cluster_ids=CLUSTER_IDS, plug_ids=PLUG_IDS
        )
        assert errors == [], (name, errors)
        assert valid is not None, name
        assert trimmed == paths, name

    valid, _, _ = parse_proposal(
        _message("payload"), cluster_ids=CLUSTER_IDS, plug_ids=PLUG_IDS
    )
    assert valid["devices"][1]["label"] == "Balanced two-phase cooker or oven heati…"


async def test_litellm_truncation_case(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """The truncation request ended as the proxy answered it: a normal 200, not truncated.

    The plan expected either a 400 or a 200 flagged prompt_truncated. The recorded
    request (7488 bytes, the preview <data> block repeated three times) was too small
    to overflow the model's context: LiteLLM returned 200 with finish_reason stop and
    prompt_tokens that grew with the request, so truncation_suspected is false. This
    asserts the recorded behaviour (03-SPIKE.md "## Truncation").
    """
    client = LlmClient(hass, BASE_URL, "sk-replay-not-a-secret")
    count = 0

    def _responder(data: dict[str, Any]) -> dict[str, Any]:
        nonlocal count
        count += 1
        return _cassette("truncation")

    mock_llm(aioclient_mock, BASE_URL, _responder)

    resp = await client.async_complete({"model": MODEL, "messages": []})

    assert count == 1
    assert resp.finish_reason == "stop"
    assert resp.model == MODEL
    assert resp.cost_header is not None

    prompt_tokens = resp.usage["prompt_tokens"]
    payload_tokens = _cassette("payload")["json"]["usage"]["prompt_tokens"]
    assert prompt_tokens == 2834
    assert prompt_tokens > payload_tokens
    assert truncation_suspected(prompt_tokens, TRUNCATION_BYTES) is False
    assert (
        truncation_suspected(
            prompt_tokens, TRUNCATION_BYTES, digest_bytes=TRUNCATION_DIGEST_BYTES
        )
        is False
    )


def test_litellm_payload_not_flagged() -> None:
    """The real, non-truncated payload is neither flagged nor undercounted.

    The spike predicted a false positive here for a ratio-calibrated floor: after the
    denser truncation request (0.378 tok/B), 0.8 x 5514 x 0.378 = 1670 > 1664. The
    content-aware floor depends on this request's own digest share only.
    """
    prompt_tokens = _cassette("payload")["json"]["usage"]["prompt_tokens"]
    assert prompt_tokens == 1664

    assert truncation_suspected(prompt_tokens, PAYLOAD_BYTES) is False
    assert (
        truncation_suspected(prompt_tokens, PAYLOAD_BYTES, digest_bytes=PAYLOAD_DIGEST_BYTES)
        is False
    )

    # The conservative estimate shown in the consent notice is not below the measurement.
    assert estimate_tokens(PAYLOAD_BYTES) >= prompt_tokens
