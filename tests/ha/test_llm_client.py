"""Tests for LlmClient async_complete, retries, error mapping, and key secrecy."""

import json
import logging
from typing import Any
import pytest

from homeassistant.core import HomeAssistant

from custom_components.powerguesser import const
from custom_components.powerguesser.const import RESPONSE_MAX_BYTES
from custom_components.powerguesser.llm_client import (
    LlmAuthError,
    LlmBadResponse,
    LlmBudgetError,
    LlmClient,
    LlmConnectError,
    LlmHttpError,
    LlmTimeoutError,
)
from tests.conftest import load_llm_cassette, mock_llm


@pytest.fixture(autouse=True)
def zero_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable retry backoff delays during client tests."""
    monkeypatch.setattr(const, "RETRY_BACKOFF_S", 0.0)


async def test_async_complete_success_200(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """Successful 200 response returns LlmResponse with message, usage, and cost_header."""
    base_url = "http://llm.test:4000/v1"
    key = "sk-test-secret"
    client = LlmClient(hass, base_url, key)

    cassette = load_llm_cassette("client", "ok_with_cost_header")
    mock_llm(aioclient_mock, base_url, lambda data: cassette)

    body = {"model": "m1", "messages": [{"role": "user", "content": "hello"}]}
    resp = await client.async_complete(body)

    assert resp.message == {"role": "assistant", "content": '{"status":"completed"}'}
    assert resp.finish_reason == "stop"
    assert resp.usage == {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110}
    assert resp.model == "m1-2026"
    assert resp.content_bytes == len('{"status":"completed"}'.encode("utf-8"))
    assert resp.cost_header == "0.00042"


async def test_async_complete_500_retries_and_success(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """500, 500, 200 succeeds on the third attempt."""
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, "sk-test-secret")

    req_count = 0

    def _responder(data: dict[str, Any]) -> dict[str, Any]:
        nonlocal req_count
        req_count += 1
        if req_count < 3:
            return load_llm_cassette("client", "error_500")
        return load_llm_cassette("client", "ok_with_cost_header")

    mock_llm(aioclient_mock, base_url, _responder)

    resp = await client.async_complete({"model": "m1"})
    assert resp.finish_reason == "stop"
    assert req_count == 3


async def test_async_complete_500_max_retries_raises(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """500 x 3 raises LlmHttpError(500) after exactly 3 requests."""
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, "sk-test-secret")

    req_count = 0

    def _responder(data: dict[str, Any]) -> dict[str, Any]:
        nonlocal req_count
        req_count += 1
        return load_llm_cassette("client", "error_500")

    mock_llm(aioclient_mock, base_url, _responder)

    with pytest.raises(LlmHttpError) as exc_info:
        await client.async_complete({"model": "m1"})

    assert exc_info.value.status == 500
    assert "Internal server error" in exc_info.value.excerpt
    assert req_count == 3


async def test_async_complete_429_throttling_retried(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """429 with throttling_error and Retry-After is retried."""
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, "sk-test-secret")

    req_count = 0

    def _responder(data: dict[str, Any]) -> dict[str, Any]:
        nonlocal req_count
        req_count += 1
        if req_count < 2:
            return load_llm_cassette("client", "throttle_429")
        return load_llm_cassette("client", "ok_with_cost_header")

    mock_llm(aioclient_mock, base_url, _responder)

    resp = await client.async_complete({"model": "m1"})
    assert resp.finish_reason == "stop"
    assert req_count == 2


async def test_async_complete_429_budget_exceeded_no_retry(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """429 with budget_exceeded raises LlmBudgetError on the first attempt without retry."""
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, "sk-test-secret")

    req_count = 0

    def _responder(data: dict[str, Any]) -> dict[str, Any]:
        nonlocal req_count
        req_count += 1
        return load_llm_cassette("client", "budget_429")

    mock_llm(aioclient_mock, base_url, _responder)

    with pytest.raises(LlmBudgetError):
        await client.async_complete({"model": "m1"})

    assert req_count == 1


async def test_async_complete_non_retryable_4xx(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """400, 401, 403, 404, and 422 are not retried (exactly 1 request each)."""
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, "sk-test-secret")

    cases = [
        ("bad_400", LlmHttpError, 400),
        ("unauthorized_401", LlmAuthError, 401),
        ("forbidden_403", LlmAuthError, 403),
        ("not_found_404", LlmHttpError, 404),
        ("unprocessable_422", LlmHttpError, 422),
    ]

    for cassette_name, exc_type, expected_status in cases:
        aioclient_mock.clear_requests()
        req_count = 0

        def _responder(data: dict[str, Any], name=cassette_name) -> dict[str, Any]:
            nonlocal req_count
            req_count += 1
            return load_llm_cassette("client", name)

        mock_llm(aioclient_mock, base_url, _responder)

        with pytest.raises(exc_type) as exc_info:
            await client.async_complete({"model": "m1"})

        assert exc_info.value.status == expected_status
        assert req_count == 1


async def test_async_complete_timeout_and_connect_errors_retried_3_times(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """TimeoutError and ConnectionError are retried up to 3 times."""
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, "sk-test-secret")

    # Timeout test
    req_count = 0

    def _timeout_responder(data: dict[str, Any]) -> dict[str, Any]:
        nonlocal req_count
        req_count += 1
        return {"exc": "timeout"}

    mock_llm(aioclient_mock, base_url, _timeout_responder)
    with pytest.raises(LlmTimeoutError):
        await client.async_complete({"model": "m1"})
    assert req_count == 3

    # Connect test
    aioclient_mock.clear_requests()
    req_count = 0

    def _connect_responder(data: dict[str, Any]) -> dict[str, Any]:
        nonlocal req_count
        req_count += 1
        return {"exc": "connect"}

    mock_llm(aioclient_mock, base_url, _connect_responder)
    with pytest.raises(LlmConnectError):
        await client.async_complete({"model": "m1"})
    assert req_count == 3


async def test_async_complete_bad_responses(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """Oversized body, invalid JSON, and missing choices raise LlmBadResponse."""
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, "sk-test-secret")

    # 1. Too large
    oversized = "x" * (RESPONSE_MAX_BYTES + 10)
    mock_llm(aioclient_mock, base_url, lambda data: {"status": 200, "text": oversized})
    with pytest.raises(LlmBadResponse, match="response_too_large"):
        await client.async_complete({"model": "m1"})

    # 2. Not JSON
    aioclient_mock.clear_requests()
    mock_llm(aioclient_mock, base_url, lambda data: load_llm_cassette("client", "not_json"))
    with pytest.raises(LlmBadResponse, match="invalid_json"):
        await client.async_complete({"model": "m1"})

    # 3. No choices
    aioclient_mock.clear_requests()
    mock_llm(aioclient_mock, base_url, lambda data: load_llm_cassette("client", "no_choices"))
    with pytest.raises(LlmBadResponse, match="no_choices"):
        await client.async_complete({"model": "m1"})


async def test_api_key_never_leaks_in_logs_or_exceptions(
    hass: HomeAssistant,
    aioclient_mock: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The API key appears only in Authorization header; never in caplog or exception str."""
    caplog.set_level(logging.DEBUG)
    base_url = "http://llm.test:4000/v1"
    secret_key = "sk-super-secret-key-probe-test"
    client = LlmClient(hass, base_url, secret_key)

    # 1. Probe success
    mock_llm(aioclient_mock, base_url, lambda data: load_llm_cassette("probe", "json_schema_ok"))
    await client.async_probe("m1")

    # 2. Probe 401
    aioclient_mock.clear_requests()
    mock_llm(aioclient_mock, base_url, lambda data: load_llm_cassette("probe", "unauthorized"))
    with pytest.raises(LlmAuthError) as exc_p401:
        await client.async_probe("m1")
    assert secret_key not in str(exc_p401.value)

    # 3. Complete 200
    aioclient_mock.clear_requests()
    mock_llm(aioclient_mock, base_url, lambda data: load_llm_cassette("client", "ok_with_cost_header"))
    await client.async_complete({"model": "m1"})

    # 4. Complete 500 x 3
    aioclient_mock.clear_requests()
    mock_llm(aioclient_mock, base_url, lambda data: load_llm_cassette("client", "error_500"))
    with pytest.raises(LlmHttpError) as exc_500:
        await client.async_complete({"model": "m1"})
    assert secret_key not in str(exc_500.value)

    # 5. Complete timeout
    aioclient_mock.clear_requests()
    mock_llm(aioclient_mock, base_url, lambda data: {"exc": "timeout"})
    with pytest.raises(LlmTimeoutError) as exc_timeout:
        await client.async_complete({"model": "m1"})
    assert secret_key not in str(exc_timeout.value)

    # Check caplog across all executed calls
    assert secret_key not in caplog.text


@pytest.mark.parametrize(
    "reply",
    [
        {"text": "[]"},
        {"json": {"choices": ["not-a-dict"]}},
        {"json": {"choices": [{"message": None, "finish_reason": "stop"}]}},
        {"json": {"choices": [{"message": "hello", "finish_reason": "stop"}]}},
        {"json": {"choices": [{"message": {"tool_calls": ["x"]}, "finish_reason": "stop"}]}},
        {
            "json": {
                "choices": [
                    {"message": {"tool_calls": [{"function": None}]}, "finish_reason": "stop"}
                ]
            }
        },
    ],
)
async def test_async_complete_bad_shapes_raise_bad_response(
    hass: HomeAssistant, aioclient_mock: Any, reply: dict[str, Any]
) -> None:
    """Any JSON shape other than the OpenAI one is a classified LlmBadResponse (WR-05)."""
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, "sk-test-secret")
    mock_llm(aioclient_mock, base_url, lambda data: {"status": 200, **reply})

    with pytest.raises(LlmBadResponse) as exc_info:
        await client.async_complete({"model": "m1"})
    assert exc_info.value.reason == "bad_shape"


async def test_async_complete_non_dict_usage_is_dropped(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """A usage field that is not an object is treated as missing (WR-05)."""
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, "sk-test-secret")
    reply = {
        "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
        "usage": "lots",
        "model": "m1",
    }
    mock_llm(aioclient_mock, base_url, lambda data: {"status": 200, "json": reply})

    resp = await client.async_complete({"model": "m1"})
    assert resp.usage == {}


async def test_probe_bad_shape_is_not_a_connect_error(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """A probe reply with a malformed message falls through the ladder instead of crashing (WR-05)."""
    from custom_components.powerguesser.llm_client import LlmUnusableModel

    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, "sk-test-secret")
    mock_llm(
        aioclient_mock,
        base_url,
        lambda data: {"status": 200, "json": {"choices": [{"message": None}]}},
    )
    with pytest.raises(LlmUnusableModel):
        await client.async_probe("m1")

    aioclient_mock.clear_requests()
    mock_llm(aioclient_mock, base_url, lambda data: {"status": 400, "json": {"error": "nope"}})
    with pytest.raises(LlmUnusableModel):
        await client.async_probe("m1")


async def test_async_complete_429_with_string_error_is_http_error(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """A 429 whose error is a plain string is retried and classified, not an AttributeError (WR-05)."""
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, "sk-test-secret")
    mock_llm(aioclient_mock, base_url, lambda data: {"status": 429, "json": {"error": "slow down"}})

    with pytest.raises(LlmHttpError) as exc_info:
        await client.async_complete({"model": "m1"})
    assert exc_info.value.status == 429


ECHO_KEY = "sk-echo-0123456789abcdef"


def test_redact_secrets_and_endpoint_secrets() -> None:
    """Secrets are replaced longest first; empty secrets are ignored; URL passwords are listed."""
    from custom_components.powerguesser.llm_client import (
        REDACTED_MARKER,
        endpoint_secrets,
        redact_secrets,
    )

    assert REDACTED_MARKER == "**REDACTED**"
    assert (
        redact_secrets("a SECRET-KEY b SECRET-KEY", ["SECRET-KEY"])
        == "a **REDACTED** b **REDACTED**"
    )
    assert redact_secrets("plain text", ["", ""]) == "plain text"
    assert redact_secrets("plain text", []) == "plain text"
    # The longer secret contains the shorter one: it must be redacted whole.
    assert (
        redact_secrets("x sk-abc-12-long y", ["sk-abc-12", "sk-abc-12-long"])
        == "x **REDACTED** y"
    )

    secrets = endpoint_secrets("sk-x", "http://user:p%40ss@llm.test/v1")
    assert "sk-x" in secrets
    assert "p%40ss" in secrets
    assert "p@ss" in secrets
    assert len(secrets) == len(set(secrets))
    assert endpoint_secrets("", "http://llm.test/v1") == []


@pytest.mark.parametrize(
    ("status", "response", "expected_requests"),
    [
        (500, {"json": {"error": {"message": f"Upstream failed for key {ECHO_KEY}"}}}, 3),
        (400, {"json": {"error": {"message": f"Bad request, Authorization: Bearer {ECHO_KEY}"}}}, 1),
        (422, {"text": f"unprocessable: header was Bearer {ECHO_KEY}"}, 1),
    ],
    ids=["500-json", "400-json", "422-text"],
)
async def test_error_excerpt_redacts_echoed_key(
    hass: HomeAssistant,
    aioclient_mock: Any,
    status: int,
    response: dict[str, Any],
    expected_requests: int,
) -> None:
    """A provider error that echoes the API key never carries it in excerpt or str(err)."""
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, ECHO_KEY)
    req_count = 0

    def _responder(data: dict[str, Any]) -> dict[str, Any]:
        nonlocal req_count
        req_count += 1
        return {"status": status, **response}

    mock_llm(aioclient_mock, base_url, _responder)

    with pytest.raises(LlmHttpError) as exc_info:
        await client.async_complete({"model": "m1"})

    err = exc_info.value
    assert err.status == status
    assert req_count == expected_requests
    assert ECHO_KEY not in err.excerpt
    assert ECHO_KEY not in str(err)
    assert "**REDACTED**" in err.excerpt


def test_redact_secrets_skips_placeholder_keys() -> None:
    """Keys shorter than REDACT_MIN_SECRET_CHARS are placeholders and are not redacted (IN-09).

    Dummy keys like `x` or `ollama` would otherwise garble the provider's diagnostic text.
    """
    from custom_components.powerguesser.const import REDACT_MIN_SECRET_CHARS
    from custom_components.powerguesser.llm_client import redact_secrets

    assert REDACT_MIN_SECRET_CHARS == 8
    below = "k" * (REDACT_MIN_SECRET_CHARS - 1)
    at = "k" * REDACT_MIN_SECRET_CHARS
    assert redact_secrets(f"token {below} rejected", [below]) == f"token {below} rejected"
    assert redact_secrets(f"token {at} rejected", [at]) == "token **REDACTED** rejected"
    text = "maximum context length exceeded; try ollama pull"
    assert redact_secrets(text, ["x"]) == text
    assert redact_secrets(text, ["ollama"]) == text
    assert redact_secrets(text, ["EMPTY"]) == text
    # A short entry next to a real secret: the real one is still redacted.
    assert redact_secrets("ollama sk-real-0123", ["ollama", "sk-real-0123"]) == (
        "ollama **REDACTED**"
    )


async def test_placeholder_key_keeps_error_text_readable(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """A dummy key such as `x` leaves the provider's error excerpt intact (IN-09)."""
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, "x")
    message = "maximum context length exceeded"
    mock_llm(
        aioclient_mock,
        base_url,
        lambda data: {"status": 400, "json": {"error": {"message": message}}},
    )

    with pytest.raises(LlmHttpError) as exc_info:
        await client.async_complete({"model": "m1"})

    assert exc_info.value.excerpt == message


def _hostile_json(obj: Any) -> str:
    """Serialise like the strictest real encoders: Python ensure_ascii plus Go's &<> escapes.

    `"` and `\\` are escaped by every encoder, non-ASCII by Python's ensure_ascii,
    `&`, `<` and `>` by Go's encoding/json. Outside strings JSON has none of `&<>`,
    so replacing them on the serialised text only touches string contents.
    """
    text = json.dumps(obj)
    return text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


ESCAPED_KEYS = [
    'pa"ss-0123456789',
    "back\\slash-0123456789",
    "schlüssel-0123456789",
    "amp&lt<gt>-0123456789",
]


@pytest.mark.parametrize("key", ESCAPED_KEYS, ids=["quote", "backslash", "non-ascii", "go-amp-lt-gt"])
@pytest.mark.parametrize(
    "shape",
    [
        lambda k: {"detail": f"invalid token {k}"},
        lambda k: {"error": f"invalid token {k}"},
        lambda k: {"error": {"message": [f"invalid token {k}"]}},
        lambda k: {"error": {"code": 7, "info": {f"token {k}": [f"bad {k}"]}}},
    ],
    ids=["detail", "error-string", "message-list", "nested-key"],
)
async def test_error_excerpt_redacts_json_escaped_key(
    hass: HomeAssistant, aioclient_mock: Any, key: str, shape: Any
) -> None:
    """A key that the server's JSON encoder escapes is still redacted (WR-18).

    The raw-body fallback compared the plain key against serialised JSON text, so an
    escaped echo of the key passed through. No escaped or plain form may survive.
    """
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, key)
    body = _hostile_json(shape(key))
    mock_llm(aioclient_mock, base_url, lambda data: {"status": 400, "text": body})

    with pytest.raises(LlmHttpError) as exc_info:
        await client.async_complete({"model": "m1"})

    err = exc_info.value
    escaped_forms = {
        key,
        json.dumps(key)[1:-1],
        json.dumps(key, ensure_ascii=False)[1:-1],
        _hostile_json(key)[1:-1],
        repr(key)[1:-1],
    }
    for form in escaped_forms:
        assert form not in err.excerpt, form
        assert form not in str(err), form
    # The leak-free secret part must not survive either (catches partial redaction).
    assert "0123456789" not in err.excerpt
    assert "**REDACTED**" in err.excerpt


async def test_error_excerpt_keeps_json_body_readable(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """A redacted JSON error body stays readable: decoded text, no escapes (WR-18)."""
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, "schlüssel-0123456789")
    body = _hostile_json({"detail": "Größe <falsch> & token schlüssel-0123456789"})
    mock_llm(aioclient_mock, base_url, lambda data: {"status": 422, "text": body})

    with pytest.raises(LlmHttpError) as exc_info:
        await client.async_complete({"model": "m1"})

    assert exc_info.value.excerpt == '{"detail": "Größe <falsch> & token **REDACTED**"}'


async def test_error_excerpt_deeply_nested_json_does_not_crash(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """JSON nested too deep to walk falls back to the raw-text pass, never RecursionError."""
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, ECHO_KEY)
    depth = 5000
    body = "[" * depth + json.dumps(ECHO_KEY) + "]" * depth
    mock_llm(aioclient_mock, base_url, lambda data: {"status": 400, "text": body})

    with pytest.raises(LlmHttpError) as exc_info:
        await client.async_complete({"model": "m1"})

    assert ECHO_KEY not in exc_info.value.excerpt


async def test_error_excerpt_redacts_key_straddling_cut(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """A key straddling the 500-character cut leaves no fragment (redaction happens first)."""
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, ECHO_KEY)
    body = "x" * 490 + ECHO_KEY + " tail"
    mock_llm(aioclient_mock, base_url, lambda data: {"status": 404, "text": body})

    with pytest.raises(LlmHttpError) as exc_info:
        await client.async_complete({"model": "m1"})

    excerpt = exc_info.value.excerpt
    assert len(excerpt) <= 500
    assert ECHO_KEY[:6] not in excerpt
    assert ECHO_KEY[:6] not in str(exc_info.value)


async def test_empty_key_redacts_nothing(
    hass: HomeAssistant, aioclient_mock: Any
) -> None:
    """With no API key configured the error text is passed through unchanged."""
    base_url = "http://llm.test:4000/v1"
    client = LlmClient(hass, base_url, "")
    mock_llm(aioclient_mock, base_url, lambda data: {"status": 400, "text": "plain error text"})

    with pytest.raises(LlmHttpError) as exc_info:
        await client.async_complete({"model": "m1"})

    assert exc_info.value.excerpt == "plain error text"
