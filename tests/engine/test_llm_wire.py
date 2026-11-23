"""Tests for pure engine llm_wire module."""

from custom_components.powerguesser.engine.llm_wire import (
    LADDER,
    PROBE_EXPECTED,
    PROBE_PROMPT,
    PROBE_SCHEMA,
    TOOL_NAME,
    Strategy,
    apply_profile,
    apply_strategy,
    build_probe_body,
    extract_json,
    model_matches,
)


def test_strategy_enum_and_ladder_order() -> None:
    """Strategy enum values and ladder precedence."""
    assert Strategy.JSON_SCHEMA == "json_schema"
    assert Strategy.TOOL_CALL == "tool_call"
    assert Strategy.JSON_OBJECT == "json_object"
    assert Strategy.FENCED == "fenced"
    assert LADDER == (
        Strategy.JSON_SCHEMA,
        Strategy.TOOL_CALL,
        Strategy.JSON_OBJECT,
        Strategy.FENCED,
    )


def test_apply_strategy_json_schema() -> None:
    """apply_strategy with JSON_SCHEMA sets strict response_format."""
    body: dict[str, Any] = {"model": "m1"}
    apply_strategy(body, schema=PROBE_SCHEMA, name="probe", strategy=Strategy.JSON_SCHEMA)
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "probe",
            "strict": True,
            "schema": PROBE_SCHEMA,
        },
    }
    assert "tools" not in body


def test_apply_strategy_tool_call() -> None:
    """apply_strategy with TOOL_CALL adds tools and forced tool_choice."""
    body: dict[str, Any] = {"model": "m1"}
    apply_strategy(body, schema=PROBE_SCHEMA, name="probe", strategy=Strategy.TOOL_CALL)
    assert "response_format" not in body
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "parameters": PROBE_SCHEMA,
            },
        }
    ]
    assert body["tool_choice"] == {
        "type": "function",
        "function": {"name": TOOL_NAME},
    }


def test_apply_strategy_json_object() -> None:
    """apply_strategy with JSON_OBJECT sets json_object and adds JSON prompt."""
    body: dict[str, Any] = {
        "model": "m1",
        "messages": [{"role": "system", "content": "Initial prompt."}],
    }
    apply_strategy(body, schema=PROBE_SCHEMA, name="probe", strategy=Strategy.JSON_OBJECT)
    assert body["response_format"] == {"type": "json_object"}
    assert "JSON" in body["messages"][0]["content"]
    assert "schema" in body["messages"][0]["content"]


def test_apply_strategy_fenced() -> None:
    """apply_strategy with FENCED sets no response_format and requests fenced block."""
    body: dict[str, Any] = {
        "model": "m1",
        "messages": [{"role": "system", "content": "Initial prompt."}],
    }
    apply_strategy(body, schema=PROBE_SCHEMA, name="probe", strategy=Strategy.FENCED)
    assert "response_format" not in body
    assert "```json" in body["messages"][0]["content"]


def test_apply_profile_parameters() -> None:
    """apply_profile configures max_completion_tokens or max_tokens and temperature."""
    body1 = {"model": "m1"}
    apply_profile(body1, max_tokens=200, profile=None)
    assert body1["max_completion_tokens"] == 200
    # Default (no profile): temperature is not sent (LiteLLM router fallback, 03-09 A2)
    assert "temperature" not in body1

    body2 = {"model": "m1"}
    apply_profile(body2, max_tokens=150, profile={"token_param": "max_tokens", "send_temperature": False})
    assert body2["max_tokens"] == 150
    assert "temperature" not in body2
    assert "max_completion_tokens" not in body2


def test_build_probe_body() -> None:
    """build_probe_body sets model, prompt, strategy, and profile."""
    body = build_probe_body("test-model", Strategy.JSON_SCHEMA)
    assert body["model"] == "test-model"
    assert body["messages"][1]["content"] == PROBE_PROMPT
    assert body["response_format"]["type"] == "json_schema"
    assert body["max_completion_tokens"] == 200


def test_extract_json_from_tool_calls() -> None:
    """extract_json extracts arguments from message tool_calls."""
    msg = {
        "content": None,
        "tool_calls": [
            {
                "id": "1",
                "function": {
                    "name": "submit",
                    "arguments": "{\"ok\": true, \"word\": \"pong\", \"n\": 7}",
                },
            }
        ],
    }
    obj, loc = extract_json(msg)
    assert loc == "tool_calls"
    assert obj == PROBE_EXPECTED


def test_extract_json_from_plain_content() -> None:
    """extract_json extracts direct JSON from content string."""
    msg = {"content": "{\"ok\": true, \"word\": \"pong\", \"n\": 7}"}
    obj, loc = extract_json(msg)
    assert loc == "content"
    assert obj == PROBE_EXPECTED


def test_extract_json_from_fenced_block() -> None:
    """extract_json extracts JSON from markdown fenced code block."""
    msg = {
        "content": "Explanation text\n```json\n{\"ok\": true, \"word\": \"pong\", \"n\": 7}\n```\nDone."
    }
    obj, loc = extract_json(msg)
    assert loc == "fenced"
    assert obj == PROBE_EXPECTED


def test_extract_json_from_outermost_object_in_prose() -> None:
    """extract_json extracts outermost {...} object embedded in prose."""
    msg = {
        "content": "Sure, here is your object: {\"ok\": true, \"word\": \"pong\", \"n\": 7} - enjoy!"
    }
    obj, loc = extract_json(msg)
    assert loc == "content"
    assert obj == PROBE_EXPECTED


def test_extract_json_non_json() -> None:
    """extract_json returns None, 'none' when no valid JSON is present."""
    msg = {"content": "Plain text with no json."}
    obj, loc = extract_json(msg)
    assert loc == "none"
    assert obj is None


def test_apply_profile_temperature_default_off() -> None:
    """A profile without send_temperature sends no temperature."""
    body = {"model": "m1", "temperature": 0.5}
    apply_profile(body, max_tokens=100, profile={"token_param": "max_tokens"})
    assert "temperature" not in body


def test_apply_profile_send_temperature_back_compat() -> None:
    """Stored profiles with send_temperature True still send temperature 0."""
    body = {"model": "m1"}
    apply_profile(
        body,
        max_tokens=100,
        profile={"token_param": "max_completion_tokens", "send_temperature": True},
    )
    assert body["temperature"] == 0


def test_apply_profile_explicit_temperature_value() -> None:
    """profile["temperature"] sets the value sent when send_temperature is True."""
    body = {"model": "m1"}
    apply_profile(body, max_tokens=100, profile={"send_temperature": True, "temperature": 1})
    assert body["temperature"] == 1

    body2 = {"model": "m1"}
    apply_profile(body2, max_tokens=100, profile={"send_temperature": False, "temperature": 1})
    assert "temperature" not in body2


def test_build_probe_body_default_has_no_temperature() -> None:
    """The default probe body carries the model and no temperature."""
    body = build_probe_body("kimi-for-coding", Strategy.JSON_SCHEMA)
    assert body["model"] == "kimi-for-coding"
    assert "temperature" not in body


def test_model_matches_exact_and_case() -> None:
    """Equal names match case-insensitively."""
    assert model_matches("gpt-4o", "gpt-4o")
    assert model_matches("GPT-4o", "gpt-4O")


def test_model_matches_dated_suffix() -> None:
    """A returned name with a dash suffix after the requested name matches."""
    assert model_matches("gpt-4o", "gpt-4o-2024-08-06")
    assert model_matches("m1", "m1-2026")


def test_model_matches_provider_prefix() -> None:
    """A provider/ prefix (up to the last slash) is ignored on both sides."""
    assert model_matches("openai/gpt-4o", "gpt-4o-2024-08-06")
    assert model_matches("kimi-for-coding", "moonshot/kimi-for-coding")
    assert model_matches("a/b/model-x", "model-x")


def test_model_matches_flags_router_fallback() -> None:
    """The kimi -> mimo router fallback seen in the 03-09 spike is a mismatch."""
    assert not model_matches("kimi-for-coding", "xiaomi/mimo-v2.6-flash")
    assert not model_matches("gpt-4o", "gpt-4o2")
    assert not model_matches("gpt-4o-mini", "gpt-4o")


def test_model_matches_missing_returned_is_unknown() -> None:
    """A missing or empty returned model is not flagged."""
    assert model_matches("kimi-for-coding", None)
    assert model_matches("kimi-for-coding", "")
    assert model_matches("kimi-for-coding", "   ")
