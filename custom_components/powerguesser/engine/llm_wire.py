"""Wire formats, probe definitions, and strategy handling for LLM endpoints."""

from collections.abc import Mapping
from enum import StrEnum
import json
import re
from typing import Any

TOOL_NAME = "submit"


class Strategy(StrEnum):
    """Structured-output strategy supported by the endpoint."""

    JSON_SCHEMA = "json_schema"
    TOOL_CALL = "tool_call"
    JSON_OBJECT = "json_object"
    FENCED = "fenced"


LADDER = (
    Strategy.JSON_SCHEMA,
    Strategy.TOOL_CALL,
    Strategy.JSON_OBJECT,
    Strategy.FENCED,
)

PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "word": {"type": "string"},
        "n": {"type": "integer"},
    },
    "required": ["ok", "word", "n"],
    "additionalProperties": False,
}

PROBE_EXPECTED: dict[str, Any] = {"ok": True, "word": "pong", "n": 7}
PROBE_PROMPT = "Return ok=true, word=pong, n=7."


def apply_strategy(
    body: dict[str, Any],
    *,
    schema: dict[str, Any],
    name: str = "probe",
    strategy: Strategy,
) -> dict[str, Any]:
    """Apply structured-output strategy to request body."""
    if strategy == Strategy.JSON_SCHEMA:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "strict": True,
                "schema": schema,
            },
        }
    elif strategy == Strategy.TOOL_CALL:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "parameters": schema,
                },
            }
        ]
        body["tool_choice"] = {
            "type": "function",
            "function": {"name": TOOL_NAME},
        }
    elif strategy == Strategy.JSON_OBJECT:
        body["response_format"] = {"type": "json_object"}
        schema_text = json.dumps(schema)
        addon = f"\nOutput must be valid JSON matching this schema: {schema_text}"
        messages = body.get("messages", [])
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += addon
        else:
            messages.insert(0, {"role": "system", "content": f"You are a helpful assistant.{addon}"})
            body["messages"] = messages
    elif strategy == Strategy.FENCED:
        schema_text = json.dumps(schema)
        addon = f"\nReturn exactly one fenced ```json block containing an object matching this schema: {schema_text}"
        messages = body.get("messages", [])
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += addon
        else:
            messages.insert(0, {"role": "system", "content": f"You are a helpful assistant.{addon}"})
            body["messages"] = messages

    return body


def apply_profile(
    body: dict[str, Any],
    *,
    max_tokens: int,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply parameter profile (token parameter name, temperature) to body.

    Temperature is sent only when the profile asks for it (send_temperature True),
    with the value profile["temperature"] (default 0). The default is not to send
    it: some routers (LiteLLM) silently fall back to another model when the
    requested one rejects temperature 0, so the request never sees a 400.
    """
    param = profile.get("token_param", "max_completion_tokens") if profile else "max_completion_tokens"
    body[param] = max_tokens
    send_temperature = bool(profile.get("send_temperature", False)) if profile else False
    if send_temperature:
        body["temperature"] = profile.get("temperature", 0) if profile else 0
    elif "temperature" in body:
        del body["temperature"]
    return body


def _bare_model_name(name: str) -> str:
    """Return the model name without any provider/ prefix, lower-cased."""
    return name.strip().rsplit("/", 1)[-1].strip().lower()


def model_matches(requested: str, returned: str | None) -> bool:
    """Return whether the model the endpoint answered with is the one requested.

    A provider/ prefix (everything up to the last "/") is ignored on both sides and
    the comparison is case-insensitive. A returned name equal to the requested one,
    or the requested one followed by "-" and a suffix (a dated snapshot such as
    gpt-4o -> gpt-4o-2024-08-06), matches. A missing or empty returned model is
    unknown and is not flagged.
    """
    if not returned or not str(returned).strip():
        return True
    req = _bare_model_name(str(requested))
    ret = _bare_model_name(str(returned))
    if not req or not ret:
        return True
    return ret == req or ret.startswith(f"{req}-")


def build_probe_body(
    model: str,
    strategy: Strategy,
    profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build request body for a probe rung."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": PROBE_PROMPT},
        ],
    }
    apply_strategy(body, schema=PROBE_SCHEMA, name="probe", strategy=strategy)
    apply_profile(body, max_tokens=200, profile=profile)
    return body


def extract_json(message: Mapping[str, Any]) -> tuple[Any | None, str]:
    """Extract JSON object from a message (tool_calls, content, fenced block, or outermost object)."""
    tool_calls = message.get("tool_calls")
    if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
        first = tool_calls[0]
        if isinstance(first, dict):
            fn = first.get("function", {})
            args = fn.get("arguments")
            if args:
                try:
                    return json.loads(args), "tool_calls"
                except Exception:
                    pass

    content = message.get("content")
    if content and isinstance(content, str):
        text = content.strip()
        try:
            return json.loads(text), "content"
        except Exception:
            pass

        # Try fenced code block
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if m:
            try:
                return json.loads(m.group(1).strip()), "fenced"
            except Exception:
                pass

        # Try outermost {...}
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1]), "content"
            except Exception:
                pass

    return None, "none"
