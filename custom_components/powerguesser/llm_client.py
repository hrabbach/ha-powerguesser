"""OpenAI-compatible LLM client and capability probe."""

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import ipaddress
import json
import logging
import random
from typing import Any
import yarl

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util import dt as dt_util

from . import const
from .const import (
    ERROR_BODY_MAX_CHARS,
    LLM_CONNECT_TIMEOUT_S,
    LLM_MAX_ATTEMPTS,
    LLM_TIMEOUT_S,
    PROBE_RUNG_TIMEOUT_S,
    REDACT_MIN_SECRET_CHARS,
    RESPONSE_MAX_BYTES,
    RETRY_BACKOFF_S,
)
from .engine.llm_wire import (
    LADDER,
    PROBE_EXPECTED,
    Strategy,
    build_probe_body,
    extract_json,
    model_matches,
)

_LOGGER = logging.getLogger(__name__)

# Replaces a secret in provider error text (same marker as HA diagnostics redaction).
REDACTED_MARKER = "**REDACTED**"


def endpoint_secrets(api_key: str, base_url: str) -> list[str]:
    """Return every secret that must never leave the client in error text (LLM-01, D-10).

    The non-empty API key plus the base-URL userinfo password, raw and percent-decoded.
    normalise_base_url rejects userinfo, so the password entries are defence in depth.
    """
    secrets: list[str] = []
    candidates: list[str | None] = [api_key]
    try:
        url = yarl.URL(base_url)
        candidates += [url.raw_password, url.password]
    except (TypeError, ValueError):
        pass
    for secret in candidates:
        if secret and secret not in secrets:
            secrets.append(secret)
    return secrets


def redact_secrets(text: str, secrets: Sequence[str]) -> str:
    """Replace every secret in text with REDACTED_MARKER, longest first.

    Secrets shorter than REDACT_MIN_SECRET_CHARS are placeholder keys and are skipped (IN-09).
    """
    real = {s for s in secrets if len(s) >= REDACT_MIN_SECRET_CHARS}
    for secret in sorted(real, key=len, reverse=True):
        text = text.replace(secret, REDACTED_MARKER)
    return text


def secret_text_forms(secrets: Sequence[str]) -> list[str]:
    """Return each redactable secret plus the forms a JSON encoder escapes it to (IN-10).

    For text whose JSON structure is lost (a stored, already-cut excerpt): the plain
    secret, Python's ensure_ascii and ensure_ascii=False escapes, and Go's &<> escapes.
    """
    forms: list[str] = []
    for secret in secrets:
        if len(secret) < REDACT_MIN_SECRET_CHARS:
            continue
        ascii_form = json.dumps(secret)[1:-1]
        go_form = (
            ascii_form.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
        )
        for form in (secret, ascii_form, json.dumps(secret, ensure_ascii=False)[1:-1], go_form):
            if form not in forms:
                forms.append(form)
    return forms


def redact_json_obj(obj: Any, secrets: Sequence[str]) -> Any:
    """Return a copy of a decoded JSON value with secrets redacted in every string and key (WR-18).

    Redacting the decoded structure is what makes JSON-escaped echoes of a key (`\\"`,
    `\\\\`, `\\u00fc`, Go's `\\u0026`) visible to redact_secrets at all.
    """
    if isinstance(obj, str):
        return redact_secrets(obj, secrets)
    if isinstance(obj, dict):
        return {
            redact_secrets(str(k), secrets): redact_json_obj(v, secrets) for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [redact_json_obj(v, secrets) for v in obj]
    return obj


class LlmError(Exception):
    """Base exception for LLM client errors."""


class LlmAuthError(LlmError):
    """Authentication failed (401/403)."""

    def __init__(self, status: int) -> None:
        """Initialize auth error."""
        super().__init__(f"Authentication failed with status {status}")
        self.status = status


class LlmConnectError(LlmError):
    """Connection failure or network error."""


class LlmTimeoutError(LlmError):
    """Request timed out."""


class LlmBudgetError(LlmError):
    """Budget exceeded error (429 with budget_exceeded)."""


class LlmHttpError(LlmError):
    """HTTP error with status code and error excerpt."""

    def __init__(self, status: int, excerpt: str = "") -> None:
        """Initialize HTTP error."""
        super().__init__(f"HTTP {status}: {excerpt[:ERROR_BODY_MAX_CHARS]}")
        self.status = status
        self.excerpt = excerpt[:ERROR_BODY_MAX_CHARS]


class LlmBadResponse(LlmError):
    """Malformed or unexpected response."""

    def __init__(self, reason: str) -> None:
        """Initialize bad response error."""
        super().__init__(f"Bad response: {reason}")
        self.reason = reason


class LlmUnusableModel(LlmError):
    """Model did not validate on any probe ladder rung."""


@dataclass(frozen=True)
class EndpointLocation:
    """Endpoint location metadata."""

    host: str
    local: bool
    transport: str


@dataclass(frozen=True)
class LlmResponse:
    """Standardized response from LLM completion."""

    message: dict[str, Any]
    finish_reason: str | None
    usage: dict[str, Any]
    model: str
    content_bytes: int
    cost_header: str | None


def normalise_base_url(raw: str) -> str:
    """Normalise base URL.

    Must be http or https, host required, no userinfo, query, or fragment.
    Trailing /v1 matched case-insensitively and normalised to exactly one /v1.
    """
    try:
        url = yarl.URL(raw)
    except Exception as exc:
        raise ValueError("invalid_url") from exc

    if url.scheme not in ("http", "https"):
        raise ValueError("invalid_url")
    if not url.host:
        raise ValueError("invalid_url")
    if url.user is not None or url.password is not None:
        raise ValueError("invalid_url")
    if url.query_string or url.query:
        raise ValueError("invalid_url")
    if url.fragment:
        raise ValueError("invalid_url")

    path = url.path.rstrip("/")
    if path.lower().endswith("/v1"):
        path = path[:-3]

    port_str = f":{url.explicit_port}" if url.explicit_port else ""
    return f"{url.scheme}://{url.raw_host}{port_str}{path}/v1"


def endpoint_location(base_url: str) -> EndpointLocation:
    """Extract host, local flag, and transport from base URL."""
    url = yarl.URL(base_url)
    host = (url.host or "").lower()
    transport = url.scheme

    local = False
    if host in ("localhost", "127.0.0.1", "::1") or host.endswith(".local"):
        local = True
    else:
        try:
            ip = ipaddress.ip_address(host)
            if ip.is_loopback or ip.is_private or ip.is_link_local:
                local = True
        except ValueError:
            pass

    return EndpointLocation(host=host, local=local, transport=transport)


def endpoint_origin(base_url: str) -> tuple[str, str, int | None]:
    """Return (scheme, host, port) of a base URL; the default port is filled in for http/https."""
    url = yarl.URL(base_url)
    return (url.scheme.lower(), (url.host or "").lower(), url.port)


class LlmClient:
    """Async client for OpenAI-compatible LLM endpoints."""

    def __init__(self, hass: HomeAssistant, base_url: str, api_key: str) -> None:
        """Initialize LLM client."""
        self.hass = hass
        self.base_url = base_url
        self.api_key = api_key
        self._secrets = endpoint_secrets(api_key, base_url)
        self._session = async_get_clientsession(hass)
        self._location = endpoint_location(base_url)

    def _headers(self) -> dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _post(
        self, body: dict[str, Any], timeout_s: float
    ) -> tuple[int, dict[str, Any] | None, Mapping[str, str], bytes]:
        """Execute POST to /chat/completions with strict redirects and timeouts."""
        url = f"{self.base_url}/chat/completions"
        timeout = aiohttp.ClientTimeout(
            total=timeout_s, sock_connect=LLM_CONNECT_TIMEOUT_S
        )
        try:
            async with self._session.post(
                url,
                json=body,
                headers=self._headers(),
                allow_redirects=False,
                timeout=timeout,
            ) as resp:
                raw_bytes = await resp.read()
                _LOGGER.debug("POST %s status %d", self._location.host, resp.status)
                parsed: dict[str, Any] | None = None
                try:
                    parsed = json.loads(raw_bytes)
                except Exception:
                    pass
                return resp.status, parsed, resp.headers, raw_bytes
        except TimeoutError as err:
            raise LlmTimeoutError("Request timed out") from err
        except aiohttp.ClientConnectionError as err:
            raise LlmConnectError("Connection error") from err
        except aiohttp.ClientError as err:
            raise LlmConnectError(f"HTTP client error: {type(err).__name__}") from err

    async def async_probe(self, model: str) -> dict[str, Any]:
        """Probe endpoint capability ladder and determine parameter profile.

        Temperature is not sent by default: a LiteLLM router can silently fall back
        to another model when the requested one rejects temperature 0 (03-09 spike),
        so the probe would never see the 400 it relies on. The model the endpoint
        answered with is compared with the requested one and a mismatch is stored
        and logged, never failed.
        """
        profile: dict[str, Any] = {
            "token_param": "max_completion_tokens",
            "send_temperature": False,
        }

        for strategy in LADDER:
            while True:
                body = build_probe_body(model, strategy, profile)
                status, parsed, headers, _ = await self._post(body, PROBE_RUNG_TIMEOUT_S)

                if status in (401, 403):
                    raise LlmAuthError(status)

                err_info = parsed.get("error") if isinstance(parsed, dict) else None
                if status == 400 and isinstance(err_info, dict):
                    err_msg = str(err_info.get("message", ""))
                    err_param = str(err_info.get("param", ""))
                    retry_needed = False

                    if "max_completion_tokens" in err_msg or "max_completion_tokens" in err_param:
                        if profile["token_param"] != "max_tokens":
                            profile["token_param"] = "max_tokens"
                            retry_needed = True

                    if "temperature" in err_msg or "temperature" in err_param:
                        if profile["send_temperature"]:
                            profile["send_temperature"] = False
                            retry_needed = True

                    if retry_needed:
                        continue

                break

            if status == 200 and isinstance(parsed, dict):
                choices = parsed.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    first_choice = choices[0]
                    if isinstance(first_choice, dict):
                        msg = first_choice.get("message", {})
                        if isinstance(msg, dict) and not msg.get("refusal"):
                            extracted, _ = extract_json(msg)
                            if extracted == PROBE_EXPECTED:
                                model_returned = parsed.get("model", model)
                                mismatch = not model_matches(model, model_returned)
                                if mismatch:
                                    _LOGGER.warning(
                                        "LLM endpoint %s answered the probe for model %s "
                                        "with model %s; the router may have fallen back "
                                        "to another model",
                                        self._location.host,
                                        model,
                                        model_returned,
                                    )
                                return {
                                    "strategy": strategy.value,
                                    "token_param": profile["token_param"],
                                    "send_temperature": profile["send_temperature"],
                                    "model_returned": model_returned,
                                    "model_mismatch": mismatch,
                                    "probed_at": dt_util.utcnow().isoformat(),
                                }

        raise LlmUnusableModel("Model did not return valid probe response on any rung")

    async def async_complete(
        self,
        body: dict[str, Any],
        *,
        timeout_s: float = LLM_TIMEOUT_S,
        on_billable_failure: Callable[[bool], None] | None = None,
    ) -> LlmResponse:
        """Execute chat completion call with bounded retries and response validation.

        on_billable_failure(will_retry) is called for every attempt that ended in a
        timeout or a 500/502/504 after the body was sent: the provider may have processed
        and billed it, but no usage comes back. The caller counts it against its caps and
        may raise to stop a retry that would exceed them.
        """
        attempt = 0
        last_error: Exception | None = None

        while attempt < const.LLM_MAX_ATTEMPTS:
            attempt += 1
            status: int | None = None
            parsed: dict[str, Any] | None = None
            headers: Mapping[str, str] = {}
            raw_bytes: bytes = b""

            try:
                status, parsed, headers, raw_bytes = await self._post(body, timeout_s)
            except (LlmTimeoutError, LlmConnectError) as err:
                last_error = err
                if isinstance(err, LlmTimeoutError) and on_billable_failure is not None:
                    on_billable_failure(attempt < const.LLM_MAX_ATTEMPTS)
                if attempt >= const.LLM_MAX_ATTEMPTS:
                    raise
                sleep_s = (
                    0.0
                    if const.RETRY_BACKOFF_S == 0
                    else (
                        const.RETRY_BACKOFF_S * (2 ** (attempt - 1))
                        + random.uniform(0, 1)
                    )
                )
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                continue

            # Check response size cap
            if len(raw_bytes) > const.RESPONSE_MAX_BYTES:
                raise LlmBadResponse("response_too_large")

            # Check status 200
            if status == 200:
                if parsed is None:
                    raise LlmBadResponse("invalid_json")
                # The endpoint is a trust boundary: check every shape we index into (WR-05)
                if not isinstance(parsed, dict):
                    raise LlmBadResponse("bad_shape")
                choices = parsed.get("choices")
                if not choices or not isinstance(choices, list) or len(choices) == 0:
                    raise LlmBadResponse("no_choices")
                choice = choices[0]
                if not isinstance(choice, dict):
                    raise LlmBadResponse("bad_shape")
                msg = choice.get("message", {})
                if not isinstance(msg, dict):
                    raise LlmBadResponse("bad_shape")
                tool_calls_raw = msg.get("tool_calls")
                if tool_calls_raw is not None and (
                    not isinstance(tool_calls_raw, list)
                    or any(
                        not isinstance(tc, dict) or not isinstance(tc.get("function"), dict)
                        for tc in tool_calls_raw
                    )
                ):
                    raise LlmBadResponse("bad_shape")
                usage = parsed.get("usage")

                # Calculate content_bytes
                content_bytes = 0
                tool_calls = msg.get("tool_calls")
                if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                    args = tool_calls[0].get("function", {}).get("arguments", "")
                    if isinstance(args, str):
                        content_bytes = len(args.encode("utf-8"))
                else:
                    content_str = msg.get("content", "")
                    if isinstance(content_str, str):
                        content_bytes = len(content_str.encode("utf-8"))

                return LlmResponse(
                    message=msg,
                    finish_reason=choice.get("finish_reason"),
                    usage=usage if isinstance(usage, dict) else {},
                    model=str(parsed.get("model") or ""),
                    content_bytes=content_bytes,
                    cost_header=headers.get("x-litellm-response-cost"),
                )

            # Extract the error excerpt: full text first, then redact the endpoint secrets,
            # then cut, so a secret straddling the cut leaves no fragment (LLM-01, D-10).
            # A JSON body is redacted on its decoded values and re-serialised: its raw text
            # may carry the key JSON-escaped, which a plain-text search misses (WR-18).
            # The raw-text pass stays for bodies that are not JSON.
            error_text = ""
            try:
                if parsed and isinstance(parsed, dict) and "error" in parsed:
                    err_dict = parsed["error"]
                    if isinstance(err_dict, dict) and "message" in err_dict:
                        message = err_dict["message"]
                        if isinstance(message, str):
                            error_text = message
                        else:
                            error_text = json.dumps(
                                redact_json_obj(message, self._secrets), ensure_ascii=False
                            )
                if not error_text and raw_bytes and parsed is not None:
                    error_text = json.dumps(
                        redact_json_obj(parsed, self._secrets), ensure_ascii=False
                    )
            except RecursionError:
                # Nested deep enough to parse but not to walk: the raw-text pass only.
                error_text = ""
            if not error_text and raw_bytes:
                error_text = raw_bytes.decode("utf-8", errors="replace")
            excerpt = redact_secrets(error_text, self._secrets)[:const.ERROR_BODY_MAX_CHARS]

            # Check non-retryable statuses
            if status in (401, 403):
                raise LlmAuthError(status)

            if status == 429:
                err_dict = parsed.get("error") if isinstance(parsed, dict) else None
                if not isinstance(err_dict, dict):
                    err_dict = {}
                err_type = str(err_dict.get("type", "")).lower()
                err_code = str(err_dict.get("code", "")).lower()
                if "budget_exceeded" in err_type or "budget_exceeded" in err_code:
                    raise LlmBudgetError("budget_exceeded")

            # Check non-retryable 4xx
            if status in (400, 404, 422):
                raise LlmHttpError(status, excerpt)

            # Retryable statuses: 408, 429 (throttling), 500, 502, 503, 504
            retryable = status in (408, 429, 500, 502, 503, 504)
            if not retryable:
                raise LlmHttpError(status, excerpt)

            if status in (500, 502, 504) and on_billable_failure is not None:
                on_billable_failure(attempt < const.LLM_MAX_ATTEMPTS)

            if attempt >= const.LLM_MAX_ATTEMPTS:
                raise LlmHttpError(status, excerpt)

            # Retry sleep calculation
            retry_after_hdr = headers.get("Retry-After")
            retry_after_s: float | None = None
            if retry_after_hdr:
                try:
                    retry_after_s = float(retry_after_hdr)
                except (ValueError, TypeError):
                    retry_after_s = None

            if const.RETRY_BACKOFF_S == 0:
                sleep_s = 0.0
            elif retry_after_s is not None:
                sleep_s = min(retry_after_s, 60.0)
            else:
                sleep_s = const.RETRY_BACKOFF_S * (2 ** (attempt - 1)) + random.uniform(
                    0, 1
                )

            if sleep_s > 0:
                await asyncio.sleep(sleep_s)

        if last_error:
            raise last_error
        raise LlmHttpError(500, "Max attempts reached")
