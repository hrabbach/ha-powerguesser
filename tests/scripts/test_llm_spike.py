"""Tests for dev spike script (scripts/llm_spike.py)."""

from collections.abc import Mapping
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
import pytest

_SPIKE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "llm_spike.py"


def _load_spike_module():
    """Load scripts/llm_spike.py as a module via spec_from_file_location."""
    if not _SPIKE_PATH.exists():
        pytest.fail(f"Script {_SPIKE_PATH} does not exist yet")
    spec = importlib.util.spec_from_file_location("llm_spike", _SPIKE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec from {_SPIKE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_redact():
    mod = _load_spike_module()
    redact = mod.redact

    secret = "sk-super-secret-12345"
    data = {
        "headers": {"Authorization": f"Bearer {secret}", "Content-Type": "application/json"},
        "nested": [
            {"msg": f"Key is {secret}!"},
            123,
            True,
            None,
        ],
        "other": "normal text",
    }

    result = redact(data, secret)
    assert result["headers"]["Authorization"] == "Bearer **REDACTED**"
    assert result["nested"][0]["msg"] == "Key is **REDACTED**!"
    assert result["nested"][1] == 123
    assert result["other"] == "normal text"

    # Deep copy verification
    assert data["headers"]["Authorization"] == f"Bearer {secret}"

    # Empty secret leaves object unchanged
    assert redact(data, "") == data
    assert redact(data, None) == data


def test_keep_headers():
    mod = _load_spike_module()
    keep_headers = mod.keep_headers

    headers = {
        "X-LiteLLM-Response-Cost": "0.00042",
        "x-litellm-model-id": "model-xyz",
        "Authorization": "Bearer secret",
        "Set-Cookie": "session=123",
        "Content-Type": "application/json",
    }

    kept = keep_headers(headers)
    assert kept == {
        "x-litellm-response-cost": "0.00042",
        "x-litellm-model-id": "model-xyz",
    }


def test_render_table():
    mod = _load_spike_module()
    render_table = mod.render_table

    results = [
        {
            "rung": "json_schema",
            "variant": "default",
            "status": 200,
            "finish_reason": "stop",
            "json_location": "content",
            "valid": True,
            "prompt_tokens": 150,
            "completion_tokens": 25,
            "bytes": 450,
            "est_tokens": 150,
            "requested_model": "gpt-4o",
            "model": "gpt-4o-2024-08-06",
            "model_mismatch": False,
            "cost_header": "0.0005",
            "error": "",
        }
    ]

    table = render_table(results)
    assert "| rung | variant | status | finish_reason | json_location | valid |" in table
    assert "| json_schema | default | 200 | stop | content | True |" in table
    assert "| requested_model | model | model_mismatch | cost_header | error |" in table
    assert "| gpt-4o | gpt-4o-2024-08-06 | False | 0.0005 |  |" in table


def test_missing_env_vars(monkeypatch, capsys):
    mod = _load_spike_module()

    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    with pytest.raises(SystemExit) as exc_info:
        mod.main(["probe"])
    assert exc_info.value.code == 2

    captured = capsys.readouterr()
    assert "LLM_BASE_URL" in captured.err or "LLM_BASE_URL" in captured.out

    # Now provide BASE_URL but miss MODEL
    monkeypatch.setenv("LLM_BASE_URL", "http://test-proxy:8000")
    with pytest.raises(SystemExit) as exc_info:
        mod.main(["probe"])
    assert exc_info.value.code == 2

    captured = capsys.readouterr()
    assert "LLM_MODEL" in captured.err or "LLM_MODEL" in captured.out


def test_stdout_never_contains_key(monkeypatch, capsys, tmp_path):
    mod = _load_spike_module()

    fake_key = "sk-super-secret-token-do-not-leak"
    monkeypatch.setenv("LLM_BASE_URL", "http://fake-proxy:8000")
    monkeypatch.setenv("LLM_MODEL", "fake-model")
    monkeypatch.setenv("LLM_API_KEY", fake_key)

    async def fake_post(url, headers, raw_body, timeout_s=600.0):
        # Even if proxy echoes headers or key in response:
        return (
            200,
            {
                "x-litellm-response-cost": "0.0001",
                "x-litellm-model-id": "fake-model",
                "server": "LiteLLM/1.0",
                "authorization": f"Bearer {fake_key}",
            },
            {
                "choices": [
                    {
                        "message": {"content": '{"ok": true, "word": "pong", "n": 7}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10},
                "model": "fake-model",
            },
            "",
        )

    monkeypatch.setattr(mod, "_send_request", fake_post)

    out_dir = tmp_path / "spike-out"
    ret = mod.main(["probe", "--out", str(out_dir)])
    assert ret == 0

    captured = capsys.readouterr()
    assert fake_key not in captured.out
    assert fake_key not in captured.err
    # Output must only contain host and table path
    assert "Host: fake-proxy:8000\n" in captured.out
    assert "Table: " in captured.out

    # Check cassettes.json
    cassettes_file = out_dir / "cassettes.json"
    assert cassettes_file.exists()
    content = cassettes_file.read_text(encoding="utf-8")
    assert fake_key not in content
    assert "Authorization" not in content
    assert "Bearer" not in content

    # Check table.md
    table_file = out_dir / "table.md"
    assert table_file.exists()
    t_content = table_file.read_text(encoding="utf-8")
    assert fake_key not in t_content


def test_subcommand_probe_sends_seven_independent_requests(monkeypatch, tmp_path):
    mod = _load_spike_module()

    monkeypatch.setenv("LLM_BASE_URL", "http://proxy:8000")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    requests_sent = []

    async def fake_post(url, headers, raw_body, timeout_s=600.0):
        body = json.loads(raw_body.decode("utf-8"))
        requests_sent.append(body)
        return (
            200,
            {"x-litellm-response-cost": "0.0001", "x-litellm-model-id": "test-model"},
            {
                "choices": [
                    {
                        "message": {"content": '{"ok": true, "word": "pong", "n": 7}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                "model": "test-model",
            },
            "",
        )

    monkeypatch.setattr(mod, "_send_request", fake_post)

    out_dir = tmp_path / "spike-out"
    ret = mod.main(["probe", "--out", str(out_dir)])
    assert ret == 0
    # 4 ladder rungs + 3 json_schema variants (max_tokens, temperature 0, temperature 1)
    assert len(requests_sent) == 7

    # The default profile sends no temperature (03-09 A2); only the two variants do
    temps = [b.get("temperature") for b in requests_sent]
    assert temps == [None, None, None, None, None, 0, 1]
    assert all(b["model"] == "test-model" for b in requests_sent)
    assert "max_tokens" in requests_sent[4] and "max_completion_tokens" not in requests_sent[4]

    cassettes = json.loads((out_dir / "cassettes.json").read_text(encoding="utf-8"))
    assert set(cassettes.keys()) == {
        "json_schema",
        "tool_call",
        "json_object",
        "fenced",
        "json_schema_max_tokens",
        "json_schema_temperature_0",
        "json_schema_temperature_1",
    }
    for entry in cassettes.values():
        assert "request_sha256" in entry
        assert entry["status"] == 200
        assert "headers" in entry


def test_payload_and_truncation_accumulation(monkeypatch, tmp_path):
    mod = _load_spike_module()

    monkeypatch.setenv("LLM_BASE_URL", "http://proxy:8000")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    sent_bodies = []

    async def fake_post(url, headers, raw_body, timeout_s=600.0):
        body = json.loads(raw_body.decode("utf-8"))
        sent_bodies.append(body)
        return (
            200,
            {"x-litellm-response-cost": "0.0002", "x-litellm-model-id": "test-model"},
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "analysis": "good",
                                    "schema_version": 1,
                                    "devices": [],
                                }
                            )
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 500, "completion_tokens": 50},
                "model": "test-model",
            },
            "",
        )

    monkeypatch.setattr(mod, "_send_request", fake_post)

    out_dir = tmp_path / "spike-out"

    # Step 1: probe
    assert mod.main(["probe", "--out", str(out_dir)]) == 0

    # Step 2: payload
    body_file = tmp_path / "preview_body.json"
    body_data = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are assistant."},
            {"role": "user", "content": "<data>\nid=kettle n=5\n</data>\nReturn proposal."},
        ],
    }
    body_file.write_text(json.dumps(body_data), encoding="utf-8")

    assert mod.main(["payload", "--body", str(body_file), "--strategy", "json_schema", "--out", str(out_dir)]) == 0

    # Step 3: truncation
    assert mod.main(["truncation", "--body", str(body_file), "--repeat", "3", "--out", str(out_dir)]) == 0

    # Verify truncation body repeated data 3 times
    trunc_body = sent_bodies[-1]
    trunc_user_msg = trunc_body["messages"][1]["content"]
    assert trunc_user_msg.count("id=kettle n=5") == 3

    # Verify all 9 entries exist in cassettes.json and 9 rows in table.md
    cassettes = json.loads((out_dir / "cassettes.json").read_text(encoding="utf-8"))
    assert len(cassettes) == 9
    assert "payload" in cassettes
    assert "truncation" in cassettes

    table_lines = (out_dir / "table.md").read_text(encoding="utf-8").strip().splitlines()
    # 2 header lines + 9 rows = 11 lines
    data_rows = [line for line in table_lines if line.startswith("|") and not line.startswith("| ---") and not line.startswith("| rung")]
    assert len(data_rows) == 9


def test_probe_records_model_mismatch(monkeypatch, tmp_path):
    """Each result records the requested model and whether the proxy answered with another."""
    mod = _load_spike_module()

    monkeypatch.setenv("LLM_BASE_URL", "http://proxy:8000")
    monkeypatch.setenv("LLM_MODEL", "kimi-for-coding")
    monkeypatch.setenv("LLM_API_KEY", "")

    async def fake_post(url, headers, raw_body, timeout_s=600.0):
        body = json.loads(raw_body.decode("utf-8"))
        # Router fallback only when temperature is sent (as seen live in 03-09)
        returned = "xiaomi/mimo-v2.6-flash" if "temperature" in body else "kimi-for-coding"
        return (
            200,
            {},
            {
                "choices": [
                    {
                        "message": {"content": '{"ok": true, "word": "pong", "n": 7}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                "model": returned,
            },
            "",
        )

    monkeypatch.setattr(mod, "_send_request", fake_post)

    out_dir = tmp_path / "spike-out"
    assert mod.main(["probe", "--out", str(out_dir)]) == 0

    results = {
        r["name"]: r
        for r in json.loads((out_dir / ".results.json").read_text(encoding="utf-8"))
    }
    assert results["json_schema"]["requested_model"] == "kimi-for-coding"
    assert results["json_schema"]["model_mismatch"] is False
    assert results["json_schema_temperature_0"]["model_mismatch"] is True
    assert results["json_schema_temperature_0"]["model"] == "xiaomi/mimo-v2.6-flash"
    assert results["json_schema_temperature_1"]["model_mismatch"] is True

    table = (out_dir / "table.md").read_text(encoding="utf-8")
    assert "| kimi-for-coding | xiaomi/mimo-v2.6-flash | True |" in table


def test_model_mismatch_blank_without_returned_model(monkeypatch, tmp_path):
    """An error response without a model leaves model_mismatch unknown (None)."""
    mod = _load_spike_module()

    monkeypatch.setenv("LLM_BASE_URL", "http://proxy:8000")
    monkeypatch.setenv("LLM_MODEL", "kimi-for-coding")
    monkeypatch.setenv("LLM_API_KEY", "")

    async def fake_post(url, headers, raw_body, timeout_s=600.0):
        return (400, {}, {"error": {"message": "bad", "code": "400"}}, "")

    monkeypatch.setattr(mod, "_send_request", fake_post)

    out_dir = tmp_path / "spike-out"
    assert mod.main(["probe", "--out", str(out_dir)]) == 0
    results = json.loads((out_dir / ".results.json").read_text(encoding="utf-8"))
    assert all(r["model_mismatch"] is None for r in results)
    assert all(r["requested_model"] == "kimi-for-coding" for r in results)


def _all_outputs(out_dir: Path, captured) -> dict[str, str]:
    """Every artefact the spike writes plus its stdout/stderr (T-03-35)."""
    return {
        "table.md": (out_dir / "table.md").read_text(encoding="utf-8"),
        ".results.json": (out_dir / ".results.json").read_text(encoding="utf-8"),
        "cassettes.json": (out_dir / "cassettes.json").read_text(encoding="utf-8"),
        "stdout": captured.out,
        "stderr": captured.err,
    }


def test_auth_error_echoing_key_is_redacted_everywhere(monkeypatch, capsys, tmp_path):
    """A 401 whose error body echoes the key must not store or print it anywhere (T-03-35)."""
    mod = _load_spike_module()

    fake_key = "sk-echoed-secret-key-0123456789"
    monkeypatch.setenv("LLM_BASE_URL", "http://proxy:8000")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", fake_key)

    async def fake_post(url, headers, raw_body, timeout_s=600.0):
        error = {
            "error": {
                "message": f"Authentication Error, invalid key {fake_key}",
                "type": "auth_error",
                "code": "401",
            }
        }
        return (401, {}, error, json.dumps(error))

    monkeypatch.setattr(mod, "_send_request", fake_post)

    out_dir = tmp_path / "spike-out"
    assert mod.main(["probe", "--out", str(out_dir)]) == 0

    outputs = _all_outputs(out_dir, capsys.readouterr())
    for name, content in outputs.items():
        assert fake_key not in content, f"key leaked into {name}"
    results = json.loads(outputs[".results.json"])
    assert all(r["status"] == 401 for r in results)
    assert all("**REDACTED**" in r["error"] for r in results)


@pytest.mark.parametrize("status", [401, 400])
@pytest.mark.parametrize(
    "fake_key",
    ['sk-"quoted"-secret-0123456789', "sk-schlüssel-secret-0123456789"],
    ids=["quote", "non-ascii"],
)
def test_json_escaped_key_in_detail_body_is_redacted_everywhere(
    monkeypatch, capsys, tmp_path, fake_key, status
):
    """A key the JSON encoder escapes, echoed in a {"detail": ...} body, is stored nowhere (T-03-35).

    The excerpt used to be redact_all(json.dumps(error_obj)): json.dumps escapes `"` and
    non-ASCII first, so the plain-key search missed the echoed key.
    """
    mod = _load_spike_module()

    monkeypatch.setenv("LLM_BASE_URL", "http://proxy:8000")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", fake_key)

    async def fake_post(url, headers, raw_body, timeout_s=600.0):
        error = {"detail": f"invalid token {fake_key}"}
        return (status, {}, error, json.dumps(error))

    monkeypatch.setattr(mod, "_send_request", fake_post)

    out_dir = tmp_path / "spike-out"
    assert mod.main(["probe", "--out", str(out_dir)]) == 0

    outputs = _all_outputs(out_dir, capsys.readouterr())
    forms = {
        fake_key,
        json.dumps(fake_key)[1:-1],
        json.dumps(fake_key, ensure_ascii=False)[1:-1],
    }
    for name, content in outputs.items():
        for form in forms:
            assert form not in content, f"key form {form!r} leaked into {name}"
        assert "0123456789" not in content, f"key fragment leaked into {name}"
    results = json.loads(outputs[".results.json"])
    assert all(r["status"] == status for r in results)
    assert all("**REDACTED**" in r["error"] for r in results)


def test_auth_error_key_straddling_excerpt_cut_is_redacted(monkeypatch, capsys, tmp_path):
    """The key is redacted before the 500-char error excerpt is cut, so no key prefix survives."""
    mod = _load_spike_module()

    fake_key = "sk-straddling-secret-key-abcdefghijklmnop"
    monkeypatch.setenv("LLM_BASE_URL", "http://proxy:8000")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", fake_key)

    # json.dumps({"message": "<pad><key>"}) starts with '{"message": "' (13 chars);
    # put the key across the 500-char cut.
    pad = "x" * (500 - 13 - 10)

    async def fake_post(url, headers, raw_body, timeout_s=600.0):
        error = {"error": {"message": pad + fake_key}}
        return (401, {}, error, json.dumps(error))

    monkeypatch.setattr(mod, "_send_request", fake_post)

    out_dir = tmp_path / "spike-out"
    assert mod.main(["probe", "--out", str(out_dir)]) == 0

    outputs = _all_outputs(out_dir, capsys.readouterr())
    key_prefix = fake_key[:10]
    for name in ("table.md", ".results.json"):
        assert key_prefix not in outputs[name], f"key prefix leaked into {name}"


def test_base_url_userinfo_never_printed_or_stored(monkeypatch, capsys, tmp_path):
    """Credentials embedded in LLM_BASE_URL never reach stdout, and the password is never stored."""
    mod = _load_spike_module()

    user = "spikeuser"
    password = "pw-secret-in-url-987"
    monkeypatch.setenv("LLM_BASE_URL", f"https://{user}:{password}@proxy.example:8443/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "")

    async def fake_post(url, headers, raw_body, timeout_s=600.0):
        # A proxy that echoes the requested URL back in its error body
        error = {"error": {"message": f"upstream refused {url}"}}
        return (401, {}, error, json.dumps(error))

    monkeypatch.setattr(mod, "_send_request", fake_post)

    out_dir = tmp_path / "spike-out"
    assert mod.main(["probe", "--out", str(out_dir)]) == 0

    captured = capsys.readouterr()
    host_lines = [line for line in captured.out.splitlines() if line.startswith("Host: ")]
    assert host_lines == ["Host: proxy.example:8443"]
    assert user not in captured.out and password not in captured.out
    assert user not in captured.err and password not in captured.err

    outputs = _all_outputs(out_dir, captured)
    for name, content in outputs.items():
        assert password not in content, f"URL password leaked into {name}"


def _forbid_network(mod, monkeypatch) -> list[str]:
    """Replace the sender and aiohttp.ClientSession so any network attempt is recorded and fails."""
    calls: list[str] = []

    async def no_send(url, headers, raw_body, timeout_s=600.0):
        calls.append(url)
        raise AssertionError("preview-body must not send any request")

    def no_session(*args, **kwargs):
        calls.append("ClientSession")
        raise AssertionError("preview-body must not open an HTTP session")

    monkeypatch.setattr(mod, "_send_request", no_send)
    monkeypatch.setattr(mod.aiohttp, "ClientSession", no_session)
    return calls


@pytest.mark.slow
@pytest.mark.parametrize(
    ("strategy", "marker"),
    [("json_schema", "response_format"), ("tool_call", "tools")],
)
def test_preview_body_offline(monkeypatch, capsys, tmp_path, strategy, marker):
    """preview-body builds the request body offline from a synthetic buffer, LLM_MODEL only (A3-offline)."""
    mod = _load_spike_module()

    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "test-model")
    calls = _forbid_network(mod, monkeypatch)

    out_dir = tmp_path / "out"
    ret = mod.main(
        ["preview-body", "--days", "3", "--strategy", strategy, "--out", str(out_dir), "--seed", "3"]
    )
    assert ret == 0
    assert calls == []

    body_file = out_dir / "preview_body.json"
    assert body_file.exists()
    body = json.loads(body_file.read_text(encoding="utf-8"))

    # Same shape as the HA preview action's "body": model, system + user messages, strategy, profile
    assert body["model"] == "test-model"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    user = body["messages"][1]["content"]
    assert "<data>" in user and "</data>" in user
    assert marker in body
    if strategy == "json_schema":
        assert body["response_format"]["type"] == "json_schema"
    else:
        assert body["tools"][0]["function"]["name"] == "submit"
    assert "temperature" not in body
    assert body["max_completion_tokens"] == mod.preview_max_tokens()

    out = capsys.readouterr()
    assert str(body_file) in out.out
    assert "bytes" in out.out
    assert "clusters" in out.out and "plugs" in out.out
    assert "extract" in out.out and "MB" in out.out
    # On Linux the figure is VmHWM, so it carries no fallback marker (IN-11).
    assert "fallback" not in out.out and "peak unavailable" not in out.out
    assert "Host:" not in out.out


def test_preview_body_missing_model_exits_2(monkeypatch, capsys, tmp_path):
    """preview-body needs only LLM_MODEL; without it the script exits 2 naming that variable."""
    mod = _load_spike_module()

    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    calls = _forbid_network(mod, monkeypatch)

    out_dir = tmp_path / "out"
    with pytest.raises(SystemExit) as exc_info:
        mod.main(["preview-body", "--out", str(out_dir)])
    assert exc_info.value.code == 2

    captured = capsys.readouterr()
    assert "LLM_MODEL" in captured.err
    assert "LLM_BASE_URL" not in captured.err
    assert calls == []
    assert not (out_dir / "preview_body.json").exists()
