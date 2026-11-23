#!/usr/bin/env python3
"""Dev spike script for LiteLLM proxy (probe, payload, truncation, preview-body).

Loads engine/ under alias 'powerguesser_engine' (source: custom_components/powerguesser/engine/worker_entry.py)
and runs standalone requests without requiring Home Assistant.

preview-body (03-09 deviation A3-offline) needs no endpoint and makes no network call: it
builds a synthetic buffer with tests/fixtures/learning/synth.py, runs the extract stage through
the integration's worker_entry.py, and writes the request body prompts.build_request produces.
"""

from collections.abc import Mapping, Sequence
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import unquote, urlsplit

import aiohttp

# Load engine package under alias 'powerguesser_engine'
# Source: custom_components/powerguesser/engine/worker_entry.py
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENGINE = _REPO_ROOT / "custom_components" / "powerguesser" / "engine"
_spec = importlib.util.spec_from_file_location(
    "powerguesser_engine",
    _ENGINE / "__init__.py",
    submodule_search_locations=[str(_ENGINE)],
)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load engine spec from {_ENGINE}")

_mod = importlib.util.module_from_spec(_spec)
sys.modules["powerguesser_engine"] = _mod
_spec.loader.exec_module(_mod)

from powerguesser_engine.learning.budget import estimate_tokens  # noqa: E402
from powerguesser_engine.learning.prompts import PROPOSAL_SCHEMA, SCHEMA_NAME  # noqa: E402
from powerguesser_engine.llm_wire import (  # noqa: E402
    LADDER,
    PROBE_EXPECTED,
    PROBE_SCHEMA,
    Strategy,
    apply_profile,
    apply_strategy,
    build_probe_body,
    extract_json,
    model_matches,
)

KEPT_HEADERS = ("x-litellm-response-cost", "x-litellm-model-id")


def redact(obj: Any, secret: str | None) -> Any:
    """Return a deep copy where every string containing secret has it replaced by **REDACTED**."""
    if not secret:
        return copy.deepcopy(obj)

    if isinstance(obj, str):
        return obj.replace(secret, "**REDACTED**")
    if isinstance(obj, dict):
        return {k: redact(v, secret) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(item, secret) for item in obj]
    if isinstance(obj, tuple):
        return tuple(redact(item, secret) for item in obj)
    return obj


def redact_all(obj: Any, secrets: Sequence[str]) -> Any:
    """Return a deep copy with every non-empty secret replaced (longest first, so overlaps redact fully)."""
    result = copy.deepcopy(obj)
    for secret in sorted({s for s in secrets if s}, key=len, reverse=True):
        result = redact(result, secret)
    return result


def url_secrets(base_url: str) -> list[str]:
    """Return the userinfo password embedded in base_url (raw and percent-decoded), if any."""
    try:
        password = urlsplit(base_url).password
    except ValueError:
        return []
    if not password:
        return []
    return sorted({password, unquote(password)})


def spike_secrets(api_key: str | None, base_url: str) -> list[str]:
    """Every secret that must never be stored or printed: API key and base-URL password (T-03-35)."""
    return [s for s in [api_key or "", *url_secrets(base_url)] if s]


def display_host(base_url: str) -> str:
    """Return hostname[:port] of base_url, never its userinfo part (T-03-35)."""
    try:
        parts = urlsplit(base_url)
        hostname = parts.hostname
        port = parts.port
    except ValueError:
        return "<unparseable LLM_BASE_URL>"
    if not hostname:
        return "<unparseable LLM_BASE_URL>"
    if ":" in hostname:
        hostname = f"[{hostname}]"
    return f"{hostname}:{port}" if port is not None else hostname


def keep_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Keep only x-litellm-response-cost and x-litellm-model-id (case-insensitive)."""
    result = {}
    for k, v in headers.items():
        k_lower = k.lower()
        if k_lower in KEPT_HEADERS:
            result[k_lower] = str(v)
    return result


def render_table(results: Sequence[Mapping[str, Any]]) -> str:
    """Render markdown summary table from request results."""
    headers = [
        "rung",
        "variant",
        "status",
        "finish_reason",
        "json_location",
        "valid",
        "prompt_tokens",
        "completion_tokens",
        "bytes",
        "est_tokens",
        "requested_model",
        "model",
        "model_mismatch",
        "cost_header",
        "error",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in results:
        row = [
            str(r.get("rung", "") or ""),
            str(r.get("variant", "") or ""),
            str(r.get("status", "") if r.get("status") is not None else ""),
            str(r.get("finish_reason", "") or ""),
            str(r.get("json_location", "") or ""),
            str(r.get("valid", "") if r.get("valid") is not None else ""),
            str(r.get("prompt_tokens", "") if r.get("prompt_tokens") is not None else ""),
            str(r.get("completion_tokens", "") if r.get("completion_tokens") is not None else ""),
            str(r.get("bytes", "") if r.get("bytes") is not None else ""),
            str(r.get("est_tokens", "") if r.get("est_tokens") is not None else ""),
            str(r.get("requested_model", "") or ""),
            str(r.get("model", "") or ""),
            str(r.get("model_mismatch", "") if r.get("model_mismatch") is not None else ""),
            str(r.get("cost_header", "") or ""),
            str(r.get("error", "") or "").replace("\n", " "),
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


async def _send_request(
    url: str,
    headers: dict[str, str],
    raw_body: bytes,
    timeout_s: float = 600.0,
) -> tuple[int, dict[str, str], Any, str]:
    """Send single HTTP POST request and return (status, headers, parsed_json_or_none, raw_text)."""
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            url,
            headers=headers,
            data=raw_body,
            allow_redirects=False,
        ) as resp:
            resp_headers = dict(resp.headers)
            raw_text = await resp.text()
            parsed_json = None
            try:
                parsed_json = json.loads(raw_text)
            except Exception:
                pass
            return resp.status, resp_headers, parsed_json, raw_text


async def _execute_single(
    name: str,
    rung: str,
    variant: str,
    body: dict[str, Any],
    base_url: str,
    api_key: str | None,
    is_probe: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute request, parse response, and return (result_dict, cassette_dict)."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    raw_body = json.dumps(body).encode("utf-8")
    body_bytes = len(raw_body)
    est_in = estimate_tokens(body_bytes)
    body_sha256 = hashlib.sha256(raw_body).hexdigest()

    secrets = spike_secrets(api_key, base_url)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    status, resp_headers, resp_json, raw_text = await _send_request(
        url=url,
        headers=headers,
        raw_body=raw_body,
        timeout_s=600.0,
    )

    filtered_headers = keep_headers(resp_headers)
    cost_header = filtered_headers.get("x-litellm-response-cost", "")

    finish_reason = None
    json_loc = "none"
    valid = False
    prompt_tokens = None
    completion_tokens = None
    model_name = None
    error_excerpt = ""

    if isinstance(resp_json, dict):
        usage = resp_json.get("usage")
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
        model_name = resp_json.get("model")

        choices = resp_json.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            first = choices[0]
            if isinstance(first, dict):
                finish_reason = first.get("finish_reason")
                msg = first.get("message", {})
                extracted, loc = extract_json(msg)
                json_loc = loc
                if is_probe:
                    valid = (extracted == PROBE_EXPECTED)
                else:
                    valid = (extracted is not None)

        if status >= 400:
            error_obj = resp_json.get("error", resp_json)
            # Redact the parsed object before serialising: json.dumps escapes `"` and
            # non-ASCII, so a key in its output no longer matches the plain-text search.
            # Then redact the text before the cut, so a key straddling char 500 leaves no
            # prefix (T-03-35).
            error_excerpt = redact_all(
                json.dumps(redact_all(error_obj, secrets)), secrets
            )[:500]
    else:
        if status >= 400 or not raw_text:
            error_excerpt = redact_all(raw_text, secrets)[:500]

    # A router can silently answer with another model (03-09: kimi-for-coding ->
    # xiaomi/mimo-v2.6-flash on temperature 0); unknown when no model came back.
    requested_model = body.get("model")
    model_mismatch: bool | None = None
    if requested_model and isinstance(model_name, str) and model_name.strip():
        model_mismatch = not model_matches(str(requested_model), model_name)

    result = {
        "name": name,
        "rung": rung,
        "variant": variant,
        "status": status,
        "finish_reason": finish_reason,
        "json_location": json_loc,
        "valid": valid,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "bytes": body_bytes,
        "est_tokens": est_in,
        "requested_model": requested_model,
        "model": model_name,
        "model_mismatch": model_mismatch,
        "cost_header": cost_header,
        "error": error_excerpt,
    }

    cassette = {
        "request_sha256": body_sha256,
        "status": status,
        "headers": filtered_headers,
    }
    if resp_json is not None:
        cassette["json"] = resp_json
    else:
        cassette["text"] = raw_text

    return result, cassette


# Default parameter profile the integration's probe starts from (llm_client.async_probe):
# no temperature (03-09 A2), max_completion_tokens.
DEFAULT_PROFILE: dict[str, Any] = {
    "token_param": "max_completion_tokens",
    "send_temperature": False,
}

_WORKER_ENTRY = _ENGINE / "worker_entry.py"
_SYNTH_PATH = _REPO_ROOT / "tests" / "fixtures" / "learning" / "synth.py"


def preview_max_tokens() -> int:
    """Return max_tokens as async_preview_payload sends it (const.DEFAULT_MAX_TOKENS)."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from custom_components.powerguesser.const import DEFAULT_MAX_TOKENS

    return int(DEFAULT_MAX_TOKENS)


def _load_synth() -> Any:
    """Load the synthetic-buffer fixture (tests/ is dev-only, like this script)."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location("powerguesser_spike_synth", _SYNTH_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load synth fixture from {_SYNTH_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["powerguesser_spike_synth"] = mod
    spec.loader.exec_module(mod)
    return mod


def _run_extract_worker(job_path: Path) -> dict[str, Any]:
    """Run the extract stage through worker_entry.py as learning._async_run_stage does.

    Same stage invocation (sys.executable -P worker_entry.py extract --job <job>), without
    ionice and the stdin watchdog. The worker reports its own peak memory (VmHWM from
    /proc/self/status, Plan 03-10), so spawning it directly from this process, which has
    Home Assistant and numpy imported, no longer inflates the printed figure.
    """
    import subprocess

    argv = [
        sys.executable,
        "-P",
        str(_WORKER_ENTRY),
        "extract",
        "--job",
        str(job_path),
    ]
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, check=False)
    done: dict[str, Any] | None = None
    error: str | None = None
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") == "done":
            done = event
        elif event.get("event") == "error":
            error = str(event.get("error"))
    if error is not None or proc.returncode != 0 or done is None:
        raise RuntimeError(
            f"extract failed (exit {proc.returncode}): {error or proc.stderr[-500:]}"
        )
    return done


def _preview_body(model: str, days: int, strategy: Strategy, seed: int, out_dir: Path) -> int:
    """Build preview_body.json offline from a synthetic buffer through the extract worker (A3-offline)."""
    import tempfile
    import time
    from datetime import UTC, datetime

    import numpy as np

    from powerguesser_engine.buffer import RETENTION_DAYS
    from powerguesser_engine.intervals import gate
    from powerguesser_engine.learning.prompts import build_request, payload_bytes
    from powerguesser_engine.learning.worker import peak_memory_text

    synth = _load_synth()
    spec = synth.SynthSpec(days=days, seed=seed, oven=True, washer=True, phases=True)
    ms_per_day = 86_400_000
    end_ms = int(time.time() * 1000) // ms_per_day * ms_per_day  # today's UTC midnight
    # Same window as learning._build_extract_job_data: RETENTION_DAYS day bounds,
    # days without data are dropped by the worker's data_days filter.
    day_bounds_ms = synth.utc_day_bounds(end_ms, RETENTION_DAYS)
    first_weekday = datetime.fromtimestamp(day_bounds_ms[0] / 1000, tz=UTC).weekday()

    with tempfile.TemporaryDirectory(prefix="powerguesser-preview-") as tmp:
        tmp_path = Path(tmp)
        data = synth.build_buffer(tmp_path / "buffer.sqlite", spec, end_ms)

        # Effective interval as the profile coordinator derives it: median gross interval.
        grid_ts = np.array([t for t, _ in data.rows[synth.ENTITY_GRID]], dtype=np.int64)
        effective_s = float(np.median(np.diff(grid_ts))) / 1000.0
        grid_s = int(round(effective_s))
        enabled_types = [t.value for t in gate(effective_s, False)[0]]

        metered_channels = [data.channel_ids[synth.ENTITY_WASHER]] if spec.washer else []
        phase_channels: dict[str, int] = {}
        if spec.phases:
            phase_channels = {
                "l1": data.channel_ids[synth.ENTITY_L1],
                "l2": data.channel_ids[synth.ENTITY_L2],
                "l3": data.channel_ids[synth.ENTITY_L3],
            }

        stage_dir = tmp_path / "prepare"
        job_extract = {
            "buffer_path": str(tmp_path / "buffer.sqlite"),
            "stage_dir": str(stage_dir),
            "grid_s": grid_s,
            "day_bounds_ms": day_bounds_ms,
            "first_weekday": first_weekday,
            "holdout_days": 0,
            "train_fraction": 0.7,
            "enabled_event_types": enabled_types,
            "metered_channels": metered_channels,
            "phase_channels": phase_channels,
        }
        stage_dir.mkdir(parents=True, exist_ok=True)
        job_path = stage_dir / "job_extract.json"
        job_path.write_text(json.dumps(job_extract, indent=2), encoding="utf-8")

        done = _run_extract_worker(job_path)
        digest = json.loads((stage_dir / "digest.json").read_text(encoding="utf-8"))

    profile = {**DEFAULT_PROFILE, "strategy": strategy.value}
    body = build_request(
        digest,
        model=model,
        profile=profile,
        max_tokens=preview_max_tokens(),
        names=None,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    body_path = out_dir / "preview_body.json"
    body_path.write_text(json.dumps(body, indent=2), encoding="utf-8")

    n_bytes = payload_bytes(body)
    stats = done.get("stats", {})
    wall_s = float(stats.get("wall_s", 0.0))
    n_clusters = len(digest.get("clusters", []))
    n_plugs = len(digest.get("plugs", []))
    sys.stdout.write(
        f"Wrote {body_path}: {n_bytes} bytes, ~{estimate_tokens(n_bytes)} est. tokens; "
        f"digest {n_clusters} clusters, {n_plugs} plugs; "
        f"extract {wall_s:.1f} s, peak {peak_memory_text(stats)} "
        f"(dev machine, synthetic {days} days)\n"
    )
    return 0


def _body_has_strategy(body: dict[str, Any]) -> bool:
    """Check if request body already has a structured output strategy applied."""
    if "response_format" in body or "tools" in body:
        return True
    messages = body.get("messages", [])
    for msg in messages:
        content = msg.get("content", "")
        if "Output must be valid JSON" in content or "fenced ```json" in content:
            return True
    return False


async def async_run(argv: list[str]) -> int:
    """Async main routine."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Dev spike script for LiteLLM proxy")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # probe
    sub_probe = subparsers.add_parser("probe", help="Run capability probe rungs")
    sub_probe.add_argument("--out", default="spike-out", help="Output directory")

    # payload
    sub_payload = subparsers.add_parser("payload", help="Run payload review request")
    sub_payload.add_argument("--body", required=True, help="Path to preview body JSON file")
    sub_payload.add_argument("--strategy", default=None, help="Strategy to apply if body has none")
    sub_payload.add_argument("--out", default="spike-out", help="Output directory")

    # truncation
    sub_trunc = subparsers.add_parser("truncation", help="Run truncation check request")
    sub_trunc.add_argument("--body", required=True, help="Path to preview body JSON file")
    sub_trunc.add_argument("--repeat", type=int, default=3, help="Repeat multiplier for data block")
    sub_trunc.add_argument("--out", default="spike-out", help="Output directory")

    # preview-body (03-09 A3-offline): offline, LLM_MODEL only, no network
    sub_prev = subparsers.add_parser(
        "preview-body",
        help="Build preview_body.json offline from a synthetic buffer (no network)",
    )
    sub_prev.add_argument("--days", type=int, default=21, help="Synthetic history in days")
    sub_prev.add_argument(
        "--strategy",
        default=Strategy.JSON_SCHEMA.value,
        choices=[s.value for s in LADDER],
        help="Structured-output strategy (the probe-selected one)",
    )
    sub_prev.add_argument("--seed", type=int, default=7, help="Synthetic data seed")
    sub_prev.add_argument("--out", default="spike-out", help="Output directory")

    args = parser.parse_args(argv)

    if args.subcommand == "preview-body":
        model = os.environ.get("LLM_MODEL")
        if not model:
            sys.stderr.write("ERROR: LLM_MODEL environment variable is required\n")
            return 2
        if args.days < 1:
            sys.stderr.write("ERROR: --days must be at least 1\n")
            return 2
        # BufferDB refuses to run on an event-loop thread; build off the loop.
        return await asyncio.to_thread(
            _preview_body,
            model=model,
            days=args.days,
            strategy=Strategy(args.strategy),
            seed=args.seed,
            out_dir=Path(args.out),
        )

    base_url = os.environ.get("LLM_BASE_URL")
    if not base_url:
        sys.stderr.write("ERROR: LLM_BASE_URL environment variable is required\n")
        return 2

    model = os.environ.get("LLM_MODEL")
    if not model:
        sys.stderr.write("ERROR: LLM_MODEL environment variable is required\n")
        return 2

    api_key = os.environ.get("LLM_API_KEY", "")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cassettes_path = out_dir / "cassettes.json"
    table_path = out_dir / "table.md"

    # Load existing cassettes and results if present for accumulation
    existing_cassettes: dict[str, Any] = {}
    if cassettes_path.exists():
        try:
            existing_cassettes = json.loads(cassettes_path.read_text(encoding="utf-8"))
        except Exception:
            existing_cassettes = {}

    existing_results: list[dict[str, Any]] = []
    # If table exists, we can keep track of results in an internal sidecar or just rebuild
    results_path = out_dir / ".results.json"
    if results_path.exists():
        try:
            existing_results = json.loads(results_path.read_text(encoding="utf-8"))
        except Exception:
            existing_results = []

    new_results: list[dict[str, Any]] = []

    if args.subcommand == "probe":
        # 4 rungs of ladder with default profile
        for strat in LADDER:
            body = build_probe_body(model=model, strategy=strat, profile=None)
            res, cas = await _execute_single(
                name=strat.value,
                rung=strat.value,
                variant="default",
                body=body,
                base_url=base_url,
                api_key=api_key,
                is_probe=True,
            )
            new_results.append(res)
            existing_cassettes[strat.value] = cas

        # Variant 1: json_schema with max_tokens token_param
        body_max_tok = build_probe_body(
            model=model,
            strategy=Strategy.JSON_SCHEMA,
            profile={"token_param": "max_tokens"},
        )
        res_mt, cas_mt = await _execute_single(
            name="json_schema_max_tokens",
            rung="json_schema",
            variant="max_tokens",
            body=body_max_tok,
            base_url=base_url,
            api_key=api_key,
            is_probe=True,
        )
        new_results.append(res_mt)
        existing_cassettes["json_schema_max_tokens"] = cas_mt

        # Variants 2 and 3: json_schema sending temperature 0 and 1. The default
        # profile sends no temperature (03-09 A2), so these capture what the proxy
        # does with an explicit temperature (router fallback to another model).
        for temp in (0, 1):
            name = f"json_schema_temperature_{temp}"
            body_temp = build_probe_body(
                model=model,
                strategy=Strategy.JSON_SCHEMA,
                profile={"send_temperature": True, "temperature": temp},
            )
            res_t, cas_t = await _execute_single(
                name=name,
                rung="json_schema",
                variant=f"temperature_{temp}",
                body=body_temp,
                base_url=base_url,
                api_key=api_key,
                is_probe=True,
            )
            new_results.append(res_t)
            existing_cassettes[name] = cas_t

    elif args.subcommand == "payload":
        body_path = Path(args.body)
        raw_content = body_path.read_text(encoding="utf-8")
        body = json.loads(raw_content)

        if args.strategy and not _body_has_strategy(body):
            strat = Strategy(args.strategy)
            body = apply_strategy(
                body,
                schema=PROPOSAL_SCHEMA,
                name=SCHEMA_NAME,
                strategy=strat,
            )

        res, cas = await _execute_single(
            name="payload",
            rung="payload",
            variant=args.strategy or "as_is",
            body=body,
            base_url=base_url,
            api_key=api_key,
            is_probe=False,
        )
        new_results.append(res)
        existing_cassettes["payload"] = cas

    elif args.subcommand == "truncation":
        body_path = Path(args.body)
        raw_content = body_path.read_text(encoding="utf-8")
        body = json.loads(raw_content)

        repeat = max(1, args.repeat)
        # Repeat the <data> block content N times inside user message
        messages = body.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                m = re.search(r"<data>([\s\S]*?)</data>", content)
                if m:
                    inner = m.group(1).strip()
                    repeated = "\n".join([inner] * repeat)
                    msg["content"] = content[: m.start()] + f"<data>\n{repeated}\n</data>" + content[m.end() :]

        res, cas = await _execute_single(
            name="truncation",
            rung="truncation",
            variant=f"repeat_{repeat}",
            body=body,
            base_url=base_url,
            api_key=api_key,
            is_probe=False,
        )
        new_results.append(res)
        existing_cassettes["truncation"] = cas

    # Merge results preserving order by name
    updated_results_map = {r["name"]: r for r in existing_results}
    for r in new_results:
        updated_results_map[r["name"]] = r

    # Desired order: the 4 rungs, the 3 json_schema variants, payload, truncation
    order = [
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
    all_results = []
    for k in order:
        if k in updated_results_map:
            all_results.append(updated_results_map[k])
    for k, v in updated_results_map.items():
        if k not in order:
            all_results.append(v)

    # Redact every stored structure once (API key + base-URL password, T-03-35); only
    # the redacted copies are written: cassettes, table.md and the .results.json sidecar.
    secrets = spike_secrets(api_key, base_url)
    redacted_cassettes = redact_all(existing_cassettes, secrets)
    redacted_results = redact_all(all_results, secrets)

    cassettes_path.write_text(json.dumps(redacted_cassettes, indent=2), encoding="utf-8")
    table_path.write_text(render_table(redacted_results), encoding="utf-8")
    results_path.write_text(json.dumps(redacted_results, indent=2), encoding="utf-8")

    # Print host (never userinfo) and table path only
    sys.stdout.write(f"Host: {display_host(base_url)}\n")
    sys.stdout.write(f"Table: {table_path}\n")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for CLI execution."""
    import asyncio

    if argv is None:
        argv = sys.argv[1:]
    rc = asyncio.run(async_run(argv))
    if rc != 0:
        sys.exit(rc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
