"""Token estimation and budget calculations for LLM learning."""

from dataclasses import dataclass
from datetime import datetime
import math

BYTES_PER_TOKEN = 3.0

CAP_ORDER = ("run_input", "run_output", "month_input", "month_output", "month_cost")

TRUNC_FACTOR = 0.8
TRUNC_DIGEST_TOKENS_PER_BYTE = 0.4
TRUNC_OTHER_BYTES_PER_TOKEN = 4.5
RATIO_ALPHA = 0.3


@dataclass(frozen=True)
class Caps:
    """Configured token and cost limits."""

    run_input: int = 300_000
    run_output: int = 128_000
    month_input: int = 2_000_000
    month_output: int = 600_000
    month_cost: float | None = None
    price_input_per_1m: float | None = None
    price_output_per_1m: float | None = None


@dataclass(frozen=True)
class Usage:
    """Current token and cost counters for a run and month."""

    run_input: int = 0
    run_output: int = 0
    month_input: int = 0
    month_output: int = 0
    month_cost: float | None = None


def estimate_tokens(n_bytes: int, calibrated_ratio: float | None = None) -> int:
    """Estimate input token count from payload byte size.

    Uses a conservative heuristic (BYTES_PER_TOKEN = 3.0) for JSON with numeric digits,
    or the higher of heuristic and calibrated ratio if given.
    """
    base = math.ceil(n_bytes / BYTES_PER_TOKEN)
    if calibrated_ratio is not None:
        calibrated = math.ceil(n_bytes * calibrated_ratio)
        return max(base, calibrated)
    return base


def output_estimate(content_bytes: int) -> int:
    """Estimate output tokens from response content bytes when usage is missing."""
    return math.ceil(content_bytes / BYTES_PER_TOKEN)


def month_key(dt: datetime) -> str:
    """Format local datetime as YYYY-MM month key."""
    return f"{dt.year:04d}-{dt.month:02d}"


def request_cost(input_tokens: int, output_tokens: int, caps: Caps) -> float | None:
    """Calculate request cost in currency units if both prices are set."""
    if caps.price_input_per_1m is None or caps.price_output_per_1m is None:
        return None
    return (input_tokens / 1_000_000.0) * caps.price_input_per_1m + (
        output_tokens / 1_000_000.0
    ) * caps.price_output_per_1m


def check_request(
    est_in: int, max_tokens: int, usage: Usage, caps: Caps
) -> str | None:
    """Check whether a request would exceed any token cap or the cost limit.

    Returns the first breached cap name from CAP_ORDER, or None if within limits.
    Equality to cap is allowed; strictly greater than cap is refused.
    """
    if usage.run_input + est_in > caps.run_input:
        return "run_input"
    if usage.run_output + max_tokens > caps.run_output:
        return "run_output"
    if usage.month_input + est_in > caps.month_input:
        return "month_input"
    if usage.month_output + max_tokens > caps.month_output:
        return "month_output"
    if (
        caps.month_cost is not None
        and caps.price_input_per_1m is not None
        and caps.price_output_per_1m is not None
    ):
        req_c = request_cost(est_in, max_tokens, caps)
        if req_c is not None:
            current_cost = usage.month_cost or 0.0
            if current_cost + req_c > caps.month_cost:
                return "month_cost"
    return None


def truncation_suspected(
    prompt_tokens: int,
    n_bytes: int,
    digest_bytes: int = 0,
) -> bool:
    """Check if the endpoint appears to have silently truncated the prompt.

    The floor follows the request's own composition: the numeric digest tokenizes far
    denser than the prefix, schema and pretty-printed JSON around it, so one learned
    tokens/byte ratio cannot serve every request kind (refine, repair). digest_bytes is
    the escaped digest's share of n_bytes (0 for repair turns).
    """
    if n_bytes <= 0:
        return False
    other_bytes = n_bytes - digest_bytes
    expected_min = TRUNC_FACTOR * (
        digest_bytes * TRUNC_DIGEST_TOKENS_PER_BYTE
        + other_bytes / TRUNC_OTHER_BYTES_PER_TOKEN
    )
    return prompt_tokens < expected_min


def update_ratio(
    ratio: float | None,
    n_obs: int,
    prompt_tokens: int,
    n_bytes: int,
) -> tuple[float, int]:
    """Update the EWMA prompt token ratio (prompt_tokens / bytes)."""
    if n_bytes <= 0:
        return (ratio if ratio is not None else 0.0, n_obs)
    observed = prompt_tokens / n_bytes
    if ratio is None or n_obs == 0:
        return (observed, 1)
    new_ratio = (1.0 - RATIO_ALPHA) * ratio + RATIO_ALPHA * observed
    return (new_ratio, n_obs + 1)
