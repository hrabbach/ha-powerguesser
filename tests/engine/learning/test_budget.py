"""Unit tests for budget and token estimation."""

from datetime import datetime
import math

from custom_components.powerguesser.engine.learning.budget import (
    BYTES_PER_TOKEN,
    CAP_ORDER,
    Caps,
    Usage,
    check_request,
    estimate_tokens,
    month_key,
    output_estimate,
    request_cost,
    truncation_suspected,
    update_ratio,
)


def test_bytes_per_token_constant() -> None:
    """Verify default BYTES_PER_TOKEN is conservative 3.0."""
    assert BYTES_PER_TOKEN == 3.0


def test_estimate_tokens_zero_bytes() -> None:
    """Zero bytes should estimate to 0 tokens."""
    assert estimate_tokens(0) == 0


def test_estimate_tokens_heuristic() -> None:
    """Test heuristic ceil(n_bytes / 3.0)."""
    assert estimate_tokens(1) == 1
    assert estimate_tokens(2) == 1
    assert estimate_tokens(3) == 1
    assert estimate_tokens(4) == 2
    assert estimate_tokens(5) == 2
    assert estimate_tokens(6) == 2
    assert estimate_tokens(7) == 3


def test_estimate_tokens_calibrated_ratio() -> None:
    """Test calibrated ratio comparison against base heuristic."""
    # When ratio produces larger value than heuristic, ratio wins
    # 10 bytes: heuristic = ceil(10 / 3) = 4. ratio 1.0 -> 10.
    assert estimate_tokens(10, calibrated_ratio=1.0) == 10

    # When ratio produces smaller value than heuristic, heuristic wins (conservative)
    # 10 bytes: heuristic = 4. ratio 0.1 -> ceil(1.0) = 1.
    assert estimate_tokens(10, calibrated_ratio=0.1) == 4


def test_output_estimate() -> None:
    """Verify output estimation uses ceil(content_bytes / 3)."""
    assert output_estimate(0) == 0
    assert output_estimate(1) == 1
    assert output_estimate(3) == 1
    assert output_estimate(4) == 2
    assert output_estimate(100) == 34


def test_month_key() -> None:
    """Verify month_key formats datetime as YYYY-MM."""
    dt1 = datetime(2026, 3, 5, 14, 30)
    assert month_key(dt1) == "2026-03"
    dt2 = datetime(2026, 12, 31, 23, 59)
    assert month_key(dt2) == "2026-12"


def test_request_cost() -> None:
    """Verify cost calculation in float as tokens / 1e6 x price."""
    # Without both prices set, returns None
    caps_no_price = Caps()
    assert request_cost(1000, 1000, caps_no_price) is None

    caps_one_price = Caps(price_input_per_1m=2.5)
    assert request_cost(1000, 1000, caps_one_price) is None

    caps_both = Caps(price_input_per_1m=2.5, price_output_per_1m=10.0)
    # 1_000_000 in ($2.50) + 500_000 out ($5.00) = $7.50
    cost = request_cost(1_000_000, 500_000, caps_both)
    assert cost is not None
    assert abs(cost - 7.50) < 1e-6


def test_check_request_each_cap_and_boundaries() -> None:
    """Test each cap boundary: equality allowed, +1 refused."""
    caps = Caps(
        run_input=1000,
        run_output=500,
        month_input=5000,
        month_output=2000,
        month_cost=10.0,
        price_input_per_1m=2.0,
        price_output_per_1m=4.0,
    )

    # 1. run_input
    # Exactly equal -> allowed (None)
    usage = Usage(run_input=800)
    assert check_request(est_in=200, max_tokens=100, usage=usage, caps=caps) is None
    # 1 token more -> refused
    assert check_request(est_in=201, max_tokens=100, usage=usage, caps=caps) == "run_input"

    # 2. run_output
    usage = Usage(run_output=400)
    assert check_request(est_in=100, max_tokens=100, usage=usage, caps=caps) is None
    assert check_request(est_in=100, max_tokens=101, usage=usage, caps=caps) == "run_output"

    # 3. month_input
    usage = Usage(month_input=4800)
    assert check_request(est_in=200, max_tokens=100, usage=usage, caps=caps) is None
    assert check_request(est_in=201, max_tokens=100, usage=usage, caps=caps) == "month_input"

    # 4. month_output
    usage = Usage(month_output=1900)
    assert check_request(est_in=100, max_tokens=100, usage=usage, caps=caps) is None
    assert check_request(est_in=100, max_tokens=101, usage=usage, caps=caps) == "month_output"

    # 5. month_cost
    # cost of 100 in + 100 max_out = (100/1e6)*2 + (100/1e6)*4 = 0.0006
    usage = Usage(month_cost=9.9994)
    # Exactly reaches 10.0 -> allowed
    assert check_request(est_in=100, max_tokens=100, usage=usage, caps=caps) is None
    # Over 10.0 -> month_cost
    usage_over = Usage(month_cost=9.9995)
    assert check_request(est_in=100, max_tokens=100, usage=usage_over, caps=caps) == "month_cost"


def test_check_request_order_of_reasons() -> None:
    """When multiple caps would be exceeded, check_request returns first in CAP_ORDER."""
    assert CAP_ORDER == ("run_input", "run_output", "month_input", "month_output", "month_cost")

    caps = Caps(
        run_input=100,
        run_output=100,
        month_input=100,
        month_output=100,
        month_cost=0.00001,
        price_input_per_1m=1.0,
        price_output_per_1m=1.0,
    )

    # All breach
    assert check_request(est_in=200, max_tokens=200, usage=Usage(), caps=caps) == "run_input"

    # run_input fits, rest breach
    assert check_request(est_in=50, max_tokens=200, usage=Usage(month_input=80), caps=caps) == "run_output"

    # run_input and run_output fit, month_input breaches
    assert check_request(est_in=50, max_tokens=50, usage=Usage(month_input=80), caps=caps) == "month_input"

    # month_output breaches
    assert check_request(est_in=50, max_tokens=50, usage=Usage(month_output=80), caps=caps) == "month_output"

    # only month_cost breaches
    caps_cost_only = Caps(
        run_input=1000,
        run_output=1000,
        month_input=1000,
        month_output=1000,
        month_cost=0.00001,
        price_input_per_1m=1.0,
        price_output_per_1m=1.0,
    )
    assert check_request(est_in=50, max_tokens=50, usage=Usage(), caps=caps_cost_only) == "month_cost"


def test_check_request_cost_requires_both_prices_and_limit() -> None:
    """Cost limit is ignored if missing a price or limit is None."""
    # Missing prices -> no month_cost check
    caps1 = Caps(month_cost=0.0001)
    assert check_request(est_in=100, max_tokens=100, usage=Usage(), caps=caps1) is None

    # Missing month_cost limit -> no month_cost check
    caps2 = Caps(price_input_per_1m=1.0, price_output_per_1m=1.0)
    assert check_request(est_in=100, max_tokens=100, usage=Usage(), caps=caps2) is None


# Live request mix of the 2026-11-18 false positive (kimi-for-coding via the owner's
# LiteLLM proxy, debug session truncation-guard-false-positive): the round-1 body was
# 20571 bytes of which 16089 were the escaped digest; tokens estimated from the
# spike-measured densities. The refine body and its token count are the observed ones.
LIVE_DIGEST_B = 16_089
LIVE_R1_B, LIVE_R1_T = 20_571, 10_617
LIVE_REFINE_B, LIVE_REFINE_T = 40_119, 16_218
# Repair turn: fixed part 4482 B + ids/errors ~600 B + ~10 kB escaped invalid output,
# no digest; ~0.24 tok/B (lower end of the estimate).
LIVE_REPAIR_B, LIVE_REPAIR_T = 15_382, 3_733


def test_truncation_floor_is_content_aware() -> None:
    """The floor is 0.8 x (digest_bytes x 0.4 + other_bytes / 4.5), boundaries on both sides."""
    # No digest (repair turn): 0.8 x 30000 / 4.5 = 5333.3
    assert truncation_suspected(prompt_tokens=100, n_bytes=30_000) is True
    assert truncation_suspected(prompt_tokens=5333, n_bytes=30_000, digest_bytes=0) is True
    assert truncation_suspected(prompt_tokens=5334, n_bytes=30_000, digest_bytes=0) is False

    # Mixed: 0.8 x (10000 x 0.4 + 20000 / 4.5) = 6755.6
    assert truncation_suspected(prompt_tokens=6755, n_bytes=30_000, digest_bytes=10_000) is True
    assert truncation_suspected(prompt_tokens=6756, n_bytes=30_000, digest_bytes=10_000) is False

    # Exactly at the floor is not truncation: 0.8 x (1000 x 0.4 + 4500 / 4.5) = 1120
    assert truncation_suspected(prompt_tokens=1119, n_bytes=5_500, digest_bytes=1_000) is True
    assert truncation_suspected(prompt_tokens=1120, n_bytes=5_500, digest_bytes=1_000) is False

    # Digest bytes raise the floor: the same count is fine without digest, flagged with it
    assert truncation_suspected(prompt_tokens=6000, n_bytes=30_000, digest_bytes=0) is False
    assert truncation_suspected(prompt_tokens=6000, n_bytes=30_000, digest_bytes=10_000) is True

    # Zero bytes edge case
    assert truncation_suspected(prompt_tokens=0, n_bytes=0) is False
    assert truncation_suspected(prompt_tokens=0, n_bytes=0, digest_bytes=0) is False


def test_live_refine_after_round1_calibration_not_flagged() -> None:
    """Regression 2026-11-18: refine 16218 tok / 40119 B after round-1 calibration is not truncation.

    The stored EWMA ratio comes from digest-dense round-1 asks; a ratio-based floor
    (0.8 x bytes x ratio = 16565) flagged the refine request, whose extra ~19.6 kB are
    pretty-printed previous proposal + scorecard at ~0.29 tok/B.
    """
    ratio, n_obs = update_ratio(None, 0, LIVE_R1_T, LIVE_R1_B)
    ratio, n_obs = update_ratio(ratio, n_obs, LIVE_R1_T, LIVE_R1_B)
    assert n_obs == 2
    # The live data reproduce the old failure condition (oracle sanity check).
    assert LIVE_REFINE_T < 0.8 * LIVE_REFINE_B * ratio

    assert truncation_suspected(LIVE_R1_T, LIVE_R1_B, digest_bytes=LIVE_DIGEST_B) is False
    assert (
        truncation_suspected(LIVE_REFINE_T, LIVE_REFINE_B, digest_bytes=LIVE_DIGEST_B)
        is False
    )

    # Calibration still drives the cap estimate (not the guard).
    assert estimate_tokens(LIVE_REFINE_B, calibrated_ratio=ratio) == math.ceil(
        LIVE_REFINE_B * ratio
    )


def test_repair_turn_after_round1_calibration_not_flagged() -> None:
    """A repair turn (~0.24 tok/B, no digest) is not flagged after digest-dense calibration."""
    ratio, n_obs = update_ratio(None, 0, LIVE_R1_T, LIVE_R1_B)
    ratio, n_obs = update_ratio(ratio, n_obs, LIVE_R1_T, LIVE_R1_B)
    # A ratio-based floor would have flagged it (0.8 x 15382 x 0.516 = 6351).
    assert LIVE_REPAIR_T < 0.8 * LIVE_REPAIR_B * ratio

    assert truncation_suspected(LIVE_REPAIR_T, LIVE_REPAIR_B, digest_bytes=0) is False


def test_genuine_truncations_of_live_requests_are_flagged() -> None:
    """Context-limit cuts of the live propose and refine requests stay detectable."""
    # Propose floor: 0.8 x (16089 x 0.4 + 4482 / 4.5) = 5945.3
    for cut in (100, 2048, 4096, 5945):
        assert truncation_suspected(cut, LIVE_R1_B, digest_bytes=LIVE_DIGEST_B) is True, cut
    assert truncation_suspected(5946, LIVE_R1_B, digest_bytes=LIVE_DIGEST_B) is False

    # Refine floor: 0.8 x (16089 x 0.4 + 24030 / 4.5) = 9420.5
    for cut in (100, 4096, 8192, 9420):
        assert (
            truncation_suspected(cut, LIVE_REFINE_B, digest_bytes=LIVE_DIGEST_B) is True
        ), cut
    assert truncation_suspected(9421, LIVE_REFINE_B, digest_bytes=LIVE_DIGEST_B) is False


def test_update_ratio() -> None:
    """Test EWMA ratio update with alpha=0.3."""
    # First observation (n_obs = 0)
    # 2500 tokens / 10000 bytes = 0.25
    ratio1, n1 = update_ratio(None, 0, prompt_tokens=2500, n_bytes=10000)
    assert ratio1 == 0.25
    assert n1 == 1

    # Second observation:
    # 3500 tokens / 10000 bytes = 0.35
    # new_ratio = (1 - 0.3) * 0.25 + 0.3 * 0.35 = 0.7 * 0.25 + 0.3 * 0.35 = 0.175 + 0.105 = 0.28
    ratio2, n2 = update_ratio(ratio1, n1, prompt_tokens=3500, n_bytes=10000)
    assert abs(ratio2 - 0.28) < 1e-6
    assert n2 == 2
