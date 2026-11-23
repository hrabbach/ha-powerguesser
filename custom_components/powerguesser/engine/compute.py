"""Pure calculation functions for gross, metered, and remainder power."""

from collections.abc import Mapping, Sequence
from typing import Any

from .model import GrossChannel, Role, Snapshot, SourceModel, Transform


class OverAttribution:

    """Accumulator for over-attribution energy (Wh) when metered > gross."""

    def __init__(self) -> None:
        self._total_wh: float = 0.0
        self._last_signed: float | None = None
        self._last_ts: float | None = None

    def add(self, signed_w: float, ts: float) -> None:
        """Add a signed residual sample at timestamp ts."""
        if self._last_ts is not None:
            if ts <= self._last_ts:
                return
            if self._last_signed is not None:
                dt_hours = (ts - self._last_ts) / 3600.0
                clipped_w = max(0.0, -self._last_signed)
                self._total_wh += clipped_w * dt_hours

        self._last_signed = signed_w
        self._last_ts = ts

    @property
    def total_wh(self) -> float:
        """Total accumulated over-attribution in Wh."""
        return self._total_wh


def channel_value_w(
    channel: GrossChannel,
    values: Mapping[str, float | None],
) -> float | None:
    """Calculate the signed power value of a channel based on its transform."""
    for entity_id in channel.entity_ids:
        if values.get(entity_id) is None:
            return None

    if channel.transform == Transform.DIRECT:
        return values[channel.entity_ids[0]]
    if channel.transform == Transform.INVERTED:
        return -values[channel.entity_ids[0]]
    if channel.transform == Transform.FROM_MINUS_TO:
        return values[channel.entity_ids[0]] - values[channel.entity_ids[1]]

    return None


def gross_w(
    channels: Sequence[GrossChannel],
    values: Mapping[str, float | None],
) -> float | None:
    """Calculate gross household power from channel readings.

    Sums grid, solar (clamped at min 0.0), battery, and home channels.
    Returns None if channels is empty or any channel value is None.
    Gross itself is never clamped (D-11).
    """
    if not channels:
        return None

    total = 0.0
    for channel in channels:
        val = channel_value_w(channel, values)
        if val is None:
            return None

        if channel.role == Role.SOLAR:
            val = max(val, 0.0)

        total += val

    return total


def residual_w(gross: float, metered_w: float) -> float:
    """Calculate signed remainder (gross - metered)."""
    return gross - metered_w


def compute_snapshot(
    model: SourceModel,
    values: Mapping[str, float | None],
    metered_values: Mapping[str, float | None],
    ts: float | None = None,
    accumulator: OverAttribution | None = None,
    *,
    align_status: Any = None,
    plausibility: str = "ok",
) -> Snapshot:
    """Produce a calculation Snapshot from current held states."""
    metered_sum = 0.0
    metered_unavailable: list[str] = []
    for eid, val in metered_values.items():
        if val is None:
            metered_unavailable.append(eid)
        else:
            metered_sum += val

    stale_sources = align_status.stale_sources if align_status else ()
    source_age_s = align_status.source_age_s if align_status else {}
    solar_unavailable = align_status.solar_unavailable if align_status else ()
    on_change_sources = align_status.on_change_sources if align_status else ()

    if align_status and align_status.blocking:
        return Snapshot(
            available=False,
            gross_w=None,
            remainder_w=None,
            signed_residual_w=None,
            metered_w=metered_sum,
            metered_unavailable=tuple(metered_unavailable),
            over_attribution_wh=accumulator.total_wh if accumulator else 0.0,
            clipped=False,
            ts=ts,
            stale_sources=stale_sources,
            source_age_s=source_age_s,
            solar_unavailable=solar_unavailable,
            on_change_sources=on_change_sources,
            plausibility=plausibility,
        )

    gross = gross_w(model.gross_channels, values)

    if gross is None:
        return Snapshot(
            available=False,
            gross_w=None,
            remainder_w=None,
            signed_residual_w=None,
            metered_w=metered_sum,
            metered_unavailable=tuple(metered_unavailable),
            over_attribution_wh=accumulator.total_wh if accumulator else 0.0,
            clipped=False,
            ts=ts,
            stale_sources=stale_sources,
            source_age_s=source_age_s,
            solar_unavailable=solar_unavailable,
            on_change_sources=on_change_sources,
            plausibility=plausibility,
        )

    signed = residual_w(gross, metered_sum)
    remainder = max(0.0, signed)
    clipped = signed < 0.0

    if accumulator is not None and ts is not None:
        accumulator.add(signed, ts)
        over_attribution_wh = accumulator.total_wh
    else:
        over_attribution_wh = accumulator.total_wh if accumulator else 0.0

    return Snapshot(
        available=True,
        gross_w=gross,
        remainder_w=remainder,
        signed_residual_w=signed,
        metered_w=metered_sum,
        metered_unavailable=tuple(metered_unavailable),
        over_attribution_wh=over_attribution_wh,
        clipped=clipped,
        ts=ts,
        stale_sources=stale_sources,
        source_age_s=source_age_s,
        solar_unavailable=solar_unavailable,
        on_change_sources=on_change_sources,
        plausibility=plausibility,
    )


