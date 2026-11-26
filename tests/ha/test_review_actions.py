"""Typed-error tests of the shared review backend (D-14).

The four review actions moved into ``review_actions.py`` must raise the same
typed errors the options flow maps back to ``name_invalid`` / ``candidate_gone``,
and must keep the 0.2.0 exception behaviour exactly. The unchanged options net
(``test_review_flow.py``, ``test_device_confirm.py``, ``test_device_merge_retrain.py``,
``test_options_flow.py``) pins the end-to-end behaviour; these tests exercise the
backend in isolation, without an HA setup.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from custom_components.powerguesser import review_actions
from custom_components.powerguesser.learning import CandidateGone


def _hub(
    *,
    visible: list[dict[str, Any]] | None = None,
    records: dict[str, dict[str, Any]] | None = None,
    validated: bool = True,
    record_decision: Any = None,
    confirm: Any = None,
    review: Any = None,
) -> Any:
    """A minimal hub stub: a learning view and a devices view."""
    learning = SimpleNamespace(
        visible_candidates=lambda: list(visible or []),
        candidates_validated=validated,
        candidates_created_at="2026-11-01T00:00:00+00:00",
        async_record_decision=record_decision or AsyncMock(),
        review=review
        or SimpleNamespace(async_record_confirm_override=AsyncMock()),
    )
    devices = SimpleNamespace(
        record=lambda uuid: (records or {}).get(uuid),
        async_confirm=confirm or AsyncMock(),
    )
    return SimpleNamespace(learning=learning, devices=devices)


def test_candidate_gone_is_reexported() -> None:
    """The flow may import CandidateGone from either learning or review_actions."""
    assert review_actions.CandidateGone is CandidateGone


async def test_confirm_invalid_name_raises_invalid_name() -> None:
    """A rejected name raises InvalidName, never the raw normalise_name ValueError."""
    hub = _hub(visible=[{"signature_id": "sig-a"}])
    with pytest.raises(review_actions.InvalidName):
        await review_actions.async_confirm_candidate(hub, "sig-a", "   ")
    hub.devices.async_confirm.assert_not_called()


async def test_confirm_gone_raises_candidate_gone() -> None:
    """A candidate that left the visible set raises CandidateGone."""
    hub = _hub(visible=[])
    with pytest.raises(CandidateGone):
        await review_actions.async_confirm_candidate(hub, "sig-a", "Kettle")
    hub.devices.async_confirm.assert_not_called()


async def test_confirm_uses_the_preview_rule() -> None:
    """scores.validated wins; without it the record's validated flag decides."""
    seen: dict[str, Any] = {}

    async def _confirm(
        cand: Any, name: str, *, preview: bool, trained_at: str | None
    ) -> None:
        seen.update(preview=preview, name=name)

    hub = _hub(
        visible=[{"signature_id": "sig-a", "scores": {"validated": False}}],
        validated=True,
        confirm=_confirm,
    )
    await review_actions.async_confirm_candidate(hub, "sig-a", "  Kettle  ")
    assert seen["preview"] is True
    assert seen["name"] == "Kettle"


async def test_rename_device_without_subentry_raises_candidate_gone() -> None:
    """A device record whose config subentry is gone raises CandidateGone."""
    hass = SimpleNamespace(
        config_entries=SimpleNamespace(async_update_subentry=Mock())
    )
    entry = SimpleNamespace(get_subentries_of_type=lambda kind: [])
    hub = _hub(records={"dev-1": {"subentry_id": "sub-1"}})
    with pytest.raises(CandidateGone):
        await review_actions.async_rename_device(
            hass, entry, hub, "dev-1", "Kettle"
        )
    hass.config_entries.async_update_subentry.assert_not_called()


async def test_rename_device_invalid_name_raises_invalid_name() -> None:
    """The device path validates the name first, before touching the subentry."""
    hass = SimpleNamespace(
        config_entries=SimpleNamespace(async_update_subentry=Mock())
    )
    entry = SimpleNamespace(get_subentries_of_type=lambda kind: [])
    hub = _hub(records={"dev-1": {"subentry_id": "sub-1"}})
    with pytest.raises(review_actions.InvalidName):
        await review_actions.async_rename_device(hass, entry, hub, "dev-1", "")
    hass.config_entries.async_update_subentry.assert_not_called()


async def test_merge_device_unknown_option_raises_invalid_source() -> None:
    """An unresolvable merge pick raises InvalidSource (the flow's candidate_gone)."""
    hub = _hub(records={"dev-1": {"signature_id": "sig-target"}})
    with pytest.raises(review_actions.InvalidSource):
        await review_actions.async_merge_device(hub, "dev-1", ["device_missing"])


async def test_merge_candidate_value_error_propagates() -> None:
    """The candidate merge keeps 0.2.0 behaviour: a ValueError is not caught."""
    calls: list[tuple[str, str, Any]] = []

    async def _decision(
        action: str, sig: str, *, sources: Any = None, name: Any = None
    ) -> None:
        calls.append((action, sig, sources))
        raise ValueError("Sources required for merge")

    hub = _hub(
        visible=[{"signature_id": "sig-a"}, {"signature_id": "sig-b"}],
        record_decision=_decision,
    )
    with pytest.raises(ValueError):
        await review_actions.async_merge_candidate(hub, "sig-a", None)
    assert calls == [("merge", "sig-a", [])]


async def test_dismiss_records_decision() -> None:
    """Dismiss routes through the supervisor's decision record with the action."""
    calls: list[tuple[str, str]] = []

    async def _decision(
        action: str, sig: str, *, sources: Any = None, name: Any = None
    ) -> None:
        calls.append((action, sig))

    hub = _hub(record_decision=_decision)
    await review_actions.async_dismiss_candidate(hub, "sig-a")
    assert calls == [("dismiss", "sig-a")]


async def test_merge_candidate_normalises_a_bare_string() -> None:
    """A bare string source becomes a one-element list, exactly as the flow did."""
    calls: list[Any] = []

    async def _decision(
        action: str, sig: str, *, sources: Any = None, name: Any = None
    ) -> None:
        calls.append(sources)

    hub = _hub(visible=[{"signature_id": "sig-a"}], record_decision=_decision)
    await review_actions.async_merge_candidate(hub, "sig-a", "sig-b")
    assert calls == [["sig-b"]]


def test_gate_rule_and_wording_are_shared() -> None:
    """One gate rule and one wording: the form, the log and the exception agree.

    ``capture_gate`` is the single rule the backend refuses on and the form
    shows ``confirm_anyway`` on; ``captured_runs_phrase`` is the single place
    the singular/plural wording is built. The form-side equivalence
    (``confirm_anyway`` present exactly when ``capture_gate`` is not None) is
    exercised through the real options flow in
    ``tests/ha/test_capture_confirm_gate.py``.
    """

    def capture_preview(n: int) -> dict[str, Any]:
        return {
            "signature_id": "cap_x",
            "origin": "capture",
            "scores": {"validated": False, "n_examples": n},
        }

    assert review_actions.capture_gate(capture_preview(1)) == (1, 3)
    assert review_actions.capture_gate(capture_preview(2)) == (2, 3)
    assert review_actions.capture_gate(capture_preview(3)) is None
    learned = {"signature_id": "sig-a", "scores": {"validated": True, "n_examples": 1}}
    assert review_actions.capture_gate(learned) is None

    assert review_actions.captured_runs_phrase(1) == "1 captured run"
    assert review_actions.captured_runs_phrase(2) == "2 captured runs"
    assert review_actions.captured_runs_phrase(5) == "5 captured runs"

    err = review_actions.PreviewNeedsRuns(2, 3)
    assert err.n == 2
    assert err.required == 3
    assert review_actions.captured_runs_phrase(2) in str(err)


async def test_async_confirm_candidate_raises_preview_needs_runs() -> None:
    """A 2-run capture preview refuses without override and confirms with it."""
    cand = {
        "signature_id": "cap_x",
        "origin": "capture",
        "kind": "switched",
        "scores": {"validated": False, "n_examples": 2},
    }
    override_mock = AsyncMock()
    hub = _hub(
        visible=[cand],
        review=SimpleNamespace(async_record_confirm_override=override_mock),
    )

    with pytest.raises(review_actions.PreviewNeedsRuns) as err:
        await review_actions.async_confirm_candidate(hub, "cap_x", "Kettle")
    assert err.value.n == 2
    assert err.value.required == 3
    hub.devices.async_confirm.assert_not_called()
    override_mock.assert_not_awaited()

    await review_actions.async_confirm_candidate(hub, "cap_x", "Kettle", override=True)
    hub.devices.async_confirm.assert_awaited_once()
    override_mock.assert_awaited_once_with("cap_x", n_runs=2, required=3)
