"""Fake LLM response generators and proposal fixtures for learning tests."""

from collections.abc import Mapping, Sequence
import json


def chat_completion(
    content: str | None = None,
    *,
    tool_arguments: str | None = None,
    finish_reason: str = "stop",
    usage: tuple[int, int] = (2500, 200),
    model: str = "fake-model",
    refusal: str | None = None,
) -> dict[str, object]:
    """Generate an OpenAI-compatible ChatCompletion response dictionary.

    The default prompt count is a plausible ~0.3-0.4 tok/B for the synth request bodies
    (5-8 kB) and stays above the silent-truncation floor of those requests.
    """
    msg: dict[str, object] = {"role": "assistant"}
    if refusal is not None:
        msg["refusal"] = refusal
        msg["content"] = None
    elif tool_arguments is not None:
        msg["content"] = None
        msg["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "submit",
                    "arguments": tool_arguments,
                },
            }
        ]
    else:
        msg["content"] = content

    return {
        "id": "chatcmpl-fake123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": usage[0] + usage[1],
        },
    }


def proposal_for(
    digest: Mapping[str, object],
    devices: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    """Generate a valid proposal dict for a given digest."""
    clusters = digest.get("clusters", [])
    if devices is not None:
        dev_list = list(devices)
    else:
        dev_list = []
        for i, c in enumerate(clusters[:3]):
            cid = c.get("id", f"c{i+1:02d}")
            dev_list.append(
                {
                    "ref": f"d{i+1}",
                    "label": f"Device {i+1}",
                    "category": "unknown",
                    "tracker_kind": "program" if c.get("kind") == "session" else "switched",
                    "member_clusters": [cid],
                    "states": [
                        {
                            "name": "on",
                            "cluster_ref": cid,
                            "role": "main",
                        }
                    ],
                    "phase_hint": "single",
                    "confidence": "medium",
                    "rationale": f"Candidate device from cluster {cid}.",
                    "ambiguous_with": [],
                    "reference_plug": "",
                }
            )

    return {
        "analysis": "Automated baseline proposal for clustered residual episodes.",
        "schema_version": 1,
        "devices": dev_list,
        "rejected_clusters": [],
        "merge_suggestions": [],
    }


def switched_proposal(
    digest: Mapping[str, object],
    *,
    label: str,
    category: str,
    target_w: float,
    states: Sequence[tuple[str, tuple[float, float] | None]] = (("on", None),),
    ref: str = "d1",
) -> dict[str, object]:
    """Generate a proposal for one switched device with one state per (name, dur range).

    Each entry selects the switched cluster whose rise_w p50 is nearest ``target_w``
    among the switched clusters not chosen by an earlier entry and whose dur_min p50
    lies in the entry's dur range (minutes, skipped when None). When no cluster
    qualifies, raise AssertionError listing every cluster as (id, kind, rise p50,
    dur p50) — the responder never falls back to a session cluster (D-22), so a
    learning gap surfaces as a test failure instead of being hidden.
    """
    clusters = [c for c in digest.get("clusters", []) if isinstance(c, Mapping)]

    def _listing() -> str:
        return ", ".join(
            f"({c.get('id')}, {c.get('kind')}, "
            f"rise p50={c.get('rise_w', {}).get('p50')}, "
            f"dur p50={c.get('dur_min', {}).get('p50')})"
            for c in clusters
        )

    chosen: list[tuple[str, str]] = []
    device_states: list[dict[str, object]] = []
    for name, dur_range in states:
        candidates: list[tuple[float, int, str]] = []
        for pos, c in enumerate(clusters):
            if c.get("kind") != "switched":
                continue
            cid = c.get("id")
            if not isinstance(cid, str) or cid in {cid_ for cid_, _ in chosen}:
                continue
            if dur_range is not None:
                dur_p50 = float(c.get("dur_min", {}).get("p50", 0.0) or 0.0)
                if not (dur_range[0] <= dur_p50 <= dur_range[1]):
                    continue
            rise_p50 = float(c.get("rise_w", {}).get("p50", 0.0) or 0.0)
            candidates.append((abs(rise_p50 - target_w), pos, cid))
        if not candidates:
            raise AssertionError(
                f"no switched cluster for state '{name}' near {target_w} W "
                f"with dur p50 in {dur_range}; clusters: {_listing()}"
            )
        candidates.sort(key=lambda item: (item[0], item[1]))
        cid = candidates[0][2]
        chosen.append((cid, name))
        device_states.append({"name": name, "cluster_ref": cid, "role": "main"})

    device = {
        "ref": ref,
        "label": label,
        "category": category,
        "tracker_kind": "switched",
        "member_clusters": [cid for cid, _ in chosen],
        "states": device_states,
        "phase_hint": "single",
        "confidence": "high",
        "rationale": (
            f"Resistive switched cluster near {target_w} W "
            f"with {len(chosen)} state(s) refit from real episodes."
        ),
        "ambiguous_with": [],
        "reference_plug": "",
    }

    return {
        "analysis": f"Identified {label} from switched cluster near {target_w} W.",
        "schema_version": 1,
        "devices": [device],
        "rejected_clusters": [],
        "merge_suggestions": [],
    }


def kettle_proposal(digest: Mapping[str, object]) -> dict[str, object]:
    """Generate a proposal featuring the kettle cluster nearest 2000 W."""
    clusters = digest.get("clusters", [])
    if not clusters:
        raise ValueError("Digest contains no clusters for kettle_proposal")

    best_cluster = min(
        clusters,
        key=lambda c: abs(c.get("rise_w", {}).get("p50", 0) - 2000),
    )
    cid = best_cluster.get("id", "c01")

    device = {
        "ref": "d1",
        "label": "Kettle",
        "category": "kettle",
        "tracker_kind": "switched",
        "member_clusters": [cid],
        "states": [
            {
                "name": "heat",
                "cluster_ref": cid,
                "role": "main",
            }
        ],
        "phase_hint": "single",
        "confidence": "high",
        "rationale": "High-power resistive step around 2000 W typical of an electric kettle.",
        "ambiguous_with": [],
        "reference_plug": "",
    }

    return {
        "analysis": "Identified electric kettle from strong 2kW resistive cluster.",
        "schema_version": 1,
        "devices": [device],
        "rejected_clusters": [],
        "merge_suggestions": [],
    }
