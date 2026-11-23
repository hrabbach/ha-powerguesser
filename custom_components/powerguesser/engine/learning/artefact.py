"""SignatureArtefact v1 definition, dataclasses, serialization, and fingerprinting."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json

SCHEMA_VERSION = 1


def _stage_template(payload: Mapping[str, object]) -> str:
    """Canonical JSON of a program artefact's stage template (stages + terminal).

    The template lives under ``session`` (``stages`` and ``terminal``, RESEARCH
    Part A3): it is the detected structure of a program appliance, while the
    artefact's ``states`` are the degenerate session rise boxes (RESEARCH
    Pitfall 1).
    """
    session = payload.get("session")
    session = session if isinstance(session, Mapping) else {}
    return json.dumps(
        {"stages": session.get("stages"), "terminal": session.get("terminal")},
        sort_keys=True,
        default=str,
    )


def detection_differs(
    active: Mapping[str, object] | None, new: Mapping[str, object]
) -> bool:
    """Return True when ``new`` differs from the active version's detection template.

    A switched pair compares its ``states`` only — exactly the Phase 4 rule. A
    program pair also compares the stage template (``session["stages"]`` and
    ``session["terminal"]``): a re-learned program whose stage levels or
    durations moved is a differing proposal even though its degenerate session
    states are unchanged (RESEARCH Pitfall 1). Without a usable active version
    everything differs (a device confirmed while a run was in flight).
    """
    if not isinstance(active, Mapping):
        return True
    if json.dumps(active.get("states"), sort_keys=True, default=str) != json.dumps(
        new.get("states"), sort_keys=True, default=str
    ):
        return True
    if active.get("kind") == "program" or new.get("kind") == "program":
        return _stage_template(active) != _stage_template(new)
    return False


def stage_fingerprint(
    session: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    """Level and duration boxes of a program artefact's stage template.

    The stage template carried by ``SignatureArtefact.fingerprint()`` for kind
    ``program``: identity matching compares programs by these boxes instead of
    the degenerate session states (RESEARCH Pitfall 1). An unusable template
    yields an empty list, which the identity IoU reads as "no stages".
    """
    if not isinstance(session, Mapping):
        return []
    stages = session.get("stages")
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        return []
    out: list[dict[str, object]] = []
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        out.append({"level_w": stage.get("level_w"), "dur_s": stage.get("dur_s")})
    return out



@dataclass(frozen=True)
class Box:
    """Interval box with 50th percentile and inclusive lo/hi bounds."""

    lo: float
    p50: float
    hi: float

    def to_dict(self) -> dict[str, float]:
        return {"lo": self.lo, "p50": self.p50, "hi": self.hi}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "Box":
        return cls(lo=float(d["lo"]), p50=float(d["p50"]), hi=float(d["hi"]))


@dataclass(frozen=True)
class State:
    """A distinct power state within an appliance signature."""

    name: str
    rise_w: Box
    fall_w: Box
    dur_s: Box
    plateau_w: Box
    phase_share: dict[str, tuple[float, float]] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "rise_w": self.rise_w.to_dict(),
            "fall_w": self.fall_w.to_dict(),
            "dur_s": self.dur_s.to_dict(),
            "plateau_w": self.plateau_w.to_dict(),
            "phase_share": (
                {k: [v[0], v[1]] for k, v in self.phase_share.items()}
                if self.phase_share is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "State":
        raw_ps = d.get("phase_share")
        phase_share = None
        if raw_ps is not None and isinstance(raw_ps, Mapping):
            phase_share = {str(k): (float(v[0]), float(v[1])) for k, v in raw_ps.items()}
        return cls(
            name=str(d["name"]),
            rise_w=Box.from_dict(d["rise_w"]),
            fall_w=Box.from_dict(d["fall_w"]),
            dur_s=Box.from_dict(d["dur_s"]),
            plateau_w=Box.from_dict(d["plateau_w"]),
            phase_share=phase_share,
        )


@dataclass(frozen=True)
class SignatureArtefact:
    """Locally compiled signature artefact executing NILM detection."""

    schema_version: int
    signature_id: str
    kind: str
    label: str
    category: str
    extraction: dict[str, object]
    states: tuple[State, ...]
    session: dict[str, object] | None
    time_prior: dict[str, list[float]]
    constraints: dict[str, object]
    provenance: dict[str, object]
    scores: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Convert artefact to canonical dictionary in fixed key order."""
        return {
            "schema_version": self.schema_version,
            "signature_id": self.signature_id,
            "kind": self.kind,
            "label": self.label,
            "category": self.category,
            "extraction": self.extraction,
            "states": [s.to_dict() for s in self.states],
            "session": self.session,
            "time_prior": self.time_prior,
            "constraints": self.constraints,
            "provenance": self.provenance,
            "scores": self.scores,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> "SignatureArtefact":
        states_raw = d.get("states", [])
        states = tuple(State.from_dict(s) for s in states_raw)
        return cls(
            schema_version=int(d["schema_version"]),
            signature_id=str(d["signature_id"]),
            kind=str(d["kind"]),
            label=str(d["label"]),
            category=str(d["category"]),
            extraction=dict(d["extraction"]),
            states=states,
            session=dict(d["session"]) if d.get("session") is not None else None,
            time_prior=dict(d["time_prior"]),
            constraints=dict(d["constraints"]),
            provenance=dict(d["provenance"]),
            scores=dict(d["scores"]),
        )

    def fingerprint(self) -> dict[str, object]:
        """Compute structural fingerprint for identity matching.

        A switched artefact keeps exactly its ``kind`` and ``states`` (byte-
        identical to Phase 4). A program artefact adds ``session_stages``: the
        level and duration boxes of its stage template, which is what makes two
        programs comparable (RESEARCH Pitfall 1).
        """
        fingerprint: dict[str, object] = {
            "kind": self.kind,
            "states": [
                {
                    "rise_w": s.rise_w.to_dict(),
                    "fall_w": s.fall_w.to_dict(),
                    "dur_s": s.dur_s.to_dict(),
                    "plateau_w": s.plateau_w.to_dict(),
                    "phase_share": (
                        {k: [v[0], v[1]] for k, v in s.phase_share.items()}
                        if s.phase_share is not None
                        else None
                    ),
                }
                for s in self.states
            ],
        }
        if self.kind == "program":
            fingerprint["session_stages"] = stage_fingerprint(self.session)
        return fingerprint
