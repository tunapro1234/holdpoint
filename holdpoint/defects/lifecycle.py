"""Defect lifecycle.

    open --(repair photo of the same element)--> repair_photo --(inspector accepts repair)--> closed
                                                       |
                                                       +--(inspector rejects repair)--> open

Every transition takes a Decision by a named human. There is no setter for the state and no automatic
transition: the model can suggest, it cannot open, move or close a defect. A defect cannot close without a
repair photo of the same element.

State is rebuilt from an append-only JSONL event file, so the element history is the event list itself.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

OPEN = "open"
REPAIR_PHOTO = "repair_photo"
CLOSED = "closed"
ACTIVE = (OPEN, REPAIR_PHOTO)  # element is on hold while any defect is in one of these

MACHINE_ACTORS = {"ai", "model", "system", "auto", "vlm", "llm", "holdpoint"}


class DefectError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def short_id(element_id: str) -> str:
    """Stable 8-character id for a QR label: base32 of the element id hash, e.g. 'HP-7K2QX9AB'."""
    digest = hashlib.sha256(element_id.strip().encode()).digest()
    return "HP-" + base64.b32encode(digest)[:8].decode()


@dataclass(frozen=True)
class Decision:
    """A human decision. `who` is a name or role tag typed by the person (e.g. 'inspector-1')."""

    who: str
    note: str = ""
    ts: str = field(default_factory=utc_now)

    def check(self) -> None:
        if not isinstance(self, Decision):
            raise DefectError("a transition needs a Decision")
        who = self.who.strip().lower()
        if not who:
            raise DefectError("a decision needs a person (who)")
        if who in MACHINE_ACTORS:
            raise DefectError(f"{self.who!r} is not a person; only a human decision moves a defect")


@dataclass
class Defect:
    defect_id: str
    element_id: str
    report_id: str
    photo_sha256: str
    opened_at: str
    opened_by: str
    state: str = OPEN
    repair_photo_sha256: str | None = None
    closed_at: str | None = None
    closed_by: str | None = None


def _require_decision(decision) -> Decision:
    if not isinstance(decision, Decision):
        raise DefectError("a transition needs a human Decision")
    decision.check()
    return decision


class DefectStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._defects: dict[str, Defect] = {}
        self._events: list[dict] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    self._apply(json.loads(line))

    # -- events ------------------------------------------------------------------------------
    def _emit(self, event: dict) -> None:
        self._apply(event)  # validates by applying first
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(event, sort_keys=True) + "\n")

    def _apply(self, e: dict) -> None:
        kind = e["event"]
        if kind == "opened":
            if e["defect_id"] in self._defects:
                raise DefectError(f"duplicate defect id {e['defect_id']}")
            self._defects[e["defect_id"]] = Defect(
                defect_id=e["defect_id"], element_id=e["element_id"], report_id=e["report_id"],
                photo_sha256=e["photo_sha256"], opened_at=e["ts"], opened_by=e["who"])
        else:
            d = self._defects[e["defect_id"]]
            if kind == "repair_photo":
                if d.state != OPEN:
                    raise DefectError(f"{d.defect_id}: repair photo only on an open defect (state {d.state})")
                d.state, d.repair_photo_sha256 = REPAIR_PHOTO, e["photo_sha256"]
            elif kind == "repair_rejected":
                if d.state != REPAIR_PHOTO:
                    raise DefectError(f"{d.defect_id}: nothing to reject (state {d.state})")
                d.state, d.repair_photo_sha256 = OPEN, None
            elif kind == "closed":
                if d.state != REPAIR_PHOTO or not d.repair_photo_sha256:
                    raise DefectError(f"{d.defect_id}: cannot close without a repair photo (state {d.state})")
                d.state, d.closed_at, d.closed_by = CLOSED, e["ts"], e["who"]
            else:
                raise DefectError(f"unknown event {kind!r}")
        self._events.append(e)

    # -- transitions (each needs a human Decision) ----------------------------------------------
    def open(self, element_id: str, report_id: str, photo_sha256: str, decision: Decision) -> Defect:
        dec = _require_decision(decision)
        if not element_id.strip() or not report_id.strip():
            raise DefectError("element_id and report_id are required")
        defect_id = f"DF-{len(self._defects) + 1:04d}"
        self._emit({"event": "opened", "defect_id": defect_id, "element_id": element_id,
                    "short_id": short_id(element_id), "report_id": report_id, "photo_sha256": photo_sha256,
                    "who": dec.who, "note": dec.note, "ts": dec.ts})
        return self._defects[defect_id]

    def add_repair_photo(self, defect_id: str, photo_element_id: str, photo_sha256: str,
                         decision: Decision) -> Defect:
        """photo_element_id is the element id read from the QR label in the repair photo."""
        dec = _require_decision(decision)
        d = self.get(defect_id)
        if photo_element_id != d.element_id:
            raise DefectError(f"repair photo is of {photo_element_id!r}, defect is on {d.element_id!r}")
        if photo_sha256 == d.photo_sha256:
            raise DefectError("repair photo is the same file as the defect photo")
        self._emit({"event": "repair_photo", "defect_id": defect_id, "element_id": d.element_id,
                    "photo_sha256": photo_sha256, "who": dec.who, "note": dec.note, "ts": dec.ts})
        return d

    def reject_repair(self, defect_id: str, decision: Decision) -> Defect:
        dec = _require_decision(decision)
        d = self.get(defect_id)
        self._emit({"event": "repair_rejected", "defect_id": defect_id, "element_id": d.element_id,
                    "who": dec.who, "note": dec.note, "ts": dec.ts})
        return d

    def close(self, defect_id: str, decision: Decision) -> Defect:
        dec = _require_decision(decision)
        d = self.get(defect_id)
        self._emit({"event": "closed", "defect_id": defect_id, "element_id": d.element_id,
                    "who": dec.who, "note": dec.note, "ts": dec.ts})
        return d

    # -- queries ----------------------------------------------------------------------------------
    def get(self, defect_id: str) -> Defect:
        try:
            return self._defects[defect_id]
        except KeyError:
            raise DefectError(f"no defect {defect_id}") from None

    def defects(self) -> list[Defect]:
        return list(self._defects.values())

    def active(self) -> list[Defect]:
        return [d for d in self._defects.values() if d.state in ACTIVE]

    def history(self, element: str) -> list[dict]:
        """All events for an element, by element id or by its QR short id, oldest first."""
        return [e for e in self._events
                if element in (e["element_id"], short_id(e["element_id"]))]
