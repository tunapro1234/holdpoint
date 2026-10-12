"""Append-only decision log (JSONL) with a hash chain.

Each line records who decided what, when, on which photo and model version, and links to the previous line
by hash. Editing, deleting or reordering any earlier line breaks verify().
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64
DECISIONS = ("approve", "override")
OBJECTION_CODES = (
    "not_a_crack",
    "width_wrong",
    "wrong_element",
    "photo_unusable",
    "scale_card_issue",
    "threshold_not_applicable",
    "other",
)


class LedgerError(ValueError):
    pass


def _hash(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def append(self, *, who: str, element_id: str, photo_sha256: str, model_version: str, suggestion: str,
               decision: str, objection_codes: list[str] | None = None, note: str = "",
               ts: str | None = None) -> dict:
        codes = list(objection_codes or [])
        if decision not in DECISIONS:
            raise LedgerError(f"decision must be one of {DECISIONS}")
        if not who.strip():
            raise LedgerError("who is required")
        bad = [c for c in codes if c not in OBJECTION_CODES]
        if bad:
            raise LedgerError(f"unknown objection codes {bad}")
        if decision == "override" and not codes:
            raise LedgerError("an override needs at least one objection code")
        if decision == "approve" and codes:
            raise LedgerError("objection codes only go with an override")
        prev = self.entries()
        self.verify(prev)
        record = {
            "seq": len(prev),
            "ts": ts or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "who": who,
            "element_id": element_id,
            "photo_sha256": photo_sha256,
            "model_version": model_version,
            "suggestion": suggestion,
            "decision": decision,
            "objection_codes": codes,
            "note": note,
            "prev_hash": prev[-1]["hash"] if prev else GENESIS,
        }
        record["hash"] = _hash(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def verify(self, entries: list[dict] | None = None) -> None:
        """Raise LedgerError at the first broken link."""
        prev_hash = GENESIS
        for i, e in enumerate(self.entries() if entries is None else entries):
            if e.get("seq") != i:
                raise LedgerError(f"line {i}: sequence {e.get('seq')} out of order")
            if e.get("prev_hash") != prev_hash:
                raise LedgerError(f"line {i}: prev_hash does not match line {i - 1}")
            if _hash(e) != e.get("hash"):
                raise LedgerError(f"line {i}: content changed after it was written")
            prev_hash = e["hash"]
