"""Hold output: ONE line per element that has an open defect.

This is the only payment-side output of HoldPoint. It says "hold this element" and why; what a receiving
system does with it is up to that system. No amounts, no pay-application parsing.

CSV columns: element_id, defect_report_id, status, since, decided_by
Webhook (optional): POST JSON to HOLDPOINT_HOLD_WEBHOOK, schema below.
"""
from __future__ import annotations

import csv
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .lifecycle import DefectStore

COLUMNS = ("element_id", "defect_report_id", "status", "since", "decided_by")
SCHEMA_ID = "holdpoint.hold/v1"


def hold_rows(store: DefectStore) -> list[dict]:
    by_element: dict[str, list] = {}
    for d in store.active():
        by_element.setdefault(d.element_id, []).append(d)
    rows = []
    for element_id in sorted(by_element):
        ds = sorted(by_element[element_id], key=lambda d: (d.opened_at, d.defect_id))
        rows.append({
            "element_id": element_id,
            "defect_report_id": ";".join(d.report_id for d in ds),
            "status": "hold",
            "since": ds[0].opened_at,
            "decided_by": ds[0].opened_by,
        })
    return rows


def write_csv(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def webhook_payload(rows: list[dict], generated_at: str | None = None) -> dict:
    return {
        "schema": SCHEMA_ID,
        "generated_at": generated_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "holds": rows,
    }


def validate_payload(p: dict) -> None:
    if set(p) != {"schema", "generated_at", "holds"}:
        raise ValueError(f"payload keys {sorted(p)}")
    if p["schema"] != SCHEMA_ID:
        raise ValueError(f"schema {p['schema']!r}")
    datetime.fromisoformat(p["generated_at"])
    if not isinstance(p["holds"], list):
        raise ValueError("holds must be a list")
    seen = set()
    for r in p["holds"]:
        if set(r) != set(COLUMNS):
            raise ValueError(f"hold row keys {sorted(r)}")
        if r["status"] != "hold":
            raise ValueError("status must be 'hold'")
        if not all(isinstance(r[k], str) and r[k] for k in COLUMNS):
            raise ValueError("every hold field is a non-empty string")
        if r["element_id"] in seen:
            raise ValueError(f"element {r['element_id']} appears twice")
        seen.add(r["element_id"])
        datetime.fromisoformat(r["since"])


def post_webhook(rows: list[dict], url: str | None = None, timeout: float = 10.0) -> int | None:
    """POST the hold list; returns the HTTP status, or None when no URL is configured."""
    url = url or os.environ.get("HOLDPOINT_HOLD_WEBHOOK")
    if not url:
        return None
    payload = webhook_payload(rows)
    validate_payload(payload)
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status
