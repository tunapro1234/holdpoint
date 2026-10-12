import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from holdpoint.defects import CLOSED, OPEN, REPAIR_PHOTO, Decision, DefectError, DefectStore, short_id
from holdpoint.defects.export import (
    COLUMNS,
    hold_rows,
    post_webhook,
    validate_payload,
    webhook_payload,
    write_csv,
)

SHA_A, SHA_B, SHA_C = "a" * 64, "b" * 64, "c" * 64
INSP = Decision("inspector-1", ts="2026-10-13T09:00:00+00:00")


@pytest.fixture
def store(tmp_path):
    return DefectStore(tmp_path / "defects.jsonl")


def test_full_lifecycle_and_persistence(store, tmp_path):
    d = store.open("P-14", "NCR-0001", SHA_A, INSP)
    assert d.state == OPEN and d.defect_id == "DF-0001"
    store.add_repair_photo(d.defect_id, "P-14", SHA_B, Decision("site-lead"))
    assert d.state == REPAIR_PHOTO
    store.close(d.defect_id, Decision("inspector-1", note="repair visible, crack sealed"))
    assert d.state == CLOSED and d.closed_by == "inspector-1"
    again = DefectStore(tmp_path / "defects.jsonl")  # state rebuilt from the event file
    assert again.get("DF-0001").state == CLOSED
    assert [e["event"] for e in again.history("P-14")] == ["opened", "repair_photo", "closed"]


@pytest.mark.parametrize("bad", [None, "inspector-1", {"who": "x"}, Decision(""), Decision("  "),
                                 Decision("model"), Decision("AI"), Decision("system")])
def test_transitions_need_a_human_decision(store, bad):
    with pytest.raises(DefectError):
        store.open("P-14", "NCR-1", SHA_A, bad)
    d = store.open("P-14", "NCR-1", SHA_A, INSP)
    with pytest.raises(DefectError):
        store.add_repair_photo(d.defect_id, "P-14", SHA_B, bad)
    store.add_repair_photo(d.defect_id, "P-14", SHA_B, INSP)
    with pytest.raises(DefectError):
        store.close(d.defect_id, bad)
    with pytest.raises(DefectError):
        store.reject_repair(d.defect_id, bad)
    assert d.state == REPAIR_PHOTO


def test_state_has_no_shortcut(store):
    d = store.open("P-14", "NCR-1", SHA_A, INSP)
    assert not hasattr(store, "set_state")
    with pytest.raises(DefectError):
        store.close(d.defect_id, INSP)  # no repair photo yet
    assert d.state == OPEN


def test_repair_photo_must_match_element(store):
    d = store.open("P-14", "NCR-1", SHA_A, INSP)
    with pytest.raises(DefectError):
        store.add_repair_photo(d.defect_id, "P-15", SHA_B, INSP)  # QR says another element
    with pytest.raises(DefectError):
        store.add_repair_photo(d.defect_id, "P-14", SHA_A, INSP)  # same file as the defect photo
    assert d.state == OPEN and d.repair_photo_sha256 is None


def test_rejected_repair_goes_back_to_open_and_cannot_close(store):
    d = store.open("P-14", "NCR-1", SHA_A, INSP)
    store.add_repair_photo(d.defect_id, "P-14", SHA_B, INSP)
    store.reject_repair(d.defect_id, Decision("inspector-1", note="crack still visible"))
    assert d.state == OPEN and d.repair_photo_sha256 is None
    with pytest.raises(DefectError):
        store.close(d.defect_id, INSP)
    store.add_repair_photo(d.defect_id, "P-14", SHA_C, INSP)
    store.close(d.defect_id, INSP)
    assert d.state == CLOSED
    with pytest.raises(DefectError):  # closed is final
        store.add_repair_photo(d.defect_id, "P-14", "d" * 64, INSP)


def test_short_id_history(store):
    sid = short_id("P-14")
    assert sid == short_id("P-14") and sid.startswith("HP-") and len(sid) == 11
    assert sid != short_id("P-15")
    store.open("P-14", "NCR-1", SHA_A, INSP)
    store.open("P-15", "NCR-2", SHA_B, INSP)
    assert [e["report_id"] for e in store.history(sid)] == ["NCR-1"]


# -- export -----------------------------------------------------------------------------------

def populated(store):
    a = store.open("P-14", "NCR-1", SHA_A, Decision("inspector-1", ts="2026-10-13T09:00:00+00:00"))
    store.open("P-14", "NCR-3", SHA_C, Decision("inspector-2", ts="2026-10-13T11:00:00+00:00"))
    b = store.open("P-15", "NCR-2", SHA_B, Decision("inspector-1", ts="2026-10-13T10:00:00+00:00"))
    c = store.open("P-16", "NCR-4", "e" * 64, INSP)
    store.add_repair_photo(b.defect_id, "P-15", "f" * 64, INSP)  # repair photo in, not yet closed: still held
    store.add_repair_photo(c.defect_id, "P-16", "9" * 64, INSP)
    store.close(c.defect_id, INSP)  # closed: no hold
    return a, b, c


def test_hold_rows_only_open_elements_one_line_each(store):
    populated(store)
    rows = hold_rows(store)
    assert [r["element_id"] for r in rows] == ["P-14", "P-15"]
    p14 = rows[0]
    assert p14 == {"element_id": "P-14", "defect_report_id": "NCR-1;NCR-3", "status": "hold",
                   "since": "2026-10-13T09:00:00+00:00", "decided_by": "inspector-1"}
    assert all(set(r) == set(COLUMNS) for r in rows)
    assert not any("amount" in k or "pay" in k for r in rows for k in r)


def test_no_open_defects_no_rows(store):
    assert hold_rows(store) == []


def test_csv(store, tmp_path):
    populated(store)
    path = write_csv(hold_rows(store), tmp_path / "holds.csv")
    with path.open() as fh:
        rd = list(csv.DictReader(fh))
    assert tuple(rd[0].keys()) == COLUMNS and len(rd) == 2 and rd[1]["status"] == "hold"


def test_payload_schema(store):
    populated(store)
    p = webhook_payload(hold_rows(store), generated_at="2026-10-13T12:00:00+00:00")
    validate_payload(p)
    for broken in (
        {**p, "extra": 1},
        {**p, "schema": "other/v1"},
        {**p, "holds": [{**p["holds"][0], "status": "released"}]},
        {**p, "holds": [{**p["holds"][0], "amount": "4800"}]},
        {**p, "holds": [p["holds"][0], p["holds"][0]]},
    ):
        with pytest.raises(ValueError):
            validate_payload(broken)


@pytest.fixture
def receiver():
    got = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            got.append((self.headers["Content-Type"], json.loads(body)))
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)  # local test receiver only
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/holds", got
    srv.shutdown()
    srv.server_close()


def test_webhook_post_local_receiver(store, receiver, monkeypatch):
    url, got = receiver
    populated(store)
    monkeypatch.setenv("HOLDPOINT_HOLD_WEBHOOK", url)
    assert post_webhook(hold_rows(store)) == 204
    ctype, payload = got[0]
    assert ctype == "application/json"
    validate_payload(payload)
    assert [h["element_id"] for h in payload["holds"]] == ["P-14", "P-15"]


def test_webhook_off_without_url(store, monkeypatch):
    monkeypatch.delenv("HOLDPOINT_HOLD_WEBHOOK", raising=False)
    assert post_webhook([]) is None
