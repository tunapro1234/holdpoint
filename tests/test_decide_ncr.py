import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from holdpoint.decide import ACCEPT, GAUGE, NOT_MEASURABLE, REPAIR, ThresholdTable, suggest
from holdpoint.ledger import GENESIS, Ledger, LedgerError
from holdpoint.ncr import (
    DefectReport,
    LlmWriter,
    NumberGuardError,
    TemplateWriter,
    check_numbers,
    draft_report,
    numbers_in,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "thresholds.example.yaml"
TABLE = ThresholdTable.from_dict({"name": "t", "label": "example values, not a standard",
                                  "classes": {"ext": {"max_width_mm": 0.3}}})


# -- decide -----------------------------------------------------------------------

@pytest.mark.parametrize(
    "band,state",
    [
        ((0.10, 0.25), ACCEPT),
        ((0.10, 0.30), ACCEPT),  # touching the limit is still within it
        ((0.20, 0.40), GAUGE),  # straddles
        ((0.30, 0.45), GAUGE),  # lower edge on the limit: could be exactly at it
        ((0.31, 0.50), REPAIR),
        ((0.60, 0.90), REPAIR),
    ],
)
def test_band_vs_limit(band, state):
    s = suggest(band, "ext", TABLE)
    assert s.state == state and s.limit_mm == 0.3 and s.band_mm == band


def test_no_card_not_measurable():
    s = suggest(None, "ext", TABLE)
    assert s.state == NOT_MEASURABLE and s.limit_mm is None


def test_resolution_limited_never_accepts():
    assert suggest((0.05, 0.12), "ext", TABLE, resolution_limited=True).state == GAUGE
    assert suggest((0.35, 0.5), "ext", TABLE, resolution_limited=True).state == REPAIR


def test_unknown_class_and_bad_band():
    with pytest.raises(KeyError):
        suggest((0.1, 0.2), "nope", TABLE)
    with pytest.raises(ValueError):
        suggest((0.3, 0.2), "ext", TABLE)


def test_example_table_is_labelled():
    t = ThresholdTable.load(EXAMPLE)
    assert t.label == "example values, not a standard"
    assert "example values, not a standard" in suggest((0.1, 0.2), "interior_dry", t).table_label
    with pytest.raises(ValueError):
        ThresholdTable.from_dict({"classes": {}})


# -- ncr / number guard -------------------------------------------------------------

def report(band=(0.15, 0.35), suggestion=GAUGE, limit=0.3):
    return DefectReport(element_id="P-14", photo_sha256="ab12" * 16, band_mm=band, suggestion=suggestion,
                        limit_mm=limit, vlm_label="crack", vlm_sentence="Thin line with sharp edges.",
                        model_version="unet-r18:2026-10-13")


def test_numbers_in_ranges_and_ids():
    assert numbers_in("width 0.15-0.35 mm on P-14, 3 cracks") == ["0.15", "0.35", "14", "3"]
    assert numbers_in("hash ab12cd") == []


def test_guard_accepts_numbers_from_fields():
    f = report().fields()
    check_numbers("Element P-14 shows 0.15-0.35 mm against 0.30 mm.", f)  # 0.30 == 0.3


def test_guard_catches_invented_number():
    f = report().fields()
    with pytest.raises(NumberGuardError) as e:
        check_numbers("Element P-14 shows a 0.42 mm crack, about 2 m long.", f)
    assert e.value.invented == ["0.42", "2"]


def test_template_text_passes_guard():
    r = draft_report(report())
    assert r.text_source == "template"
    assert "0.15-0.35 mm" in r.text and "inspector" in r.text


def test_template_no_card():
    r = draft_report(report(band=None, suggestion=NOT_MEASURABLE, limit=None))
    assert "width not measurable" in r.text


def fake_client(reply):
    sent = {}

    def create(**kw):
        sent.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), sent


def test_llm_text_used_when_clean():
    client, sent = fake_client("Element P-14 has a crack of 0.15-0.35 mm; the limit is 0.3 mm. An inspector decides.")
    r = draft_report(report(), LlmWriter(client=client))
    assert r.text_source == "llm" and r.text.startswith("Element P-14")
    assert json.loads(sent["messages"][1]["content"])["element_id"] == "P-14"


def test_llm_invented_number_rejected_falls_back_to_template():
    client, _ = fake_client("Element P-14 has a 0.5 mm crack and should be repaired within 7 days.")
    r = draft_report(report(), LlmWriter(client=client))
    assert r.text == TemplateWriter().write(report().fields())
    assert r.text_source.startswith("template (llm text rejected") and "0.5" in r.text_source


# -- ledger -----------------------------------------------------------------------------

def add(led, **kw):
    base = dict(who="inspector-1", element_id="P-14", photo_sha256="ab12" * 16,
                model_version="unet-r18:2026-10-13", suggestion=GAUGE, decision="approve")
    base.update(kw)
    return led.append(**base)


def test_ledger_chain_and_fields(tmp_path):
    led = Ledger(tmp_path / "decisions.jsonl")
    a = add(led)
    b = add(led, decision="override", objection_codes=["not_a_crack"], note="form oil stain")
    assert a["prev_hash"] == GENESIS and b["prev_hash"] == a["hash"]
    assert {"who", "ts", "model_version", "photo_sha256", "suggestion", "decision", "objection_codes"} <= b.keys()
    led.verify()
    assert len(led.entries()) == 2


@pytest.mark.parametrize("tamper", ["edit", "delete_middle", "reorder"])
def test_ledger_detects_tampering(tmp_path, tamper):
    led = Ledger(tmp_path / "decisions.jsonl")
    add(led)
    add(led, decision="override", objection_codes=["width_wrong"])
    add(led)
    lines = led.path.read_text().splitlines()
    if tamper == "edit":
        e = json.loads(lines[0])
        e["decision"] = "override"
        e["objection_codes"] = ["other"]
        lines[0] = json.dumps(e, sort_keys=True)
    elif tamper == "delete_middle":
        del lines[1]
    else:
        lines[0], lines[1] = lines[1], lines[0]
    led.path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerError):
        led.verify()
    with pytest.raises(LedgerError):  # nothing can be appended onto a broken chain
        add(led)


def test_ledger_rejects_bad_decisions(tmp_path):
    led = Ledger(tmp_path / "d.jsonl")
    with pytest.raises(LedgerError):
        add(led, decision="override")  # no objection code
    with pytest.raises(LedgerError):
        add(led, decision="override", objection_codes=["looks_fine"])
    with pytest.raises(LedgerError):
        add(led, objection_codes=["other"])  # approve with objection
    with pytest.raises(LedgerError):
        add(led, decision="maybe")
    with pytest.raises(LedgerError):
        add(led, who=" ")
    assert led.entries() == []


def test_number_guard_ignores_digits_inside_hashes():
    from holdpoint.ncr import NumberGuardError, check_numbers
    fields = {"element_id": "P-14", "photo_sha256": "39" + "ab" * 31, "band_mm": [0.15, 0.35]}
    check_numbers("Crack width 0.15-0.35 mm on P-14.", fields)
    with pytest.raises(NumberGuardError):
        check_numbers("Crack is 39 cm long.", fields)
