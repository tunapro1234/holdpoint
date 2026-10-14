from pathlib import Path

from holdpoint.decide import ThresholdTable
from tools.eval_gauge import parse_name, score

TABLE = ThresholdTable.from_dict({"name": "t", "label": "test", "classes": {"a": {"max_width_mm": 0.3}}})


def test_parse_name():
    m = parse_name(Path("dogal_07_0.35_35_20.jpg"))
    assert m["group"] == "dogal" and m["no"] == 7 and m["truth_mm"] == 0.35 and m["angle_deg"] == 20
    assert parse_name(Path("ref_01_na_20_0.jpg"))["truth_mm"] is None
    assert parse_name(Path("IMG_1234.jpg")) is None


def row(truth, band, limited=False, group="dogal"):
    return {"group": group, "truth_mm": truth, "band_mm": band, "resolution_limited": limited}


def test_score_counts_false_accept_and_coverage():
    rows = [
        row(0.40, [0.10, 0.25]),        # accept although above 0.3: false accept, not covered
        row(0.20, [0.15, 0.25]),        # accept, covered
        row(0.25, [0.20, 0.35]),        # crosses the limit: gauge, covered
        row(0.50, None),                # no scale card
        row(None, [0.1, 0.2]),          # no reading: ignored
    ]
    s = score(rows, TABLE)["all"]
    assert s["n"] == 4 and s["not_measured"] == 1
    assert s["coverage"] == round(2 / 3, 3)
    lim = s["per_limit"]["a (0.3 mm)"]
    assert lim["false_accept"] == 1 and lim["readings_above_limit"] == 1
    assert lim["gauge_rate"] == round(1 / 3, 3)


def test_resolution_limited_goes_to_gauge_not_accept():
    s = score([row(0.4, [0.0, 0.2], limited=True)], TABLE)["all"]["per_limit"]["a (0.3 mm)"]
    assert s["false_accept"] == 0 and s["gauge_rate"] == 1.0
