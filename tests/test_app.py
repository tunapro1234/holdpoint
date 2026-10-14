"""Every page opens without an exception (torch not needed); Benchmarks shows provenance."""
import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = ["Home.py"] + sorted(p.relative_to(APP).as_posix() for p in (APP / "pages").glob("*.py"))


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("HOLDPOINT_STATE", str(tmp_path / "state"))
    monkeypatch.setenv("HOLDPOINT_RESULTS", str(tmp_path / "results"))
    (tmp_path / "results").mkdir()
    return tmp_path


def run(page):
    at = AppTest.from_file(str(APP / page), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def all_text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [x.value for x in at.info] + [x.value for x in at.warning] + [x.value for x in at.success]
    parts += [str(j.value) for j in at.json]
    return "\n".join(parts)


def test_five_pages_exist():
    assert len(PAGES) == 6


@pytest.mark.parametrize("page", PAGES)
def test_page_opens(page):
    run(page)


def test_inspect_shows_result_and_badge():
    at = run("pages/1_Inspect.py")
    text = all_text(at)
    assert "SYNTHETIC EXAMPLE" in text  # bundled examples are drawn, and say so
    assert "example values, not a standard" in text
    assert any(m.label == "Crack width" for m in at.metric)


def test_inspect_to_review_to_defect(isolated):
    at = run("pages/1_Inspect.py")
    at.selectbox[0].set_value(next(o for o in at.selectbox[0].options if "head-on" in o)).run()
    at.button[0].click().run()  # Send to review queue
    assert not at.exception
    queue = (isolated / "state" / "queue.jsonl").read_text().splitlines()
    assert len(queue) == 1

    rq = run("pages/2_Review_queue.py")
    rq.text_input(key="who0").set_value("inspector-1")
    rq.checkbox(key="open0").check()
    rq.button[0].click().run()
    assert not rq.exception
    log = [json.loads(l) for l in (isolated / "state" / "decisions.jsonl").read_text().splitlines()]
    assert log[0]["who"] == "inspector-1" and log[0]["decision"] == "approve"

    df = run("pages/3_Defects.py")
    assert "No element is on hold." not in all_text(df)


def test_review_override_without_code_is_refused(isolated):
    at = run("pages/1_Inspect.py")
    at.button[0].click().run()
    rq = run("pages/2_Review_queue.py")
    rq.text_input(key="who0").set_value("inspector-1")
    rq.radio(key="dec0").set_value("override")
    rq.button[0].click().run()
    assert any("objection code" in e.value for e in rq.error)
    assert not (isolated / "state" / "decisions.jsonl").exists()


def test_benchmarks_empty_state():
    at = run("pages/4_Benchmarks.py")
    assert "No benchmark results yet" in all_text(at)


def test_benchmarks_show_provenance(isolated):
    # test fixture, labelled as such; real files come from bench/bench.py
    (isolated / "results" / "bench-fixture.json").write_text(json.dumps({
        "provenance": {"backend": "test", "device": "test-fixture-device", "model": "fixture",
                       "timestamp": "2026-10-12T00:00:00+00:00", "command": "pytest fixture", "synthetic": True},
        "rows": [{"batch": 1, "img_per_s": 1.0}],
    }))
    at = run("pages/4_Benchmarks.py")
    text = all_text(at)
    assert "test-fixture-device" in text and "pytest fixture" in text
    assert "MI300X: not measured yet" in text
    assert "SYNTHETIC EXAMPLE" in text
    assert len(at.dataframe) == 1


def test_benchmarks_file_without_provenance_is_flagged(isolated):
    (isolated / "results" / "bench-x.json").write_text(json.dumps({"rows": [{"a": 1}]}))
    at = run("pages/4_Benchmarks.py")
    assert "no provenance" in all_text(at)


def test_benchmarks_amd_notice_goes_away_with_amd_result(isolated):
    (isolated / "results" / "bench-amd.json").write_text(json.dumps({
        "provenance": {"backend": "test", "device": "AMD Instinct MI300X (fixture)", "timestamp": "t", "command": "c",
                       "synthetic": True},
        "rows": [{"batch": 1}],
    }))
    assert "not measured yet" not in all_text(run("pages/4_Benchmarks.py"))
