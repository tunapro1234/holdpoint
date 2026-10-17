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
    monkeypatch.setenv("HOLDPOINT_UNET_URL", "")  # tests never download the model
    for k in ("HOLDPOINT_LLM_BASE_URL", "HOLDPOINT_VLLM_BASE_URL"):  # no AMD endpoint unless a test sets one
        monkeypatch.delenv(k, raising=False)
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
    assert "unet-resnet18" in text  # computed by the released UNet, not the classical fallback
    assert "example values, not a standard" in text
    assert any(m.label == "Crack width" for m in at.metric)


def test_inspect_opens_on_the_hairline_example():
    at = run("pages/1_Inspect.py")
    assert "hairline" in str(at.selectbox[0].value)
    assert [m.value for m in at.metric if m.label == "Suggestion"] == ["Measure with a gauge"]


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


def test_inspect_real_photo_example_is_labelled_real():
    at = run("pages/1_Inspect.py")
    opts = [o for o in at.selectbox[0].options if "ozgenel-test" in o]
    assert opts, "bundled real-photo examples missing"
    at.selectbox[0].set_value(opts[0]).run()
    assert not at.exception
    text = all_text(at)
    assert "computed on CPU" in text and "unet-resnet18" in text
    assert any(m.label == "Crack width" for m in at.metric)


RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.mark.skipif(not (RESULTS / "vt-side-select.json").is_file(), reason="ruler check results not in results/")
def test_data_limits_ruler_check_read_from_results(isolated):
    for name in ("vt-side-select.json", "vt-ruler-gt.json"):
        if (RESULTS / name).is_file():
            (isolated / "results" / name).write_text((RESULTS / name).read_text())
    sel = json.loads((RESULTS / "vt-side-select.json").read_text())["test_at_chosen_side"]
    text = all_text(run("pages/5_Data_and_limits.py"))
    mid = sel["ruler 0.5-2 mm"]
    row = next(ln for ln in text.splitlines() if ln.startswith(f"| 0.5-2 mm | {mid['n']} |"))
    assert f"| {mid['coverage'] * 100:.0f}% |" in row  # UNet coverage for 0.5-2 mm, as in the file
    assert "not usable, check with a gauge" in text
    assert "scale card was not part of this test" in text
    assert "0.156 mm on the width pass" in text and "not measured**" not in text


def test_data_limits_without_ruler_results_says_not_measured():
    text = all_text(run("pages/5_Data_and_limits.py"))
    assert "Width against a crack ruler on real cracks: not measured" in text
    assert "examples/ozgenel-test-*" in text and "458 photos" in text


def test_upload_offers_scale_card_and_test_image():
    at = run("pages/1_Inspect.py")
    at.radio[0].set_value("Upload (live on CPU)").run()
    assert not at.exception
    labels = [b.proto.label for b in at.get("download_button")]
    assert any("Scale card (PDF" in x for x in labels) and any("drawn, not a photo" in x for x in labels)


def test_unet_weights_order_and_checksum(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    import app.common as common

    monkeypatch.setattr(common, "LOCAL_UNET", tmp_path / "none.pt")
    monkeypatch.setenv("HOLDPOINT_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("HOLDPOINT_UNET_WEIGHTS", raising=False)
    assert common.unet_weights() is None  # download turned off by the fixture

    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"not the released model")
    monkeypatch.setenv("HOLDPOINT_UNET_URL", wrong.as_uri())
    assert common.unet_weights() is None  # checksum mismatch: not used, not kept
    assert not any((tmp_path / "cache").glob("*"))

    monkeypatch.setattr(common, "UNET_SHA256", common._sha256(wrong))
    assert common.unet_weights() == tmp_path / "cache" / "model.pt"

    monkeypatch.setenv("HOLDPOINT_UNET_WEIGHTS", str(wrong))
    assert common.unet_weights() == wrong


# -- AMD endpoints switched on by env, served by the fake endpoint in tests/conftest.py (no network) -------------

from conftest import chat_reply, unreachable  # noqa: E402

AMD_URL = "http://amd.invalid/v1"


def queue_one_item():
    at = run("pages/1_Inspect.py")
    at.button[0].click().run()  # Send to review queue
    assert not at.exception


def report_text(rq):
    return rq.text_area[0].value, next(c.value for c in rq.caption if c.value.startswith("Text source"))


def test_review_without_llm_env_uses_template():
    queue_one_item()
    text, source = report_text(run("pages/2_Review_queue.py"))
    assert source == "Text source: template"


def test_review_uses_llm_when_env_set(monkeypatch, fake_endpoint):
    queue_one_item()
    monkeypatch.setenv("HOLDPOINT_LLM_BASE_URL", AMD_URL)
    bodies = fake_endpoint(lambda r: chat_reply("Element P-14: the second check saw a crack. An inspector decides."))
    rq = run("pages/2_Review_queue.py")
    text, source = report_text(rq)
    assert source == "Text source: llm" and text.startswith("Element P-14: the second check")
    assert json.loads(bodies[0]["messages"][1]["content"])["element_id"] == "P-14"
    rq.text_input(key="who0").set_value("inspector-1").run()  # a rerun reuses the draft, no second call
    assert len(bodies) == 1


def test_review_llm_unreachable_falls_back_to_template(monkeypatch, fake_endpoint):
    queue_one_item()
    monkeypatch.setenv("HOLDPOINT_LLM_BASE_URL", AMD_URL)
    fake_endpoint(unreachable)
    text, source = report_text(run("pages/2_Review_queue.py"))
    assert source == "Text source: template (llm unreachable: APIConnectionError)"
    assert "An inspector must confirm" in text


def test_review_llm_invented_number_falls_back_to_template(monkeypatch, fake_endpoint):
    queue_one_item()
    monkeypatch.setenv("HOLDPOINT_LLM_BASE_URL", AMD_URL)
    fake_endpoint(lambda r: chat_reply("Element P-14 has a crack about two millimetres wide."))
    text, source = report_text(run("pages/2_Review_queue.py"))
    assert source.startswith("Text source: template (llm text rejected") and "two" not in text


def test_inspect_second_check_on_amd_when_env_set(monkeypatch, fake_endpoint):
    baseline = run("pages/1_Inspect.py")
    monkeypatch.setenv("HOLDPOINT_VLLM_BASE_URL", AMD_URL)
    monkeypatch.setenv("HOLDPOINT_VLLM_DEVICE", "AMD Instinct MI300X")
    bodies = fake_endpoint(lambda r: chat_reply('{"label": "form_oil_stain", "sentence": "A smooth dark smear."}'))
    at = run("pages/1_Inspect.py")
    text = all_text(at)
    assert "**Second check:** form oil stain. A smooth dark smear." in text
    assert "live on AMD · model Qwen/Qwen2.5-VL-7B-Instruct" in text and "AMD Instinct MI300X" in text
    assert bodies[0]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert [(m.label, m.value) for m in at.metric] == [(m.label, m.value) for m in baseline.metric]
    at.selectbox[1].set_value(at.selectbox[1].options[-1]).run()  # a rerun does not ask the endpoint again
    assert len(bodies) == 1


def test_inspect_amd_unreachable_is_unclear_and_measures_anyway(monkeypatch, fake_endpoint):
    baseline = run("pages/1_Inspect.py")
    monkeypatch.setenv("HOLDPOINT_VLLM_BASE_URL", AMD_URL)
    fake_endpoint(unreachable)
    at = run("pages/1_Inspect.py")
    text = all_text(at)
    assert "**Second check:** unclear. Second check unavailable" in text and "APIConnectionError" in text
    assert "live on AMD, no answer (APIConnectionError)" in text
    assert [(m.label, m.value) for m in at.metric] == [(m.label, m.value) for m in baseline.metric]
