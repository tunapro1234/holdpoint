import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import make_examples  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_line_check_scores_the_drawn_line():
    truth = make_examples.line_truth(seed=1, width_mm=(0.6, 1.2), card=True, tilt=False)
    assert make_examples.line_check(truth, truth) == {"line_covered": 1.0, "mask_on_line": 1.0}
    off = np.roll(truth, 60, axis=1)  # a mask beside the line
    c = make_examples.line_check(off, truth)
    assert c["line_covered"] < make_examples.MIN_LINE_COVERED and c["mask_on_line"] < make_examples.MIN_MASK_ON_LINE
    assert make_examples.line_check(np.zeros_like(truth), truth)["line_covered"] == 0.0


def test_without_checkpoint_falls_back_to_classical_and_keeps_real_photos(tmp_path):
    real = {"file": "ozgenel-test-001.jpg", "provenance": {"model": "unet-resnet18:model.pt", "synthetic": False}}
    (tmp_path / "recorded.json").write_text(json.dumps({"provenance": {}, "results": {"abc": real}}))
    (tmp_path / "index.json").write_text(json.dumps([{"file": "ozgenel-test-001.jpg", "image_sha256": "abc"}]))

    assert make_examples.main(["--ckpt", str(tmp_path / "missing.pt"), "--out", str(tmp_path)]) == 0
    rec = json.loads((tmp_path / "recorded.json").read_text())
    index = json.loads((tmp_path / "index.json").read_text())
    assert rec["results"]["abc"] == real
    drawn = [v for v in rec["results"].values() if v["file"].startswith("example-")]
    assert len(drawn) == len(make_examples.EXAMPLES)
    for v in drawn:
        assert v["provenance"]["model"] == "classical-blackhat-otsu-v0" and v["provenance"]["synthetic"] is True
        assert "Classical CPU baseline" in v["provenance"]["note"] and "width_masks" not in v
    assert [e["file"] for e in index][-1] == "ozgenel-test-001.jpg"
    assert all("line_covered" in e for e in index if e["file"].startswith("example-"))


def test_bundled_drawn_examples_come_from_the_unet():
    index = json.loads((ROOT / "examples" / "index.json").read_text())
    rec = json.loads((ROOT / "examples" / "recorded.json").read_text())["results"]
    for e in index:
        if e["file"].startswith("example-"):
            assert rec[e["image_sha256"]]["provenance"]["model"].startswith("unet-")
            assert e["line_covered"] >= make_examples.MIN_LINE_COVERED
            assert e["mask_on_line"] >= make_examples.MIN_MASK_ON_LINE
