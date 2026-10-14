import numpy as np

from tools.eval_vt import local_band_px, read_points, skeleton_widths, summarize


def test_local_band_on_drawn_bar():
    m = np.zeros((60, 200), bool)
    m[25:34, 10:190] = True  # 9 px wide
    yx, w = skeleton_widths(m)
    lo, hi, med = local_band_px(yx, w, x=100, y=29, radius_px=10)
    assert abs(med - 9) <= 1 and lo <= 9 <= hi
    assert local_band_px(yx, w, x=100, y=5, radius_px=3) is None  # no skeleton near this point


def test_read_points(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("﻿Sample,Resolution,ROI,X,Y,Crack Ruler Measurement\nSample_3,0.75,2,898,1359,0.5\n", encoding="utf-8")
    assert read_points(p) == [{"sample": 3, "dist": 0.75, "roi": 2, "x": 898, "y": 1359, "truth_mm": 0.5}]


def test_summarize_coverage():
    rows = [{"dist": 0.5, "truth_mm": 1.0, "band_mm": [0.8, 1.2], "median_mm": 1.0},
            {"dist": 0.5, "truth_mm": 3.0, "band_mm": [1.0, 2.0], "median_mm": 1.5},
            {"dist": 1.0, "truth_mm": 0.3, "band_mm": None, "median_mm": None}]
    s = summarize(rows)
    assert s["all"]["found"] == 2 and s["all"]["coverage"] == 0.5
    assert s["ruler >=2 mm"]["bias_mm"] == -1.5
