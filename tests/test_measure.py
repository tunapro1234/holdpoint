"""Synthetic geometry tests: a drawn 50 mm ArUco marker and straight 'cracks' of known width on a flat plane,
seen head-on and under perspective. These test the measuring code, not crack detection on real photos."""
import cv2
import numpy as np
import pytest

from holdpoint.measure import detect_scale, measure_width
from holdpoint.measure.aruco import ARUCO_DICT
from holdpoint.measure.width import MEASURED, NO_CRACK, NOT_MEASURABLE

PX_PER_MM = 8  # plane resolution of the synthetic scene
PLANE = (1100, 1500)  # h, w in px


def plane_scene(width_mm, with_marker=True, marker_id=0):
    h, w = PLANE
    img = np.full((h, w), 200, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    if with_marker:
        side = int(50 * PX_PER_MM)
        mk = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), marker_id, side)
        y, x = 160, 160
        img[y - 40 : y + side + 40, x - 40 : x + side + 40] = 255  # quiet zone
        img[y : y + side, x : x + side] = mk
    # a slanted straight crack of known width, drawn as an exact rotated rectangle
    p0, p1 = np.array([620.0, 150.0]), np.array([1400.0, 950.0])
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-d[1], d[0]]) * width_mm * PX_PER_MM / 2
    poly = np.array([p0 + n, p1 + n, p1 - n, p0 - n])
    cv2.fillPoly(mask, [np.round(poly * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    img[mask > 0] = 60
    return img, mask


def camera(name):
    h, w = PLANE
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    if name == "rotated":  # in-plane rotation + zoom, no tilt
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 17, 0.9)
        return np.vstack([M, [0, 0, 1]])
    if name == "tilted":  # oblique view: far edge shrinks
        dst = np.float32([[180, 60], [w - 120, 140], [w - 40, h - 20], [30, h - 90]])
        return cv2.getPerspectiveTransform(src, dst)
    raise ValueError(name)


def view(img, mask, H):
    h, w = PLANE
    im = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=200)
    m = cv2.warpPerspective(mask.astype(np.float32), H, (w, h), flags=cv2.INTER_LINEAR) >= 0.5
    return im, m


@pytest.mark.parametrize("cam", ["rotated", "tilted"])
@pytest.mark.parametrize("width_mm", [1.0, 2.0, 4.5])
def test_band_contains_true_width(width_mm, cam):
    img, mask = plane_scene(width_mm)
    im, m = view(img, mask, camera(cam))
    scale = detect_scale(im)
    assert scale is not None
    r = measure_width(m, scale)
    assert r.status == MEASURED
    lo, hi = r.band_mm
    assert lo <= width_mm <= hi, (width_mm, cam, r)
    assert hi - lo < max(0.6 * width_mm, 0.7)  # band is informative, not just wide
    assert r.text().endswith(" mm") and "-" in r.text()
    assert not r.resolution_limited


def test_thin_crack_is_flagged_resolution_limited():
    img, mask = plane_scene(0.4)  # ~3 px wide in the image
    im, m = view(img, mask, camera("tilted"))
    r = measure_width(m, detect_scale(im))
    assert r.status == MEASURED and r.resolution_limited and "gauge" in r.reason


def test_scale_mm_per_px_head_on():
    img, _ = plane_scene(2.0)
    s = detect_scale(img)
    assert s.marker_id == 0
    assert s.mm_per_px == pytest.approx(1 / PX_PER_MM, rel=0.01)


def test_no_card_is_not_measurable():
    img, mask = plane_scene(2.0, with_marker=False)
    assert detect_scale(img) is None
    r = measure_width(mask, detect_scale(img))
    assert r.status == NOT_MEASURABLE and r.band_mm is None
    assert r.text() == "width not measurable"


def test_wrong_marker_id_is_ignored():
    img, _ = plane_scene(2.0, marker_id=3)
    assert detect_scale(img, marker_id=0) is None
    assert detect_scale(img, marker_id=3).marker_id == 3


def test_empty_mask_no_crack():
    img, mask = plane_scene(2.0)
    r = measure_width(np.zeros_like(mask), detect_scale(img))
    assert r.status == NO_CRACK


def test_calibration_error_widens_band():
    img, mask = plane_scene(2.0)
    s = detect_scale(img)
    a = measure_width(mask, s, calib_rel_err=0.0).band_mm
    b = measure_width(mask, s, calib_rel_err=0.1).band_mm
    assert b[0] < a[0] and b[1] > a[1]


def test_make_card_pdf(tmp_path):
    from pypdf import PdfReader

    from tools.make_card import make_card

    out = make_card(tmp_path / "card.pdf")
    assert out.read_bytes()[:5] == b"%PDF-"
    page = PdfReader(str(out)).pages[0]
    assert float(page.mediabox.width) == pytest.approx(595.27, abs=1)  # A4
    assert "100%" in page.extract_text()
