"""Crack width as a range, never a single number.

The mask is first rectified onto the card plane (homography from the ArUco card), so every pixel there has the
same size in mm. Then: skeleton (skimage) -> Euclidean distance transform -> local width at each skeleton
pixel -> p10..p90 band, widened by pixel quantization and the calibration error.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .aruco import Scale

MEASURED = "measured"
NOT_MEASURABLE = "width_not_measurable"
NO_CRACK = "no_crack"

MIN_RESOLVED_PX = 3.0  # cracks narrower than this many pixels are not resolved: band reads low
QUANT_PX = 0.5  # pixel quantization of a binary mask: true width is within half a pixel of 2*d - 1


@dataclass
class WidthResult:
    status: str
    band_mm: tuple[float, float] | None = None
    median_mm: float | None = None
    n_samples: int = 0
    mm_per_px: float | None = None
    calib_rel_err: float | None = None
    resolution_limited: bool = False
    reason: str = ""

    def text(self) -> str:
        if self.status != MEASURED:
            return self.status.replace("_", " ")
        lo, hi = self.band_mm
        return f"{lo:.2f}-{hi:.2f} mm"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["text"] = self.text()
        return d


def local_widths_px(mask: np.ndarray, end_trim: bool = True) -> np.ndarray:
    """Width in pixels at every skeleton pixel of a binary mask."""
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return np.zeros(0)
    m = np.pad(m, 1)  # distance to the image border counts as background
    dt = distance_transform_edt(m)
    sk = skeletonize(m)
    d = dt[sk]
    if end_trim and d.size:
        # skeleton branches running into the tips/corners of a blob read too thin; drop pixels whose disc
        # touches a skeleton end point
        ends = _skeleton_ends(sk)
        if ends.any():
            from scipy.spatial import cKDTree

            dist, _ = cKDTree(np.argwhere(ends)).query(np.argwhere(sk))
            keep = dist > 2.0 * d
            if keep.sum() >= max(5, 0.2 * d.size):
                d = d[keep]
    return 2.0 * d - 1.0


def _skeleton_ends(sk: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), np.uint8)
    nb = cv2.filter2D(sk.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    return sk & (nb == 2)


def rectify_mask(mask: np.ndarray, scale: Scale) -> tuple[np.ndarray, float]:
    """Warp the mask onto the card plane at roughly the native resolution; returns (mask, mm per rectified px)."""
    m = np.asarray(mask, dtype=np.uint8)
    ys, xs = np.nonzero(m)
    pts = np.array([[xs.min(), ys.min()], [xs.max() + 1, ys.min()], [xs.max() + 1, ys.max() + 1], [xs.min(), ys.max() + 1]],
                   dtype=np.float32).reshape(-1, 1, 2)
    plane = cv2.perspectiveTransform(pts, scale.H).reshape(-1, 2)
    res_mm = scale.mm_per_px  # one rectified pixel ~ one native pixel at the card
    x0, y0 = plane.min(axis=0)
    x1, y1 = plane.max(axis=0)
    pad = 2
    w = int(np.ceil((x1 - x0) / res_mm)) + 2 * pad
    h = int(np.ceil((y1 - y0) / res_mm)) + 2 * pad
    if w * h > 40_000_000:
        raise ValueError("rectified mask too large; check the scale card")
    T = np.array([[1 / res_mm, 0, -x0 / res_mm + pad], [0, 1 / res_mm, -y0 / res_mm + pad], [0, 0, 1]])
    out = cv2.warpPerspective(m.astype(np.float32), T @ scale.H, (w, h), flags=cv2.INTER_LINEAR)
    return out >= 0.5, res_mm


def measure_width(mask: np.ndarray, scale: Scale | None, calib_rel_err: float = 0.03,
                  lo_pct: float = 10.0, hi_pct: float = 90.0, min_samples: int = 10,
                  source_mm_per_px: float | None = None) -> WidthResult:
    """Width band in mm for the cracks in `mask`.

    calib_rel_err: relative scale uncertainty (card print error, corner localisation, out-of-plane surface);
    the band is widened by it on both sides.
    source_mm_per_px: size of one pixel of the image the mask was predicted on, when that was smaller than the
    photo (a resized model input). A crack is only resolved if it spans MIN_RESOLVED_PX of THOSE pixels.
    """
    if scale is None:
        return WidthResult(status=NOT_MEASURABLE, reason="no scale card in view")
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return WidthResult(status=NO_CRACK, mm_per_px=scale.mm_per_px, reason="empty mask")
    rect, res_mm = rectify_mask(m, scale)
    w_px = local_widths_px(rect)
    if w_px.size < min_samples:
        return WidthResult(status=NOT_MEASURABLE, mm_per_px=res_mm,
                           reason=f"too few skeleton samples ({w_px.size})")
    err = calib_rel_err
    eff_mm = max(res_mm, source_mm_per_px or 0.0)
    limited = bool(np.median(w_px) * res_mm < MIN_RESOLVED_PX * eff_mm)
    lo = (np.percentile(w_px, lo_pct) - QUANT_PX) * res_mm * (1 - err)
    hi = (np.percentile(w_px, hi_pct) + QUANT_PX) * res_mm * (1 + err)
    return WidthResult(
        status=MEASURED,
        band_mm=(round(float(max(lo, 0.0)), 3), round(float(hi), 3)),
        median_mm=round(float(np.median(w_px) * res_mm), 3),
        n_samples=int(w_px.size),
        mm_per_px=res_mm,
        calib_rel_err=round(err, 4),
        resolution_limited=limited,
        reason="crack is only a few pixels wide; check with a gauge" if limited else "",
    )
