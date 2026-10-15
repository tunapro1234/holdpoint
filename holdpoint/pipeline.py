"""One photo through the whole chain: mask -> scale card -> width band -> suggestion -> second check.

Numbers (band, limit) come from deterministic code. The backend only supplies the mask, the crack probability
and the second-check label, and every piece keeps its provenance.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .backends import Backend, image_sha256
from .decide import Suggestion, ThresholdTable, suggest
from .measure import Scale, WidthResult, detect_scale, measure_width

# Width is measured on a second UNet pass with the photo resampled to this many mm per input pixel. Chosen on the
# 5 calibration cracks of the Virginia Tech ruler set (tools/eval_vt.py --report-calib, results/vt-side-select.json):
# long side 1536 there = 238.92 mm / 1536 px. The display mask stays at the backend's default input size.
WIDTH_MM_PER_INPUT_PX = 0.1555


@dataclass
class Inspection:
    image_sha256: str
    mask: np.ndarray
    scale: Scale | None
    width: WidthResult
    suggestion: Suggestion
    p_crack: float | None
    vlm_label: str
    vlm_sentence: str
    provenance: dict = field(default_factory=dict)  # per step: segment / width_segment / classify / vlm
    width_mask: np.ndarray | None = None  # mask the width was measured on, when it differs from `mask`

    def summary(self) -> dict:
        return {
            "image_sha256": self.image_sha256,
            "width": self.width.to_dict(),
            "suggestion": self.suggestion.to_dict(),
            "p_crack": self.p_crack,
            "vlm_label": self.vlm_label,
            "vlm_sentence": self.vlm_sentence,
            "scale": self.scale.to_dict() if self.scale else None,
            "provenance": self.provenance,
        }


def card_exclusion(shape: tuple[int, int], scale: Scale | None, margin: float = 0.35) -> np.ndarray:
    """Mask of the scale card (marker + white border) so its edges are never read as cracks."""
    out = np.zeros(shape, dtype=bool)
    if scale is None:
        return out
    c = scale.corners_px
    centre = c.mean(axis=0)
    grown = centre + (c - centre) * (1 + 2 * margin)
    poly = np.round(grown).astype(np.int32)
    tmp = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(tmp, poly, 1)
    return tmp.astype(bool)


def crack_crop(img: np.ndarray, mask: np.ndarray, pad: int = 32, max_side: int = 512) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return img
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    crop = img[y0:y1, x0:x1]
    s = max_side / max(crop.shape[:2])
    if s < 1:
        crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return crop


def width_input_side(shape: tuple[int, ...], scale: Scale | None) -> int | None:
    """Long side that puts the photo at WIDTH_MM_PER_INPUT_PX, never above the photo's own size."""
    if scale is None:
        return None
    long_px = max(shape[:2])
    return int(min(long_px, max(256, round(long_px * scale.mm_per_px / WIDTH_MM_PER_INPUT_PX))))


def inspect(img: np.ndarray, backend: Backend, table: ThresholdTable, element_class: str,
            calib_rel_err: float = 0.03) -> Inspection:
    seg = backend.segment(img)
    scale = detect_scale(img)
    card = card_exclusion(seg.mask.shape, scale)
    mask = seg.mask & ~card
    prov = {"segment": seg.provenance}
    wmask = None
    side = width_input_side(img.shape, scale)
    if side is not None:
        try:
            wseg = backend.segment_at(img, side)
            wmask, prov["width_segment"] = wseg.mask & ~card, wseg.provenance
        except (NotImplementedError, KeyError):
            pass
    width = measure_width(mask if wmask is None else wmask, scale, calib_rel_err=calib_rel_err)
    sug = suggest(width.band_mm, element_class, table, resolution_limited=width.resolution_limited)
    try:
        cls = backend.classify(img)
        p, prov["classify"] = cls.p_crack, cls.provenance
    except (NotImplementedError, KeyError):
        p = None
    try:
        v = backend.vlm_check(img if not mask.any() else crack_crop(img, mask))
        label, sentence, prov["vlm"] = v.label, v.sentence, v.provenance
    except (NotImplementedError, KeyError):
        label, sentence = "unclear", "No second check available for this image."
    return Inspection(image_sha256(img), mask, scale, width, sug, p, label, sentence, prov, wmask)


def overlay(img: np.ndarray, insp: Inspection) -> np.ndarray:
    """RGB picture for the UI: crack mask in orange, card outline in cyan."""
    rgb = np.asarray(img)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    out = rgb[..., :3].astype(np.float32).copy()
    m = insp.mask
    out[m] = 0.35 * out[m] + 0.65 * np.array([255, 140, 0], np.float32)
    out = out.astype(np.uint8)
    if insp.scale is not None:
        cv2.polylines(out, [np.round(insp.scale.corners_px).astype(np.int32)], True, (0, 200, 255), 3)
    return out
