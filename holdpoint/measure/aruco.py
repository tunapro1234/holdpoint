"""Scale card detection.

The card carries one DICT_4X4_50 marker whose black square is printed at 50 mm. Its four corners give a
homography from image pixels to the card plane in mm. No marker means no scale: detect_scale returns None.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MARKER_MM = 50.0
ARUCO_DICT = cv2.aruco.DICT_4X4_50


@dataclass
class Scale:
    H: np.ndarray  # 3x3, image px -> plane mm
    corners_px: np.ndarray  # 4x2, TL TR BR BL (image px)
    marker_id: int
    marker_mm: float
    mm_per_px: float  # isotropic estimate at the marker (sqrt of area ratio)
    side_px_spread: float  # (max side - min side) / mean side; grows with perspective

    def to_dict(self) -> dict:
        return {
            "marker_id": self.marker_id,
            "marker_mm": self.marker_mm,
            "mm_per_px": round(self.mm_per_px, 6),
            "side_px_spread": round(self.side_px_spread, 4),
            "corners_px": self.corners_px.round(2).tolist(),
        }


def _gray(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return arr.astype(np.uint8)


def detect_scale(img: np.ndarray, marker_mm: float = MARKER_MM, marker_id: int | None = None) -> Scale | None:
    """Find the scale marker; return None when no (matching) marker is visible."""
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters()
    )
    gray = _gray(img)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    candidates = [(c.reshape(4, 2), int(i)) for c, i in zip(corners, ids.ravel()) if marker_id in (None, int(i))]
    if not candidates:
        return None
    # largest marker in view wins
    pts, mid = max(candidates, key=lambda t: cv2.contourArea(t[0].astype(np.float32)))
    pts = cv2.cornerSubPix(
        gray, pts.astype(np.float32).reshape(-1, 1, 2), (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01),
    ).reshape(4, 2)
    dst = np.array([[0, 0], [marker_mm, 0], [marker_mm, marker_mm], [0, marker_mm]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(pts.astype(np.float32), dst)
    area_px = cv2.contourArea(pts)
    sides = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
    return Scale(
        H=H,
        corners_px=pts,
        marker_id=mid,
        marker_mm=marker_mm,
        mm_per_px=float(np.sqrt(marker_mm**2 / area_px)),
        side_px_spread=float((sides.max() - sides.min()) / sides.mean()),
    )
