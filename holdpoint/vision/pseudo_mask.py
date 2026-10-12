"""Pseudo masks for crack patches that only carry an image-level label (SDNET2018, Özgenel classification set).

A crack is a thin structure darker than the concrete around it. Steps:
grayscale -> median blur -> morphological black-hat (dark details narrower than the kernel) -> Otsu threshold
on the black-hat response -> opening -> drop components that are small or not elongated -> closing.

These masks are weak labels for pre-training the segmentation model. They are never shown as measurements;
crack width is measured only from the model mask on a scale-card photo (holdpoint.measure.width).

    python -m holdpoint.vision.pseudo_mask SRC_DIR OUT_DIR              # write <stem>.png masks
    python -m holdpoint.vision.pseudo_mask --eval IMG_DIR GT_DIR        # IoU against real masks
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class PseudoMaskParams:
    blur: int = 5               # median blur kernel (odd), removes surface grain
    blackhat_frac: float = 0.06  # black-hat kernel as a fraction of the shorter image side
    min_contrast: float = 12.0   # Otsu threshold is never lower than this (gray levels)
    min_area_frac: float = 0.0008  # components smaller than this share of the image are dropped
    min_elongation: float = 3.0  # major/minor axis ratio a component needs to count as crack-like
    close: int = 3              # closing kernel to join broken segments


def _odd(n: float, lo: int = 3) -> int:
    k = max(lo, int(round(n)))
    return k if k % 2 else k + 1


def _elongation(stats_w: int, stats_h: int, pts: np.ndarray) -> float:
    if len(pts) < 5:
        return max(stats_w, stats_h) / max(1, min(stats_w, stats_h))
    _, (a, b), _ = cv2.minAreaRect(pts)
    return max(a, b) / max(1.0, min(a, b))


def pseudo_mask(image: np.ndarray, params: PseudoMaskParams | None = None) -> np.ndarray:
    """Return a uint8 mask (0 or 255) of crack-like dark thin structures in a BGR or gray image."""
    p = params or PseudoMaskParams()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray.astype(np.uint8), _odd(p.blur))
    k = _odd(min(gray.shape) * p.blackhat_frac, lo=7)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    otsu, _ = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(otsu, p.min_contrast)
    mask = (bh > thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    min_area = p.min_area_frac * mask.size
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        ys, xs = np.nonzero(labels[y:y + h, x:x + w] == i)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        if _elongation(w, h, pts) < p.min_elongation:
            continue
        keep[labels == i] = 255
    if p.close > 1:
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((p.close, p.close), np.uint8))
    return keep


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    a, b = pred > 0, gt > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def _images(folder: Path) -> list[Path]:
    return sorted(q for q in folder.rglob("*") if q.suffix.lower() in IMAGE_SUFFIXES)


def write_masks(src: Path, out: Path, params: PseudoMaskParams | None = None) -> int:
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path in _images(src):
        target = out / (img_path.stem + ".png")
        if target.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        cv2.imwrite(str(target), pseudo_mask(img, params))
        n += 1
    return n


def evaluate(img_dir: Path, gt_dir: Path, max_side: int = 1024, params: PseudoMaskParams | None = None) -> dict:
    """Mean IoU of pseudo masks against real masks paired by file stem (images resized to max_side)."""
    gts = {q.stem: q for q in _images(gt_dir)}
    scores = []
    for img_path in _images(img_dir):
        gt_path = gts.get(img_path.stem)
        if gt_path is None:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if img is None or gt is None:
            continue
        if gt.ndim == 3:
            gt = gt[..., 3] if gt.shape[2] == 4 else cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        s = max_side / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        scores.append(iou(pseudo_mask(img, params), gt > 127))
    arr = np.array(scores) if scores else np.array([np.nan])
    return {"n": len(scores), "mean_iou": float(np.nanmean(arr)), "median_iou": float(np.nanmedian(arr)),
            "max_side": max_side, "params": asdict(params or PseudoMaskParams())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--eval", action="store_true", help="src = images, dst = real masks; print mean IoU")
    a = ap.parse_args(argv)
    if a.eval:
        print(json.dumps(evaluate(a.src, a.dst), indent=2))
    else:
        print(f"{write_masks(a.src, a.dst)} masks written to {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
