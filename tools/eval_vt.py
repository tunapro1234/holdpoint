"""Local crack width at ruler-measured points: Virginia Tech "Crack Quantification ROIs" field set.

Data: Pradhan & Sarlo, Virginia Tech, DOI 10.7294/29361029 (CC BY 4.0). 15 real cracks, each photographed from
0.5, 0.75 and 1.0 m; 5 points per crack whose width was read with a crack ruler (ROI_Measurements.csv,
0.15-8.5 mm). 225 points in all.

The published photos are cropped to the region between the circle markers, and the markers are not in the
images, so mm per pixel is not given. The crops of one crack at three distances cover the same area (pixel size
scales with distance), so we assume every crop spans the same physical side S (mm) and FIT S on the calibration
cracks (--calib, default 1-5) from the ground-truth masks; all numbers are then reported on the other cracks only.
This tests width-from-mask and the band, not the ArUco card chain.

At each point: skeleton pixels of the mask within --radius-mm of (X, Y) -> local widths 2*EDT-1 -> band
(p10 - 0.5 px, p90 + 0.5 px) * mm/px. Masks: ground truth (measurement code alone) and, with --ckpt, the UNet.

    python tools/eval_vt.py --root ../indirilen/vt-crackroi/Dataset --out results/vt-ruler-gt.json
    python tools/eval_vt.py --root ... --ckpt runs/seg-gpu-1/model.pt --side 1024 --out results/vt-ruler-unet.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdpoint.measure.width import QUANT_PX  # noqa: E402

CLASSES = ((0.0, 0.5, "<0.5 mm"), (0.5, 2.0, "0.5-2 mm"), (2.0, 99.0, ">=2 mm"))


def read_points(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return [{"sample": int(r["Sample"].split("_")[1]), "dist": float(r["Resolution"]), "roi": int(r["ROI"]),
                 "x": int(r["X"]), "y": int(r["Y"]), "truth_mm": float(r["Crack Ruler Measurement"])}
                for r in csv.DictReader(f)]


def dist_dir(d: float) -> str:
    return {0.5: "0.5", 0.75: "0.75", 1.0: "1.0"}[d]


def skeleton_widths(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(skeleton pixel coords [N,2] as (y, x), local width in px [N])."""
    from scipy.ndimage import distance_transform_edt
    from skimage.morphology import skeletonize

    m = np.pad(mask.astype(bool), 1)
    dt = distance_transform_edt(m)
    sk = skeletonize(m)
    yx = np.argwhere(sk) - 1
    return yx, 2.0 * dt[sk] - 1.0


def local_band_px(yx: np.ndarray, w: np.ndarray, x: int, y: int, radius_px: float):
    """Band (lo, hi) and median of local widths near (x, y); None if no skeleton within the radius."""
    if not len(w):
        return None
    d = np.hypot(yx[:, 0] - y, yx[:, 1] - x)
    near = w[d <= radius_px]
    if near.size == 0:
        return None
    return (float(np.percentile(near, 10) - QUANT_PX), float(np.percentile(near, 90) + QUANT_PX),
            float(np.median(near)))


def summarize(rows: list[dict]) -> dict:
    def block(rs):
        found = [r for r in rs if r["band_mm"] is not None]
        if not found:
            return {"n": len(rs), "found": 0}
        err = np.array([r["median_mm"] - r["truth_mm"] for r in found])
        rel = np.array([abs(r["median_mm"] - r["truth_mm"]) / r["truth_mm"] for r in found])
        return {"n": len(rs), "found": len(found),
                "coverage": round(float(np.mean([r["band_mm"][0] <= r["truth_mm"] <= r["band_mm"][1] for r in found])), 3),
                "mae_mm": round(float(np.mean(np.abs(err))), 3), "bias_mm": round(float(np.mean(err)), 3),
                "median_rel_err": round(float(np.median(rel)), 3),
                "band_width_median_mm": round(float(np.median([r["band_mm"][1] - r["band_mm"][0] for r in found])), 3)}

    out = {"all": block(rows)}
    for d in sorted({r["dist"] for r in rows}):
        out[f"dist {d} m"] = block([r for r in rows if r["dist"] == d])
    for lo, hi, name in CLASSES:
        out[f"ruler {name}"] = block([r for r in rows if lo <= r["truth_mm"] < hi])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="the Dataset/ folder of the VT zip")
    ap.add_argument("--calib", default="1,2,3,4,5", help="crack samples used only to fit the crop side S")
    ap.add_argument("--radius-mm", type=float, default=3.0)
    ap.add_argument("--ckpt", type=Path, help="UNet checkpoint; without it only ground-truth masks are measured")
    ap.add_argument("--side", type=int, default=1024, help="UNet input long side")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    pts = read_points(a.root / "ROI_Measurements.csv")
    calib = {int(s) for s in a.calib.split(",")}
    images = sorted({(p["sample"], p["dist"]) for p in pts})

    def paths(s, d):
        dd = dist_dir(d)
        return (a.root / "Cropped" / dd / f"Sample_{s}" / f"Sample_{s}_{dd}_cropped.JPG",
                a.root / "Ground_Truth_Masks" / dd / f"Sample_{s}" / f"Sample_{s}_{dd}_cropped_inverted.jpg")

    backend = None
    if a.ckpt:
        from holdpoint.backends.cpu import CpuBackend
        backend = CpuBackend(weights=a.ckpt, side=a.side)

    # per image: skeleton widths for each mask kind, and the image long side (for mm/px = S / side)
    cache = {}
    for s, d in images:
        img_p, gt_p = paths(s, d)
        gt = cv2.imread(str(gt_p), cv2.IMREAD_GRAYSCALE) > 127
        h, w = gt.shape
        entry = {"size": (w, h), "gt": skeleton_widths(gt)}
        if backend is not None and s not in calib:
            rgb = cv2.cvtColor(cv2.imread(str(img_p)), cv2.COLOR_BGR2RGB)
            entry["unet"] = skeleton_widths(backend.segment(rgb).mask)
        cache[(s, d)] = entry

    # fit S on calibration cracks: S = median over points of truth_mm * side_px / median_local_width_px (GT mask)
    est = []
    for p in pts:
        if p["sample"] not in calib:
            continue
        e = cache[(p["sample"], p["dist"])]
        side_px = float(np.mean(e["size"]))
        r = local_band_px(*e["gt"], p["x"], p["y"], radius_px=0.02 * side_px)  # ~4 mm at S~200 mm, before S is known
        if r and r[2] > 0:
            est.append(p["truth_mm"] * side_px / r[2])
    S = float(np.median(est))

    results = {}
    for kind in ["gt"] + (["unet"] if backend is not None else []):
        rows = []
        for p in pts:
            if p["sample"] in calib:
                continue
            e = cache[(p["sample"], p["dist"])]
            mm_px = S / float(np.mean(e["size"]))
            r = local_band_px(*e[kind], p["x"], p["y"], radius_px=a.radius_mm / mm_px)
            rows.append({**p, "mm_per_px": round(mm_px, 5),
                         "band_mm": [round(max(r[0], 0) * mm_px, 3), round(r[1] * mm_px, 3)] if r else None,
                         "median_mm": round(r[2] * mm_px, 3) if r else None})
        results[kind] = {"summary": summarize(rows), "points": rows}

    out = {"data": "Virginia Tech Crack Quantification ROIs, DOI 10.7294/29361029 (CC BY 4.0), Pradhan & Sarlo",
           "scale": {"crop_side_mm_fitted": round(S, 2), "fitted_on_samples": sorted(calib), "n_fit_points": len(est),
                     "note": "markers are cropped out of the published photos; S fitted on calibration cracks only"},
           "radius_mm": a.radius_mm, "unet": {"ckpt": str(a.ckpt), "side": a.side} if a.ckpt else None,
           "test_samples": sorted({s for s, _ in images} - calib),
           "results": results,
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "command": "python tools/eval_vt.py " + " ".join(argv if argv is not None else sys.argv[1:])}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"scale": out["scale"], **{k: v["summary"] for k, v in results.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
