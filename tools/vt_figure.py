"""Picture of one VT test crack: the UNet's width-pass mask and, at each ruler point, the ruler reading next to
the band the app would report.

The numbers are read from results/vt-ruler-unet-<side>.json (written by tools/eval_vt.py), not recomputed; the mask
is recomputed only to draw it, with the same checkpoint and input size. Which crack is shown is decided by a rule,
not by eye: the test crack and distance whose share of held points is closest to the share over all test points
(ties: lowest sample, then nearest distance). Points where the band holds the ruler value are green, misses red,
points with no band grey.

    python tools/vt_figure.py --root ../indirilen/vt-crackroi/Dataset --ckpt runs/seg-gpu-1/model.pt \
        --out docs/vt-test-crack.jpg
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.eval_vt import dist_dir  # noqa: E402

GREEN, RED, GREY, ORANGE = (90, 220, 120), (255, 90, 80), (200, 200, 200), (255, 140, 0)


def held(p: dict) -> bool | None:
    if not p.get("band_mm"):
        return None
    return p["band_mm"][0] <= p["truth_mm"] <= p["band_mm"][1]


def pick(points: list[dict]) -> tuple[int, float]:
    """The (sample, distance) whose held share is closest to the overall held share."""
    found = [p for p in points if held(p) is not None]
    overall = sum(held(p) for p in found) / len(found)
    groups: dict[tuple[int, float], list[dict]] = {}
    for p in found:
        groups.setdefault((p["sample"], p["dist"]), []).append(p)
    return min(groups, key=lambda k: (abs(sum(held(p) for p in groups[k]) / len(groups[k]) - overall), k[0], k[1]))


def draw(rgb: np.ndarray, mask: np.ndarray, points: list[dict], width: int = 1600) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    out[mask] = 0.35 * out[mask] + 0.65 * np.array(ORANGE, np.float32)
    out = out.astype(np.uint8)
    k = width / out.shape[1]
    out = cv2.resize(out, (width, round(out.shape[0] * k)), interpolation=cv2.INTER_AREA)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3
    # labels alternate below-right / above-left along the crack so neighbours do not overlap
    for i, p in enumerate(sorted(points, key=lambda q: -q["x"])):
        x, y = round(p["x"] * k), round(p["y"] * k)
        h = held(p)
        color = GREY if h is None else (GREEN if h else RED)
        band = "no band" if h is None else f"band {p['band_mm'][0]:.1f}-{p['band_mm'][1]:.1f} mm"
        lines = (f"ruler {p['truth_mm']:g} mm", band)
        cv2.circle(out, (x, y), 26, color, 5)
        tw = max(cv2.getTextSize(s, font, scale, thick)[0][0] for s in lines)
        lh = cv2.getTextSize("Ag", font, scale, thick)[0][1] + 18
        x0 = x + 40 if i % 2 == 0 else x - 40 - tw
        y0 = y + 40 if i % 2 == 0 else y - 40 - 2 * lh
        x0 = int(min(max(x0, 8), width - tw - 16))
        y0 = int(min(max(y0, 8), out.shape[0] - 2 * lh - 16))
        bx = min(max(x, x0), x0 + tw + 20)
        by = min(max(y, y0), y0 + 2 * lh + 12)
        cv2.line(out, (x, y), (bx, by), color, 3, cv2.LINE_AA)  # leader: which circle this label belongs to
        box = out[y0:y0 + 2 * lh + 12, x0:x0 + tw + 20].astype(np.float32)
        out[y0:y0 + 2 * lh + 12, x0:x0 + tw + 20] = (0.35 * box + 0.65 * np.array((25, 25, 25))).astype(np.uint8)
        for j, s in enumerate(lines):
            cv2.putText(out, s, (x0 + 10, y0 + lh * (j + 1)), font, scale, color, thick, cv2.LINE_AA)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="the Dataset/ folder of the VT zip")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--side", type=int, help="default: chosen_side in vt-side-select.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    side = a.side or json.loads((a.results / "vt-side-select.json").read_text())["chosen_side"]
    points = json.loads((a.results / f"vt-ruler-unet-{side}.json").read_text())["results"]["unet"]["points"]
    s, d = pick(points)
    dd = dist_dir(d)
    img = cv2.cvtColor(cv2.imread(str(a.root / "Cropped" / dd / f"Sample_{s}" / f"Sample_{s}_{dd}_cropped.JPG")),
                       cv2.COLOR_BGR2RGB)

    from holdpoint.backends.cpu import CpuBackend
    mask = CpuBackend(weights=a.ckpt, side=side).segment(img).mask
    pts = [p for p in points if (p["sample"], p["dist"]) == (s, d)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(a.out), cv2.cvtColor(draw(img, mask, pts), cv2.COLOR_RGB2BGR))
    print(json.dumps({"sample": s, "dist_m": d, "side": side, "held": sum(bool(held(p)) for p in pts),
                      "points": len(pts), "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
