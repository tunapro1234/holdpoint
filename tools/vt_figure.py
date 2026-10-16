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

GREEN, RED, GREY, ORANGE = (40, 170, 70), (210, 40, 40), (150, 150, 150), (255, 140, 0)


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
    for p in points:
        x, y = round(p["x"] * k), round(p["y"] * k)
        h = held(p)
        color = GREY if h is None else (GREEN if h else RED)
        band = "no band" if h is None else f"band {p['band_mm'][0]:.1f}-{p['band_mm'][1]:.1f} mm"
        cv2.circle(out, (x, y), 22, color, 4)
        for i, line in enumerate((f"ruler {p['truth_mm']:g} mm", band)):
            org = (min(x + 30, width - 330), y - 8 + 34 * i)
            cv2.putText(out, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 6, cv2.LINE_AA)
            cv2.putText(out, line, org, cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)
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
