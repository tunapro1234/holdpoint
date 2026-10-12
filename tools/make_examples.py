"""Build the synthetic example set shown in the app's recorded mode.

These are DRAWN images (noise texture + a drawn dark line + a drawn ArUco card), not photographs of
concrete. The recorded results are computed here on CPU with the classical baseline and labelled so.
When results computed on AMD hardware exist, point HOLDPOINT_RECORDED at that file instead.

    python tools/make_examples.py [out_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from holdpoint.backends.base import image_sha256  # noqa: E402
from holdpoint.backends.cpu import CpuBackend  # noqa: E402
from holdpoint.backends.recorded import mask_to_rle  # noqa: E402
from holdpoint.measure.aruco import ARUCO_DICT  # noqa: E402

PX_PER_MM = 8
COMMAND = "python tools/make_examples.py"


def scene(seed: int, width_mm: tuple[float, float], card: bool, tilt: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = 900, 1200
    base = rng.normal(165, 12, (h // 4, w // 4)).astype(np.float32)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    img += rng.normal(0, 5, (h, w))
    if card:
        side = 50 * PX_PER_MM
        mk = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), 0, side)
        y = x = 120
        img[y - 40 : y + side + 40, x - 40 : x + side + 40] = 250
        img[y : y + side, x : x + side] = mk
    # a wandering dark line whose width changes along its length
    xs = np.linspace(640, 1120, 60)
    ys = 120 + (xs - 640) * 1.3 + 25 * np.sin(xs / 45 + seed)
    widths = np.linspace(*width_mm, len(xs)) * PX_PER_MM
    for (x0, y0, x1, y1, wd) in zip(xs[:-1], ys[:-1], xs[1:], ys[1:], widths):
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), 60.0, max(1, int(round(wd))), cv2.LINE_AA)
    img = np.clip(img, 0, 255).astype(np.uint8)
    rgb = np.stack([img] * 3, axis=-1)
    if tilt:
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[60, 30], [w - 40, 80], [w - 20, h - 10], [20, h - 60]])
        rgb = cv2.warpPerspective(rgb, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(165, 165, 165))
    return rgb


EXAMPLES = [
    ("example-1-card-head-on", dict(seed=1, width_mm=(0.6, 1.2), card=True, tilt=False)),
    ("example-2-card-tilted", dict(seed=2, width_mm=(0.3, 0.5), card=True, tilt=True)),
    ("example-3-no-card", dict(seed=3, width_mm=(0.6, 1.0), card=False, tilt=False)),
]


def main(out_dir: str = "examples") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    be = CpuBackend(weights=None)
    results, index = {}, []
    for name, kw in EXAMPLES:
        rgb = scene(**kw)
        path = out / f"{name}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)  # hash what the app will read
        sha = image_sha256(rgb)
        seg = be.segment(rgb)
        results[sha] = {"mask": mask_to_rle(seg.mask), "p_crack": be.classify(rgb).p_crack, "file": path.name}
        index.append({"file": path.name, "image_sha256": sha, "synthetic": True, **kw})
    doc = {
        "provenance": {
            "backend": "cpu",
            "device": "cpu",
            "model": seg.provenance["model"],
            "timestamp": seg.provenance["timestamp"],
            "command": COMMAND,
            "synthetic": True,
            "note": "Drawn example images, classical CPU baseline. Not photographs, not computed on AMD.",
        },
        "results": results,
    }
    (out / "recorded.json").write_text(json.dumps(doc))
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return out


if __name__ == "__main__":
    print(main(*sys.argv[1:]))
