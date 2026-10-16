"""Build the synthetic example set shown in the app's recorded mode.

These are DRAWN images (noise texture + a drawn dark line + a drawn ArUco card), not photographs of
concrete. The recorded results are computed here on CPU with the trained UNet (--ckpt, default
runs/seg-gpu-1/model.pt), the same chain the app runs live: display mask at the app's input size, plus the
width-pass mask at the size the card asks for. Each result is labelled with its model and command.
Before anything is written, the UNet mask is checked against the drawn line: if it misses the line or
marks the background, nothing is written and the script exits with code 1. The check numbers go into index.json.
Without the checkpoint (or with --backend classical) the classical CPU baseline is used and labelled so.
Real-photo examples written by tools/make_ozgenel_examples.py are kept.

    python tools/make_examples.py [--backend unet|classical] [--ckpt runs/seg-gpu-1/model.pt]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from holdpoint.backends.base import image_sha256  # noqa: E402
from holdpoint.backends.cpu import BASELINE_MODEL, CpuBackend  # noqa: E402
from holdpoint.backends.recorded import mask_to_rle  # noqa: E402
from holdpoint.measure import detect_scale  # noqa: E402
from holdpoint.measure.aruco import ARUCO_DICT  # noqa: E402
from holdpoint.pipeline import card_exclusion, width_input_side  # noqa: E402

PX_PER_MM = 8
PREFIX = "example-"
DEFAULT_CKPT = Path("runs/seg-gpu-1/model.pt")
# the UNet mask must cover this share of the drawn line's centre line, and this share of the mask must lie on the line
MIN_LINE_COVERED = 0.9
MIN_MASK_ON_LINE = 0.9


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
    draw_line(img, seed, width_mm, 60.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    rgb = np.stack([img] * 3, axis=-1)
    return warp(rgb, (165, 165, 165)) if tilt else rgb


def draw_line(img: np.ndarray, seed: int, width_mm: tuple[float, float], value: float) -> None:
    """A wandering line whose width changes along its length."""
    xs = np.linspace(640, 1120, 60)
    ys = 120 + (xs - 640) * 1.3 + 25 * np.sin(xs / 45 + seed)
    widths = np.linspace(*width_mm, len(xs)) * PX_PER_MM
    for (x0, y0, x1, y1, wd) in zip(xs[:-1], ys[:-1], xs[1:], ys[1:], widths):
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), value, max(1, int(round(wd))), cv2.LINE_AA)


def warp(img: np.ndarray, border) -> np.ndarray:
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[60, 30], [w - 40, 80], [w - 20, h - 10], [20, h - 60]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=border)


def line_truth(seed: int, width_mm: tuple[float, float], card: bool, tilt: bool) -> np.ndarray:
    """Where the line was drawn, as a boolean mask of the example image."""
    img = np.zeros((900, 1200), np.uint8)
    draw_line(img, seed, width_mm, 255)
    img = warp(img, 0) if tilt else img
    return img >= 128


def line_check(mask: np.ndarray, truth: np.ndarray) -> dict:
    """Share of the drawn centre line under the mask, and share of the mask on (or within 3 px of) the line."""
    from skimage.morphology import skeletonize

    near = cv2.dilate(truth.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return {"line_covered": round(float(mask[skeletonize(truth)].mean()), 3),
            "mask_on_line": round(float((mask & near).sum() / max(int(mask.sum()), 1)), 3)}


EXAMPLES = [
    ("example-1-card-head-on", dict(seed=1, width_mm=(0.6, 1.2), card=True, tilt=False)),
    ("example-2-card-tilted", dict(seed=2, width_mm=(0.3, 0.5), card=True, tilt=True)),
    ("example-3-no-card", dict(seed=3, width_mm=(0.6, 1.0), card=False, tilt=False)),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["unet", "classical"], default="unet")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT, help="UNet checkpoint (default %(default)s)")
    ap.add_argument("--out", type=Path, default=Path("examples"))
    a = ap.parse_args(argv)
    cmd = " ".join(["python tools/make_examples.py", *(argv if argv is not None else sys.argv[1:])])
    a.out.mkdir(parents=True, exist_ok=True)

    use_unet = a.backend == "unet" and a.ckpt.is_file()
    if a.backend == "unet" and not use_unet:
        print(f"no UNet checkpoint at {a.ckpt}; using the classical CPU baseline", file=sys.stderr)
    be = CpuBackend(weights=a.ckpt if use_unet else None)
    model = be._model_name()
    if use_unet and model == BASELINE_MODEL:
        print("torch or segmentation_models_pytorch missing; using the classical CPU baseline", file=sys.stderr)
        use_unet = False
    ckpt_sha = hashlib.sha256(a.ckpt.read_bytes()).hexdigest() if use_unet else None
    when = datetime.now(timezone.utc).isoformat(timespec="seconds")
    device = f"CPU ({platform.machine()})"
    if use_unet:
        note = ("Drawn example image, not a photograph. UNet trained on NVIDIA RTX 3060, run here on CPU. "
                "Not computed on AMD.")
    else:
        note = "Drawn example image, not a photograph. Classical CPU baseline, not the trained UNet. Not computed on AMD."

    results, index = {}, []
    for name, kw in EXAMPLES:
        rgb = scene(**kw)
        path = a.out / f"{name}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)  # hash what the app will read
        sha = image_sha256(rgb)
        seg = be.segment(rgb)
        scale = detect_scale(rgb)
        check = line_check(seg.mask & ~card_exclusion(seg.mask.shape, scale), line_truth(**kw))
        print(f"{name}: {model}, drawn line covered {check['line_covered']:.3f}, "
              f"mask on the line {check['mask_on_line']:.3f}", flush=True)
        if use_unet and (check["line_covered"] < MIN_LINE_COVERED or check["mask_on_line"] < MIN_MASK_ON_LINE):
            print(f"{name}: the mask does not match the drawn line "
                  f"(need {MIN_LINE_COVERED} / {MIN_MASK_ON_LINE}); nothing written", file=sys.stderr)
            return 1
        prov = {"recorded_backend": "cpu", "device": device, "model": model, "timestamp": when, "command": cmd,
                "synthetic": True, "note": note, "label": f"computed on {device}, {when}, {cmd}"}
        if use_unet:
            prov |= {"checkpoint_sha256": ckpt_sha, "unet_side": be.side}
        item = {"mask": mask_to_rle(seg.mask), "p_crack": be.classify(rgb).p_crack, "file": path.name,
                "provenance": prov}
        side = width_input_side(rgb.shape, scale)
        if use_unet and side is not None:
            item["width_masks"] = {str(side): mask_to_rle(be.segment_at(rgb, side).mask)}
        results[sha] = item
        index.append({"file": path.name, "image_sha256": sha, "synthetic": True, **kw, **check})

    rec_p, idx_p = a.out / "recorded.json", a.out / "index.json"
    rec = json.loads(rec_p.read_text()) if rec_p.exists() else {"results": {}}
    old_index = json.loads(idx_p.read_text()) if idx_p.exists() else []
    keep = {k: v for k, v in rec["results"].items() if not str(v.get("file", "")).startswith(PREFIX)}
    rec = {"provenance": {"backend": "cpu", "device": device, "model": "per result", "timestamp": when,
                          "command": cmd, "note": "Bundled examples; every result carries its own provenance."},
           "results": {**results, **keep}}
    index += [e for e in old_index if not e["file"].startswith(PREFIX)]
    rec_p.write_text(json.dumps(rec))
    idx_p.write_text(json.dumps(index, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
