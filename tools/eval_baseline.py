"""Classical pseudo-mask baseline on the same split, preprocessing and metric as train_seg's evaluation.

Same view as the UNet test numbers: short side resized to --size, centre --size x --size crop, real BW masks,
pixel counts summed over the split (plus mean per-image IoU). CPU only, no torch.

    python tools/eval_baseline.py --part test --out results/baseline-pseudo-test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdpoint.vision import common  # noqa: E402
from holdpoint.vision.pseudo_mask import pseudo_mask  # noqa: E402
from holdpoint.vision.train_seg import _short_side, real_items  # noqa: E402


def centre(a: np.ndarray, s: int) -> np.ndarray:
    h, w = a.shape[:2]
    y0, x0 = (h - s) // 2, (w - s) // 2
    return a[y0:y0 + s, x0:x0 + s]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/raw/ozgenel_seg/extracted"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits/ozgenel_seg"))
    ap.add_argument("--part", default="test", choices=["train", "val", "test"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--exclude", type=Path, help="near_dupes.py output; its near_duplicates are left out")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    items = real_items(a.root, common.read_split_list(a.splits, a.part))
    drop = set(json.loads(a.exclude.read_text())["near_duplicates"]) if a.exclude else set()
    items = [it for it in items if Path(it[0]).name not in drop]
    tp = fp = fn = tn = 0
    ious = []
    for img_p, mask_p, _ in items:
        rgb = _short_side(common.load_rgb(img_p, min_side=a.size), a.size)
        truth = centre(common.load_mask(mask_p, rgb.shape[:2]), a.size) > 0
        pred = centre(pseudo_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)), a.size) > 0
        c = common.binary_counts(pred, truth)
        tp, fp, fn, tn = tp + c[0], fp + c[1], fn + c[2], tn + c[3]
        ious.append(common.prf(*c)["iou"] if c[0] + c[1] + c[2] else 1.0)
    out = {"method": "classical pseudo mask (holdpoint.vision.pseudo_mask, default params)", "part": a.part, "excluded": sorted(drop),
           "view": f"short side {a.size}, centre {a.size}x{a.size} crop (same as train_seg evaluate)",
           **common.prf(tp, fp, fn, tn), "n": len(ious), "mean_image_iou": float(np.mean(ious)) if ious else 0.0,
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "command": "python tools/eval_baseline.py " + " ".join(argv if argv is not None else sys.argv[1:])}
    text = json.dumps(out, indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
