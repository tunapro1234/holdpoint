"""Which inference scale fits the app? The UNet is trained on short side 256 crops; the app sees whole photos.

For each --sides value: the whole test photo is resized so its long side is that value (predict_masks, as in batch
inference), the probability map is resized back, and IoU is measured against the real mask at long side --ref.
CPU, one checkpoint, no training.

    python tools/eval_scale.py --ckpt runs/seg-gpu-1/model.pt --out results/seg-scale-test.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from holdpoint.vision import common  # noqa: E402
from holdpoint.vision.infer import load_checkpoint, predict_masks  # noqa: E402
from holdpoint.vision.train_seg import real_items  # noqa: E402


def main(argv=None) -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("data/raw/ozgenel_seg/extracted"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits/ozgenel_seg"))
    ap.add_argument("--part", default="test", choices=["train", "val", "test"])
    ap.add_argument("--sides", default="341,512,768,1024")
    ap.add_argument("--ref", type=int, default=1024, help="long side the truth mask is compared at")
    ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke test)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    torch.set_num_threads(4)
    device = torch.device("cpu")
    net, _ = load_checkpoint(a.ckpt, device)
    items = real_items(a.root, common.read_split_list(a.splits, a.part))
    items = items[:a.limit] if a.limit else items
    sides = [int(s) for s in a.sides.split(",")]
    data = []
    for img_p, mask_p, _ in items:
        rgb = common.load_rgb(img_p, min_side=a.ref // 2)
        s = a.ref / max(rgb.shape[:2])
        size = (round(rgb.shape[1] * s), round(rgb.shape[0] * s))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        data.append((rgb, common.load_mask(mask_p, rgb.shape[:2]) > 0))
    rows = []
    for side in sides:
        tp = fp = fn = tn = 0
        ious, t0 = [], time.perf_counter()
        for rgb, truth in data:
            c = common.binary_counts(predict_masks(net, [rgb], device, side)[0], truth)
            tp, fp, fn, tn = tp + c[0], fp + c[1], fn + c[2], tn + c[3]
            ious.append(common.prf(*c)["iou"] if c[0] + c[1] + c[2] else 1.0)
        rows.append({"side": side, **common.prf(tp, fp, fn, tn), "mean_image_iou": float(np.mean(ious)),
                     "cpu_s_per_image": (time.perf_counter() - t0) / len(data), "n": len(data)})
        print(json.dumps(rows[-1]), flush=True)
    out = {"checkpoint": str(a.ckpt), "part": a.part, "ref_long_side": a.ref, "threshold": 0.5, "by_side": rows,
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "command": "python tools/eval_scale.py " + " ".join(argv if argv is not None else sys.argv[1:])}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
