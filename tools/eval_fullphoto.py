"""UNet and the classical baseline side by side in the app's view: the whole photo, scored at long side --ref.

train_seg's evaluate (and eval_seg / eval_baseline) score a 256 x 256 centre crop at 256 px. The app runs on the
whole photo, so this tool scores both methods the way eval_scale does: the photo is resized to long side --ref,
each method sees it at long side <side>, its mask is resized back, IoU is counted at --ref. Reporting only: the
app's side was chosen on val (results/seg-scale.json); nothing here selects anything.

    python tools/eval_fullphoto.py --ckpt runs/seg-gpu-1/model.pt --exclude results/near-dupes-test.json \
        --out results/fullphoto-test.json
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
from holdpoint.vision.infer import load_checkpoint, predict_masks  # noqa: E402
from holdpoint.vision.pseudo_mask import pseudo_mask  # noqa: E402
from holdpoint.vision.train_seg import real_items  # noqa: E402


def classical(rgb: np.ndarray, side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = side / max(h, w)
    small = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    m = pseudo_mask(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0


def score(preds_truths) -> dict:
    tp = fp = fn = tn = 0
    ious = []
    for pred, truth in preds_truths:
        c = common.binary_counts(pred, truth)
        tp, fp, fn, tn = tp + c[0], fp + c[1], fn + c[2], tn + c[3]
        ious.append(common.prf(*c)["iou"] if c[0] + c[1] + c[2] else 1.0)
    return {**common.prf(tp, fp, fn, tn), "mean_image_iou": float(np.mean(ious)), "n": len(ious)}


def main(argv=None) -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("data/raw/ozgenel_seg/extracted"))
    ap.add_argument("--splits", type=Path, default=Path("data/splits/ozgenel_seg"))
    ap.add_argument("--part", default="test", choices=["train", "val", "test"])
    ap.add_argument("--sides", default="341,512", help="341 = short side 256 on a 4:3 photo, the training scale")
    ap.add_argument("--ref", type=int, default=1024)
    ap.add_argument("--exclude", type=Path, help="near_dupes.py output; reported with and without these photos")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    torch.set_num_threads(4)
    device = torch.device("cpu")
    net, _ = load_checkpoint(a.ckpt, device)
    drop = set(json.loads(a.exclude.read_text())["near_duplicates"]) if a.exclude else set()

    data = []
    for img_p, mask_p, _ in real_items(a.root, common.read_split_list(a.splits, a.part)):
        rgb = common.load_rgb(img_p, min_side=a.ref // 2)
        s = a.ref / max(rgb.shape[:2])
        rgb = cv2.resize(rgb, (round(rgb.shape[1] * s), round(rgb.shape[0] * s)), interpolation=cv2.INTER_AREA)
        data.append((Path(img_p).name, rgb, common.load_mask(mask_p, rgb.shape[:2]) > 0))

    rows = []
    for side in [int(s) for s in a.sides.split(",")]:
        for method, fn in (("unet", lambda rgb, sd=side: predict_masks(net, [rgb], device, sd)[0]),
                           ("classical", lambda rgb, sd=side: classical(rgb, sd))):
            preds = [(name, fn(rgb), truth) for name, rgb, truth in data]
            row = {"method": method, "side": side, "all": score([(p, t) for _, p, t in preds])}
            if drop:
                row["without_excluded"] = score([(p, t) for n, p, t in preds if n not in drop])
            print(json.dumps({"method": method, "side": side, "iou": round(row["all"]["iou"], 4),
                              "iou_nodup": round(row.get("without_excluded", row["all"])["iou"], 4)}), flush=True)
            rows.append(row)

    out = {"checkpoint": str(a.ckpt), "part": a.part, "ref_long_side": a.ref, "threshold": 0.5,
           "view": f"whole photo, method input at long side <side>, mask resized back, scored at long side {a.ref}",
           "excluded": sorted(drop), "rows": rows,
           "note": "reporting only; the app's side was selected on val in results/seg-scale.json",
           "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "command": "python tools/eval_fullphoto.py " + " ".join(argv if argv is not None else sys.argv[1:])}
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
