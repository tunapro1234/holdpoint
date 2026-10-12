"""Batch inference with trained checkpoints; writes a cache the recorded backend can replay.

    python -m holdpoint.vision.infer --seg runs/seg/model.pt --cls runs/cls/model.pt \
        --images data/raw/ozgenel_seg/extracted --list data/splits/ozgenel_seg/test.txt \
        --out results/recorded-ozgenel-test.json --device auto

Output (holdpoint.backends.recorded format): {"provenance": {...}, "results": {"<image sha256>": {"mask": RLE,
"p_crack": p, "file": name}}}, plus a summary with images/s. The key is the SHA-256 of the RGB pixel array as the
app reads it, so the app finds these results when the same photo is opened.
Masks: the photo is resized so its long side is --side, predicted, and the probability map is resized back.
p_crack: the patch classifier runs on non-overlapping tiles of the resized photo; p_crack is the highest tile.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ..backends.base import image_sha256
from ..backends.recorded import mask_to_rle
from . import common

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def load_checkpoint(path: Path, device):
    """Rebuild the model a training script saved. Returns (net, meta)."""
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    meta = ck.get("meta", {})
    if meta.get("task") == "segment":
        from .train_seg import build_model
        net = build_model(meta.get("encoder", "resnet18"), pretrained=False)
    elif meta.get("task") == "classify":
        from .train_cls import build_model
        net = build_model(meta.get("arch", "resnet18"), pretrained=False)
    else:
        raise ValueError(f"{path}: checkpoint without task meta")
    net.load_state_dict(ck["state_dict"])
    return net.to(device).eval(), meta


def _batch(rgbs: list[np.ndarray]):
    import torch

    return torch.stack([common.to_tensor(r) for r in rgbs])


def predict_masks(net, rgbs: list[np.ndarray], device, side: int, threshold: float = 0.5) -> list[np.ndarray]:
    """Full-resolution boolean masks for a batch of same-size RGB photos."""
    import torch

    h, w = rgbs[0].shape[:2]
    s = side / max(h, w)
    nh, nw = max(32, round(h * s)), max(32, round(w * s))
    ph, pw = (-nh) % 32, (-nw) % 32
    small = [np.pad(cv2.resize(r, (nw, nh), interpolation=cv2.INTER_AREA), ((0, ph), (0, pw), (0, 0)),
                    mode="reflect") for r in rgbs]
    with torch.no_grad():
        prob = torch.sigmoid(net(_batch(small).to(device)).float())[:, 0, :nh, :nw].cpu().numpy()
    return [cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR) >= threshold for p in prob]


def predict_crack_prob(net, rgb: np.ndarray, device, tile: int, side: int) -> float:
    import torch

    h, w = rgb.shape[:2]
    s = min(1.0, side / max(h, w))
    small = cv2.resize(rgb, (max(tile, round(w * s)), max(tile, round(h * s))), interpolation=cv2.INTER_AREA)
    tiles = [small[y:y + tile, x:x + tile] for y in range(0, small.shape[0] - tile + 1, tile)
             for x in range(0, small.shape[1] - tile + 1, tile)]
    with torch.no_grad():
        p = torch.softmax(net(_batch(tiles).to(device)).float(), 1)[:, 1]
    return float(p.max())


def image_list(images: Path, list_file: Path | None) -> list[Path]:
    if list_file:
        return [images / r for r in Path(list_file).read_text().splitlines() if r.strip()]
    return sorted(p for p in images.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", type=Path, help="segmentation checkpoint (model.pt)")
    ap.add_argument("--cls", type=Path, help="classifier checkpoint (model.pt)")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--list", type=Path, help="file with paths relative to --images (e.g. a split file)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--side", type=int, default=1024, help="long side the photo is resized to for the UNet")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if not (a.seg or a.cls):
        ap.error("give --seg and/or --cls")

    device = common.pick_device(a.device)
    info = common.device_info(device)
    seg, seg_meta = load_checkpoint(a.seg, device) if a.seg else (None, {})
    cls, cls_meta = load_checkpoint(a.cls, device) if a.cls else (None, {})
    paths = image_list(a.images, a.list)
    if a.limit:
        paths = paths[:a.limit]

    models = []
    if seg is not None:
        models.append(f"unet-{seg_meta.get('encoder')}:{common.sha256_file(a.seg)[:12]}")
    if cls is not None:
        models.append(f"{cls_meta.get('arch')}:{common.sha256_file(a.cls)[:12]}")
    command = "python -m holdpoint.vision.infer " + " ".join(argv if argv is not None else [])
    results, seconds = {}, 0.0
    for p in paths:
        rgb = common.load_rgb(p)
        t0 = time.perf_counter()
        item: dict = {"file": p.name}
        if seg is not None:
            m = predict_masks(seg, [rgb], device, a.side)[0]
            item["mask"] = mask_to_rle(m)
            item["mask_px"] = int(m.sum())
        if cls is not None:
            item["p_crack"] = predict_crack_prob(cls, rgb, device, cls_meta.get("input_size", 224), a.side)
        seconds += time.perf_counter() - t0
        results[image_sha256(rgb)] = item

    gpu = info.get("gpu") or info["device"]
    out = {"provenance": {"backend": "amd_notebook" if info.get("hip") else "batch_infer", "device": gpu,
                          "model": " + ".join(models), "timestamp": datetime.now(timezone.utc).isoformat(
                              timespec="seconds"), "command": command, "torch": info["torch"], "hip": info["hip"]},
           "summary": {"images": len(results), "model_seconds": round(seconds, 3),
                       "images_per_second": round(len(results) / seconds, 3) if seconds else None,
                       "side": a.side},
           "results": results}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out) + "\n")
    print(json.dumps(out["summary"]))
    return out


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
