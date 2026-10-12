"""Crack segmentation: segmentation_models_pytorch UNet on real masks (Özgenel) plus optional pseudo masks.

    python -m holdpoint.vision.train_seg --epochs 30 --device auto --pseudo sdnet2018 --out runs/seg
    python -m holdpoint.vision.train_seg --limit 16 --epochs 1 --weights none                  # CPU smoke test

Real data: data/raw/ozgenel_seg/extracted/{rgb,BW}/<name>.jpg with split files in data/splits/ozgenel_seg/.
Pseudo data (--pseudo sdnet2018 | ozgenel_cls): cracked patches from that dataset's train split, masks from
holdpoint.vision.pseudo_mask, loss weighted by --pseudo-weight. Validation and test use real masks only.
Writes to --out: model.pt (state_dict + meta), model.pt.sha256, metrics.json (pixel IoU, precision, recall on
val and test), train.log.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from . import common
from .pseudo_mask import pseudo_mask
from .train_cls import label_of

MASK_DIR = "BW"
IMG_DIR = "rgb"


def build_model(encoder: str, pretrained: bool):
    import segmentation_models_pytorch as smp

    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3,
                    classes=1)


def _short_side(rgb: np.ndarray, size: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = size / min(h, w)
    return cv2.resize(rgb, (max(size, round(w * s)), max(size, round(h * s))),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)


class SegSet:
    """Items are (image path, mask path | None, weight). mask None = pseudo mask computed on load.
    Images are cached in memory after the first resize (the photos are 3264 x 2448)."""

    def __init__(self, items: list[tuple[Path, Path | None, float]], size: int, train: bool, seed: int = 0):
        self.items, self.size, self.train = items, size, train
        self.rng = np.random.default_rng(seed)
        self.cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        if i not in self.cache:
            img_p, mask_p, _ = self.items[i]
            rgb = _short_side(common.load_rgb(img_p, min_side=self.size), self.size)
            if mask_p is None:
                m = (pseudo_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)) > 0).astype(np.uint8)
            else:
                m = common.load_mask(mask_p, rgb.shape[:2])
            self.cache[i] = (rgb, m)
        return self.cache[i]

    def __getitem__(self, i: int):
        import torch

        rgb, m = self._load(i)
        h, w = m.shape
        s = self.size
        if self.train:
            y0, x0 = int(self.rng.integers(h - s + 1)), int(self.rng.integers(w - s + 1))
        else:
            y0, x0 = (h - s) // 2, (w - s) // 2
        rgb, m = rgb[y0:y0 + s, x0:x0 + s], m[y0:y0 + s, x0:x0 + s]
        if self.train:
            if self.rng.random() < 0.5:
                rgb, m = rgb[:, ::-1], m[:, ::-1]
            if self.rng.random() < 0.5:
                rgb, m = rgb[::-1], m[::-1]
        mask_t = torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32))[None]
        return common.to_tensor(rgb), mask_t, float(self.items[i][2])


def real_items(root: Path, rels: list[str]) -> list[tuple[Path, Path, float]]:
    out = []
    for r in rels:
        img = root / r
        mask = root / MASK_DIR / img.name
        if not mask.exists():
            alts = list((root / MASK_DIR).glob(img.stem + ".*"))
            if not alts:
                continue
            mask = alts[0]
        out.append((img, mask, 1.0))
    return out


def dice_bce(logits, target, weight):
    import torch
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean((1, 2, 3))
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return ((bce + dice) * weight).sum() / weight.sum().clamp_min(1e-6)


def evaluate(net, loader, device) -> dict:
    import torch

    net.eval()
    tp = fp = fn = tn = 0
    ious = []
    with torch.no_grad():
        for x, m, _ in loader:
            pred = (torch.sigmoid(net(x.to(device)).float()) >= 0.5).cpu().numpy()
            truth = m.numpy() > 0.5
            for pi, ti in zip(pred, truth):
                c = common.binary_counts(pi, ti)
                tp, fp, fn, tn = tp + c[0], fp + c[1], fn + c[2], tn + c[3]
                ious.append(common.prf(*c)["iou"] if c[0] + c[1] + c[2] else 1.0)
    out = common.prf(tp, fp, fn, tn)
    out.update(n=len(ious), mean_image_iou=float(np.mean(ious)) if ious else 0.0, threshold=0.5)
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, help="default data/raw/ozgenel_seg/extracted")
    ap.add_argument("--splits", type=Path, help="default data/splits/ozgenel_seg")
    ap.add_argument("--pseudo", choices=["none", "sdnet2018", "ozgenel_cls"], default="none")
    ap.add_argument("--pseudo-root", type=Path)
    ap.add_argument("--pseudo-splits", type=Path)
    ap.add_argument("--pseudo-limit", type=int, default=2000, help="cracked patches taken from the pseudo set")
    ap.add_argument("--pseudo-weight", type=float, default=0.3)
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--weights", default="imagenet", choices=["imagenet", "none"])
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda (ROCm also answers to cuda)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=384, help="training crop side; images are resized to this short side")
    ap.add_argument("--limit", type=int, default=0, help="at most N items per split and source (smoke tests)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast on the accelerator")
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    import torch
    from torch.utils.data import DataLoader

    a = parse_args(argv)
    root = a.root or common.RAW / "ozgenel_seg" / "extracted"
    splits = a.splits or common.SPLITS / "ozgenel_seg"
    out = a.out or common.ROOT / "runs" / f"seg-{a.encoder}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    log = common.setup_log(out, "seg")
    common.seed_everything(a.seed)
    device = common.pick_device(a.device)
    info = common.device_info(device)
    log.info(f"device {info}")

    parts = {p: real_items(root, common.limit_items(common.read_split_list(splits, p), a.limit, a.seed))
             for p in ("train", "val", "test")}
    n_pseudo = 0
    if a.pseudo != "none":
        proot = a.pseudo_root or common.RAW / a.pseudo / "extracted"
        psplits = a.pseudo_splits or common.SPLITS / a.pseudo
        cracked = [r for r in common.read_split_list(psplits, "train") if label_of(a.pseudo, r) == 1]
        cracked = common.limit_items(cracked, min(a.pseudo_limit, a.limit or a.pseudo_limit), a.seed)
        parts["train"] += [(proot / r, None, a.pseudo_weight) for r in cracked]
        n_pseudo = len(cracked)
    log.info("items " + ", ".join(f"{k} {len(v)}" for k, v in parts.items()) + f" (pseudo in train: {n_pseudo})")
    sets = {p: SegSet(v, a.size, p == "train", a.seed) for p, v in parts.items()}
    loaders = {p: DataLoader(s, batch_size=a.batch, shuffle=p == "train", num_workers=a.workers)
               for p, s in sets.items()}

    net = build_model(a.encoder, a.weights == "imagenet").to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs * len(loaders["train"])))
    history, best = [], None
    with common.Timer() as t_all:
        for ep in range(a.epochs):
            net.train()
            tot, n = 0.0, 0
            for x, m, w in loaders["train"]:
                x, m, w = x.to(device), m.to(device), torch.as_tensor(w, dtype=torch.float32).to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=a.amp and device.type != "cpu"):
                    logits = net(x)
                loss = dice_bce(logits.float(), m, w)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss.detach()) * len(x)
                n += len(x)
            val = evaluate(net, loaders["val"], device)
            history.append({"epoch": ep + 1, "train_loss": tot / max(1, n), "val": val})
            log.info(f"epoch {ep + 1}/{a.epochs} loss {tot / max(1, n):.4f} val IoU {val['iou']:.3f} "
                     f"P {val['precision']:.3f} R {val['recall']:.3f}")
            if best is None or val["iou"] >= best[0]:
                best = (val["iou"], ep + 1, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[2])
    test = evaluate(net, loaders["test"], device)
    log.info(f"best epoch {best[1]} | test IoU {test['iou']:.3f} P {test['precision']:.3f} R {test['recall']:.3f}")

    meta = {"task": "segment", "arch": "unet", "encoder": a.encoder, "input_size": a.size,
            "input": {"channels": "RGB", "scale": "1/255", "mean": common.MEAN, "std": common.STD},
            "dataset": "ozgenel_seg", "pseudo": a.pseudo, "pseudo_items": n_pseudo, "pseudo_weight": a.pseudo_weight,
            "limit": a.limit, "epochs": a.epochs, "best_epoch": best[1], "pretrained": a.weights,
            "git": common.git_commit(), **info, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    ckpt, digest = common.save_checkpoint(out, best[2], meta)
    metrics = {"task": "segment", "smoke_test": bool(a.limit), "meta": meta, "checkpoint": ckpt.name,
               "sha256": digest, "seconds": round(t_all.seconds, 2), "history": history,
               "val": next(h["val"] for h in history if h["epoch"] == best[1]), "test": test,
               "command": "python -m holdpoint.vision.train_seg " + " ".join(argv if argv is not None else [])}
    common.write_json(out / "metrics.json", metrics)
    log.info(f"wrote {ckpt} sha256 {digest}")
    return metrics


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
