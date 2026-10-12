"""Crack / no-crack patch classifier (torchvision ResNet-18 or ConvNeXt-Tiny).

    python -m holdpoint.vision.train_cls --dataset sdnet2018 --epochs 5 --device auto --out runs/cls-sdnet
    python -m holdpoint.vision.train_cls --dataset ozgenel_cls --limit 16 --epochs 1 --weights none   # CPU smoke test

Reads the committed split files (data/splits/<dataset>/) and images under data/raw/<dataset>/extracted.
Writes to --out: model.pt (state_dict + meta), model.pt.sha256, metrics.json (precision/recall/F1 on val and
test), train.log. The same code runs on CPU, on ROCm (AMD Instinct) and on any other torch device.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from . import common

ARCHS = ("resnet18", "convnext_tiny")


def label_of(dataset: str, rel: str) -> int:
    folder = Path(rel).parts[-2]
    if dataset == "ozgenel_cls":
        return 1 if folder.lower() == "positive" else 0
    if dataset == "sdnet2018":
        return 1 if folder.upper().startswith("C") else 0  # CD/CP/CW cracked, UD/UP/UW uncracked
    raise KeyError(dataset)


def build_model(arch: str, pretrained: bool):
    import torch.nn as nn
    import torchvision

    if arch == "resnet18":
        net = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, 2)
    elif arch == "convnext_tiny":
        net = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
        net.classifier[2] = nn.Linear(net.classifier[2].in_features, 2)
    else:
        raise ValueError(arch)
    return net


class PatchSet:
    """(rel path, label) list -> normalized tensors; light flips/rotations when train=True."""

    def __init__(self, root: Path, items: list[tuple[str, int]], size: int, train: bool, seed: int = 0):
        self.root, self.items, self.size, self.train = Path(root), items, size, train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        rel, y = self.items[i]
        rgb = common.load_rgb(self.root / rel)
        rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.train:
            if self.rng.random() < 0.5:
                rgb = rgb[:, ::-1]
            if self.rng.random() < 0.5:
                rgb = rgb[::-1]
            rgb = np.rot90(rgb, int(self.rng.integers(4)))
        return common.to_tensor(rgb), y


def evaluate(net, loader, device) -> dict:
    import torch

    net.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(net(x.to(device)).float(), 1)[:, 1].cpu().numpy()
            probs.append(p)
            ys.append(np.asarray(y))
    if not probs:
        return {"n": 0}
    p, y = np.concatenate(probs), np.concatenate(ys)
    out = common.prf(*common.binary_counts(p >= 0.5, y == 1))
    out.update(n=int(len(y)), positives=int(y.sum()), threshold=0.5)
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="sdnet2018", choices=["sdnet2018", "ozgenel_cls"])
    ap.add_argument("--root", type=Path, help="image root (default data/raw/<dataset>/extracted)")
    ap.add_argument("--splits", type=Path, help="split folder (default data/splits/<dataset>)")
    ap.add_argument("--arch", default="resnet18", choices=ARCHS)
    ap.add_argument("--weights", default="imagenet", choices=["imagenet", "none"])
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda (ROCm also answers to cuda)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--limit", type=int, default=0, help="at most N items per split (smoke tests)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast on the accelerator")
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    import torch
    from torch.utils.data import DataLoader

    a = parse_args(argv)
    root = a.root or common.RAW / a.dataset / "extracted"
    splits = a.splits or common.SPLITS / a.dataset
    out = a.out or common.ROOT / "runs" / f"cls-{a.dataset}-{a.arch}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    log = common.setup_log(out, "cls")
    common.seed_everything(a.seed)
    device = common.pick_device(a.device)
    info = common.device_info(device)
    log.info(f"device {info}")

    def items(part):
        rels = common.read_split_list(splits, part)
        rels = common.limit_items(rels, a.limit, a.seed, key=lambda r: label_of(a.dataset, r))
        return [(r, label_of(a.dataset, r)) for r in rels]

    parts = {p: items(p) for p in ("train", "val", "test")}
    log.info("items " + ", ".join(f"{k} {len(v)}" for k, v in parts.items()))
    loaders = {p: DataLoader(PatchSet(root, v, a.size, p == "train", a.seed), batch_size=a.batch,
                             shuffle=p == "train", num_workers=a.workers)
               for p, v in parts.items()}

    net = build_model(a.arch, a.weights == "imagenet").to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, a.epochs * len(loaders["train"])))
    loss_fn = torch.nn.CrossEntropyLoss()
    history = []
    with common.Timer() as t_all:
        for ep in range(a.epochs):
            net.train()
            tot, n = 0.0, 0
            for x, y in loaders["train"]:
                x, y = x.to(device), torch.as_tensor(y).to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=a.amp and device.type != "cpu"):
                    loss = loss_fn(net(x).float(), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss.detach()) * len(y)
                n += len(y)
            val = evaluate(net, loaders["val"], device)
            history.append({"epoch": ep + 1, "train_loss": tot / max(1, n), "val": val})
            log.info(f"epoch {ep + 1}/{a.epochs} loss {tot / max(1, n):.4f} val P {val.get('precision', 0):.3f} "
                     f"R {val.get('recall', 0):.3f} F1 {val.get('f1', 0):.3f}")
    test = evaluate(net, loaders["test"], device)
    log.info(f"test P {test.get('precision', 0):.3f} R {test.get('recall', 0):.3f} F1 {test.get('f1', 0):.3f}")

    meta = {"task": "classify", "arch": a.arch, "classes": ["no_crack", "crack"], "input_size": a.size,
            "input": {"channels": "RGB", "scale": "1/255", "mean": common.MEAN, "std": common.STD},
            "dataset": a.dataset, "splits": Path(splits).name, "limit": a.limit, "epochs": a.epochs,
            "pretrained": a.weights, "git": common.git_commit(), **info,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    ckpt, digest = common.save_checkpoint(out, net.state_dict(), meta)
    metrics = {"task": "classify", "smoke_test": bool(a.limit), "meta": meta, "checkpoint": ckpt.name,
               "sha256": digest, "seconds": round(t_all.seconds, 2), "history": history,
               "val": history[-1]["val"] if history else {}, "test": test,
               "command": "python -m holdpoint.vision.train_cls " + " ".join(argv if argv is not None else [])}
    common.write_json(out / "metrics.json", metrics)
    log.info(f"wrote {ckpt} sha256 {digest}")
    return metrics


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
