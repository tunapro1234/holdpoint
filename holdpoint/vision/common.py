"""Shared pieces of the training and inference scripts: device choice, data lists, checkpoints, logs.

Device-agnostic by rule: the only accelerator check is torch.cuda.is_available(), which is also what ROCm builds
of PyTorch answer on AMD Instinct GPUs. No CUDA-only calls, no vendor tools.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"

# Images go into every model as RGB float in [0, 1], then ImageNet mean/std. Stored in each checkpoint.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def pick_device(name: str = "auto"):
    import torch

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def device_info(device) -> dict:
    import torch

    info = {"device": str(device), "torch": str(torch.__version__), "hip": getattr(torch.version, "hip", None),
            "python": platform.python_version(), "host_cpu": platform.processor() or platform.machine()}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_split_list(splits_dir: Path, part: str) -> list[str]:
    p = Path(splits_dir) / f"{part}.txt"
    return [x for x in p.read_text().splitlines() if x.strip()]


def limit_items(items: list[str], limit: int | None, seed: int, key=None) -> list[str]:
    """Deterministic subset of at most `limit` items, interleaving groups (labels) so both classes appear."""
    if not limit or limit >= len(items):
        return items
    if key is None:
        return sorted(random.Random(seed).sample(items, limit))
    groups: dict = {}
    for it in sorted(items):
        groups.setdefault(key(it), []).append(it)
    for g in groups.values():
        random.Random(f"{seed}:{len(g)}").shuffle(g)
    out, i = [], 0
    while len(out) < limit:
        for g in groups.values():
            if i < len(g) and len(out) < limit:
                out.append(g[i])
        i += 1
    return sorted(out)


def load_rgb(path: Path, min_side: int | None = None) -> np.ndarray:
    """Read an image as RGB uint8. With min_side, JPEG decoding is reduced (1/2, 1/4, 1/8) while the short side
    stays >= min_side, which makes 3264 x 2448 photos cheap to load."""
    flag = cv2.IMREAD_COLOR
    if min_side:
        h, w = _jpeg_size(path) or (None, None)
        if h:
            for f, fl in ((8, cv2.IMREAD_REDUCED_COLOR_8), (4, cv2.IMREAD_REDUCED_COLOR_4),
                          (2, cv2.IMREAD_REDUCED_COLOR_2)):
                if min(h, w) // f >= min_side:
                    flag = fl
                    break
    bgr = cv2.imread(str(path), flag)
    if bgr is None:
        raise ValueError(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Binary mask (uint8 0/1) resized to shape (h, w). JPEG masks are thresholded at 127."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"cannot read {path}")
    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def _jpeg_size(path: Path) -> tuple[int, int] | None:
    if Path(path).suffix.lower() not in {".jpg", ".jpeg"}:
        return None
    with Path(path).open("rb") as f:
        data = f.read(1 << 16)
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            return None
        marker, seg = data[i + 1], int.from_bytes(data[i + 2:i + 4], "big")
        if marker in (0xC0, 0xC1, 0xC2):
            return int.from_bytes(data[i + 5:i + 7], "big"), int.from_bytes(data[i + 7:i + 9], "big")
        i += 2 + seg
    return None


def to_tensor(rgb: np.ndarray):
    """HWC uint8 RGB -> CHW float tensor, ImageNet-normalized."""
    import torch

    x = rgb.astype(np.float32) / 255.0
    x = (x - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def setup_log(out_dir: Path, name: str) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"holdpoint.{name}.{out_dir}")
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    for h in (logging.FileHandler(out_dir / "train.log"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def load_checkpoint_dict(path) -> dict:
    """torch.load with weights_only=True (no arbitrary code from the file). Checkpoints written before this fix
    carry torch.__version__ as a TorchVersion (a str subclass) in their meta, so that one class is allowed."""
    import torch
    from torch.torch_version import TorchVersion

    with torch.serialization.safe_globals([TorchVersion]):
        return torch.load(path, map_location="cpu", weights_only=True)


def save_checkpoint(out_dir: Path, state_dict: dict, meta: dict) -> tuple[Path, str]:
    import torch

    path = out_dir / "model.pt"
    torch.save({"state_dict": state_dict, "meta": meta}, path)
    digest = sha256_file(path)
    (out_dir / "model.pt.sha256").write_text(f"{digest}  model.pt\n")
    return path, digest


def write_json(path: Path, obj: dict) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def binary_counts(pred: np.ndarray, truth: np.ndarray) -> tuple[int, int, int, int]:
    pred, truth = pred.astype(bool), truth.astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    return tp, fp, fn, tn


def prf(tp: int, fp: int, fn: int, tn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "accuracy": (tp + tn) / max(1, tp + fp + fn + tn), "iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
