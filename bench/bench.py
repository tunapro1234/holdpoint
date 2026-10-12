"""Throughput, latency and cost per image for one backend on one image set.

    python bench/bench.py --backend unet --checkpoint runs/seg/model.pt --images examples --device auto
    python bench/bench.py --backend classifier --checkpoint runs/cls/model.pt --images examples \
        --price-per-hour 1.99 --price-source "<provider price page, date read>"
    python bench/bench.py --backend cpu --images examples                      # classical CPU baseline

Backends:
  unet        segmentation checkpoint from holdpoint.vision.train_seg, batched on --device
  classifier  patch classifier checkpoint from holdpoint.vision.train_cls, batched on --device
  cpu         holdpoint.backends.cpu.CpuBackend.segment, one call per image (the app's live CPU path)
Without --checkpoint the torch backends use untrained weights; speed does not depend on the weights, and the
file says so ("weights": "untrained").

For each batch size (default 1, 8, 32) the images are cycled to fill the batch, --warmup batches are run and
discarded, then --iters batches are timed. Timed: host-to-device copy, forward pass, device-to-host copy of the
output (which also waits for the device). Image decoding and resizing are not timed.

Recorded with every run: device name, torch and ROCm (HIP) versions, `amd-smi` output when the tool exists
("not available" otherwise, nothing is filled in), git commit, the exact command. Cost per 1000 images is only
computed from a price you pass (--price-per-hour, no default) together with its source (--price-source).
Output: results/bench-<device>-<UTC time>.json (or --out-dir / --out); the backend is inside the file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from holdpoint.vision import common  # noqa: E402

SCHEMA = "holdpoint-bench/1"
BACKENDS = ("unet", "classifier", "cpu")
NOT_AVAILABLE = "not available"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


# ---- inputs ------------------------------------------------------------------

def find_images(images: Path, limit: int = 0) -> list[Path]:
    paths = sorted(p for p in Path(images).rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    return paths[:limit] if limit else paths


def square(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize the short side to size and center-crop, so every image in a batch has the same shape."""
    h, w = rgb.shape[:2]
    s = size / min(h, w)
    rgb = cv2.resize(rgb, (max(size, round(w * s)), max(size, round(h * s))),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    h, w = rgb.shape[:2]
    y0, x0 = (h - size) // 2, (w - size) // 2
    return np.ascontiguousarray(rgb[y0:y0 + size, x0:x0 + size])


def cycled(n_images: int, batch: int, k: int) -> list[int]:
    """Indices of the k-th batch when the image list is repeated to fill batches."""
    return [(k * batch + j) % n_images for j in range(batch)]


# ---- environment -------------------------------------------------------------

def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def amd_smi(timeout: int = 60):
    """Raw amd-smi output, or "not available" when the tool is not installed. Never filled in by hand."""
    exe = shutil.which("amd-smi")
    if not exe:
        return NOT_AVAILABLE
    out = {"path": exe}
    for key, args in (("version", ["version"]), ("static", ["static", "--asic", "--vram", "--json"]),
                      ("metric", ["metric", "--json"])):
        try:
            r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError) as e:
            out[key] = {"error": str(e)}
            continue
        if r.returncode != 0:
            out[key] = {"error": f"exit {r.returncode}", "stderr": r.stderr[-2000:]}
            continue
        try:
            out[key] = json.loads(r.stdout)
        except json.JSONDecodeError:
            out[key] = r.stdout.strip()
    return out


def device_block(device) -> dict:
    import torch

    hip = getattr(torch.version, "hip", None)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {"type": device.type, "name": gpu or cpu_name(), "torch": torch.__version__,
            "rocm": hip or NOT_AVAILABLE, "python": platform.python_version(), "host_cpu": cpu_name(),
            "torch_threads": torch.get_num_threads()}


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:40] or "device"


def git_state() -> dict:
    commit = common.git_commit()
    try:
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return {"commit": commit, "dirty": dirty}


# ---- runners -----------------------------------------------------------------

class TorchRunner:
    """A checkpoint (or untrained model) of a training script, fed whole batches."""

    def __init__(self, backend: str, checkpoint: Path | None, device, encoder: str, arch: str, amp: bool):
        import torch

        self.device, self.amp = device, amp
        if checkpoint:
            from holdpoint.vision.infer import load_checkpoint
            self.net, meta = load_checkpoint(checkpoint, device)
            want = "segment" if backend == "unet" else "classify"
            if meta.get("task") != want:
                raise SystemExit(f"{checkpoint} is a {meta.get('task')!r} checkpoint, backend {backend} needs {want!r}")
            name = meta.get("encoder") if backend == "unet" else meta.get("arch")
            self.model = {"name": f"{'unet-' if backend == 'unet' else ''}{name}", "weights": "trained",
                          "checkpoint": checkpoint.name, "sha256": common.sha256_file(checkpoint)}
        else:
            if backend == "unet":
                from holdpoint.vision.train_seg import build_model
                net, name = build_model(encoder, pretrained=False), f"unet-{encoder}"
            else:
                from holdpoint.vision.train_cls import build_model
                net, name = build_model(arch, pretrained=False), arch
            self.net = net.to(device).eval()
            self.model = {"name": name, "weights": "untrained", "checkpoint": None, "sha256": None,
                          "note": "random weights: valid for speed only, not for predictions"}
        self.model["parameters"] = int(sum(p.numel() for p in self.net.parameters()))
        self.torch = torch

    def prepare(self, rgbs: list[np.ndarray]):
        return [common.to_tensor(r) for r in rgbs]

    def run(self, items: list) -> None:
        torch = self.torch
        x = torch.stack(items).to(self.device)
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16,
                                             enabled=self.amp and self.device.type != "cpu"):
            y = self.net(x)
        y.float().cpu()  # copy back: the timing includes waiting for the device


class CpuBackendRunner:
    """The app's live CPU path, one segment() call per image."""

    def __init__(self, checkpoint: Path | None):
        from holdpoint.backends.cpu import CpuBackend

        self.backend = CpuBackend(weights=checkpoint)
        self.backend.weights = checkpoint  # no silent pick-up of HOLDPOINT_UNET_WEIGHTS
        prov = self.backend.provenance()
        self.model = {"name": prov["model"], "weights": "trained" if checkpoint else "classical baseline",
                      "checkpoint": checkpoint.name if checkpoint else None,
                      "sha256": common.sha256_file(checkpoint) if checkpoint else None}

    def prepare(self, rgbs: list[np.ndarray]):
        return rgbs

    def run(self, items: list) -> None:
        for img in items:
            self.backend.segment(img)


def time_batches(runner, items: list, batch: int, warmup: int, iters: int) -> dict:
    for k in range(warmup):
        runner.run([items[i] for i in cycled(len(items), batch, k)])
    lat = []
    for k in range(iters):
        sel = [items[i] for i in cycled(len(items), batch, warmup + k)]
        t0 = time.perf_counter()
        runner.run(sel)
        lat.append(time.perf_counter() - t0)
    lat_ms = np.asarray(lat) * 1000.0
    total = float(np.sum(lat))
    return {"batch": batch, "iterations": iters, "warmup": warmup, "images_timed": batch * iters,
            "img_per_s": round(batch * iters / total, 3) if total > 0 else None,
            "p50_ms": round(float(np.percentile(lat_ms, 50)), 3),
            "p95_ms": round(float(np.percentile(lat_ms, 95)), 3),
            "p50_ms_per_image": round(float(np.percentile(lat_ms, 50)) / batch, 3),
            "mean_ms": round(float(lat_ms.mean()), 3)}


def cost_block(price_per_hour: float | None, source: str | None, rows: list[dict]) -> dict:
    """$/1000 images = price per hour / images per hour * 1000, per batch size. No price, no number."""
    if price_per_hour is None:
        return {"price_per_hour_usd": None, "source": None, "usd_per_1000_images": None,
                "note": "no price given (--price-per-hour and --price-source)"}
    for r in rows:
        r["usd_per_1000_images"] = (round(price_per_hour / (r["img_per_s"] * 3600) * 1000, 6)
                                    if r.get("img_per_s") else None)
    best = max(rows, key=lambda r: r.get("img_per_s") or 0)
    return {"price_per_hour_usd": price_per_hour, "source": source,
            "usd_per_1000_images": best.get("usd_per_1000_images"), "at_batch": best["batch"],
            "formula": "price_per_hour_usd / (img_per_s * 3600) * 1000"}


# ---- main --------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=BACKENDS, required=True)
    ap.add_argument("--images", type=Path, required=True, help="folder of images (searched recursively)")
    ap.add_argument("--limit", type=int, default=0, help="use at most N images")
    ap.add_argument("--checkpoint", type=Path, help="model.pt from the training scripts")
    ap.add_argument("--encoder", default="resnet18", help="UNet encoder when no checkpoint is given")
    ap.add_argument("--arch", default="resnet18", help="classifier architecture when no checkpoint is given")
    ap.add_argument("--size", type=int, default=None, help="square input side (default 512 for unet/cpu, "
                                                           "the checkpoint's input size or 224 for classifier)")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda (ROCm also answers to cuda)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast on the accelerator")
    ap.add_argument("--price-per-hour", type=float, default=None,
                    help="USD per hour for the device; no default, needs --price-source")
    ap.add_argument("--price-source", default=None, help="where the price comes from (URL or document, date)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=None, help="exact output file (overrides --out-dir naming)")
    a = ap.parse_args(argv)
    if a.price_per_hour is not None and not (a.price_source or "").strip():
        ap.error("--price-per-hour needs --price-source")
    if a.price_source and a.price_per_hour is None:
        ap.error("--price-source given without --price-per-hour")
    if a.price_per_hour is not None and a.price_per_hour <= 0:
        ap.error("--price-per-hour must be positive")
    if a.backend == "cpu" and a.device not in ("auto", "cpu"):
        ap.error("the cpu backend runs on cpu only")
    try:
        a.batch_sizes = [int(b) for b in a.batches.split(",") if b.strip()]
    except ValueError:
        ap.error("--batches is a comma-separated list of integers")
    if not a.batch_sizes or min(a.batch_sizes) < 1 or a.warmup < 0 or a.iters < 1:
        ap.error("batch sizes and --iters must be >= 1, --warmup >= 0")
    return a


def main(argv=None) -> dict:
    import torch

    a = parse_args(argv)
    argv = list(argv) if argv is not None else sys.argv[1:]
    if a.threads:
        torch.set_num_threads(a.threads)
    device = common.pick_device("cpu" if a.backend == "cpu" else a.device)

    paths = find_images(a.images, a.limit)
    if not paths:
        raise SystemExit(f"no images under {a.images}")
    if a.backend == "cpu":
        runner = CpuBackendRunner(a.checkpoint)
    else:
        runner = TorchRunner(a.backend, a.checkpoint, device, a.encoder, a.arch, a.amp)
    size = a.size
    if size is None:
        size = 224 if a.backend == "classifier" else 512
        if a.backend == "classifier" and a.checkpoint:
            ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
            size = int(ck.get("meta", {}).get("input_size", 224))
    rgbs = [square(common.load_rgb(p, min_side=size), size) for p in paths]
    items = runner.prepare(rgbs)

    rows = [time_batches(runner, items, b, a.warmup, a.iters) for b in a.batch_sizes]
    dev = device_block(device)
    smi = amd_smi()
    now = datetime.now(timezone.utc)
    stamp = now.isoformat(timespec="seconds")
    command = "python bench/bench.py " + " ".join(argv)
    label = f"computed on {dev['name']}, {stamp}, {command}"
    cost = cost_block(a.price_per_hour, a.price_source, rows)
    images_digest = hashlib.sha256("".join(common.sha256_file(p) for p in paths).encode()).hexdigest()

    result = {
        "schema": SCHEMA,
        "provenance": {"backend": f"bench:{a.backend}", "device": dev["name"], "model": runner.model["name"],
                       "timestamp": stamp, "command": command, "git": git_state()["commit"], "label": label},
        "device": dev,
        "amd_smi": smi,
        "git": git_state(),
        "backend": a.backend,
        "model": runner.model,
        "images": {"folder": _rel(a.images), "count": len(paths), "input_side": size,
                   "preprocessing": "short side resized to input_side, center crop (not timed)",
                   "sha256_of_file_hashes": images_digest},
        "settings": {"batches": a.batch_sizes, "warmup": a.warmup, "iters": a.iters,
                     "precision": "bf16 autocast" if a.amp and device.type != "cpu" else "fp32",
                     "batching": "sequential segment() calls" if a.backend == "cpu" else "one forward per batch",
                     "timed": "host-to-device copy, forward, device-to-host copy of the output"},
        "rows": rows,
        "cost": cost,
    }
    name = slug(dev["name"]) if device.type != "cpu" else "cpu"
    out = a.out or a.out_dir / f"bench-{name}-{now:%Y%m%dT%H%M%SZ}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    common.write_json(out, result)
    for r in rows:
        print(f"batch {r['batch']:>3}: {r['img_per_s']} img/s, p50 {r['p50_ms']} ms, p95 {r['p95_ms']} ms")
    print(f"wrote {out}")
    result["path"] = str(out)
    return result


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return Path(p).name


if __name__ == "__main__":
    main()
