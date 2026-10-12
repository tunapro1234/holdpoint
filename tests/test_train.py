"""Training unit: both training scripts run on CPU with --limit 16 --epochs 1 and leave a checkpoint, its SHA-256,
metrics JSON and a log; batch inference writes a cache the recorded backend can replay; no CUDA-only calls.

The fixture images are drawn here (grey noise with a dark polyline) only to exercise the code path; they say
nothing about model quality. When the real datasets are present in data/raw, the same smoke run also goes
through them.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from holdpoint.vision import common, infer, train_cls, train_seg

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ["--limit", "16", "--epochs", "1", "--device", "cpu", "--weights", "none", "--workers", "0"]


def _drawn(seed: int, crack: bool, size: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(150, 10, (size, size)), 0, 255).astype(np.uint8)
    mask = np.zeros_like(img)
    if crack:
        pts = rng.integers(0, size, (4, 2)).astype(np.int32)
        cv2.polylines(mask, [pts], False, 255, max(2, size // 40))
        img[mask > 0] = 60
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), mask


def _write_splits(d: Path, parts: dict[str, list[str]]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for k, v in parts.items():
        (d / f"{k}.txt").write_text("".join(f"{x}\n" for x in v))
    return d


@pytest.fixture(scope="module")
def cls_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("cls")
    rels = []
    for label in ("Positive", "Negative"):
        (root / label).mkdir()
        for i in range(24):
            img, _ = _drawn(i + (100 if label == "Positive" else 0), label == "Positive", 64)
            cv2.imwrite(str(root / label / f"{i:03d}.jpg"), img)
            rels.append(f"{label}/{i:03d}.jpg")
    parts = {"train": rels[0:16] + rels[24:40], "val": rels[16:20] + rels[40:44], "test": rels[20:24] + rels[44:48]}
    return root, _write_splits(root / "splits", parts)


@pytest.fixture(scope="module")
def seg_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("seg")
    (root / "rgb").mkdir()
    (root / "BW").mkdir()
    rels = []
    for i in range(24):
        img, mask = _drawn(200 + i, True, 160)
        cv2.imwrite(str(root / "rgb" / f"{i:03d}.jpg"), img)
        cv2.imwrite(str(root / "BW" / f"{i:03d}.png"), mask)
        rels.append(f"rgb/{i:03d}.jpg")
    return root, _write_splits(root / "splits", {"train": rels[:16], "val": rels[16:20], "test": rels[20:]})


def _check_outputs(out: Path, metrics: dict, task: str) -> None:
    ck = out / "model.pt"
    assert ck.is_file() and ck.stat().st_size > 0
    digest = (out / "model.pt.sha256").read_text().split()[0]
    assert digest == common.sha256_file(ck) == metrics["sha256"]
    on_disk = json.loads((out / "metrics.json").read_text())
    assert on_disk["task"] == task and on_disk["smoke_test"] is True
    assert on_disk["meta"]["device"] == "cpu"
    for part in ("val", "test"):
        for k in ("precision", "recall", "iou"):
            assert 0.0 <= on_disk[part][k] <= 1.0
    assert "epoch 1/1" in (out / "train.log").read_text()


def test_train_cls_smoke(cls_fixture, tmp_path):
    root, splits = cls_fixture
    out = tmp_path / "cls"
    m = train_cls.main(["--dataset", "ozgenel_cls", "--root", str(root), "--splits", str(splits), "--size", "64",
                        "--batch", "8", "--out", str(out), *SMOKE])
    _check_outputs(out, m, "classify")
    assert m["val"]["n"] == 8 and m["val"]["positives"] == 4


def test_train_seg_smoke(seg_fixture, tmp_path):
    root, splits = seg_fixture
    out = tmp_path / "seg"
    m = train_seg.main(["--root", str(root), "--splits", str(splits), "--size", "128", "--batch", "4",
                        "--out", str(out), *SMOKE])
    _check_outputs(out, m, "segment")
    assert m["meta"]["encoder"] == "resnet18"


def test_train_seg_with_pseudo_masks_and_infer(cls_fixture, seg_fixture, tmp_path):
    croot, csplits = cls_fixture
    sroot, ssplits = seg_fixture
    seg_out, cls_out = tmp_path / "seg", tmp_path / "cls"
    m = train_seg.main(["--root", str(sroot), "--splits", str(ssplits), "--pseudo", "ozgenel_cls",
                        "--pseudo-root", str(croot), "--pseudo-splits", str(csplits), "--size", "64",
                        "--batch", "8", "--out", str(seg_out), *SMOKE])
    assert m["meta"]["pseudo_items"] > 0
    train_cls.main(["--dataset", "ozgenel_cls", "--root", str(croot), "--splits", str(csplits), "--size", "64",
                    "--out", str(cls_out), *SMOKE])

    cache = tmp_path / "recorded.json"
    out = infer.main(["--seg", str(seg_out / "model.pt"), "--cls", str(cls_out / "model.pt"), "--images",
                      str(sroot), "--list", str(ssplits / "test.txt"), "--side", "160", "--device", "cpu",
                      "--out", str(cache)])
    assert out["summary"]["images"] == 4

    from holdpoint.backends.recorded import RecordedBackend
    rb = RecordedBackend(cache)
    rgb = common.load_rgb(sroot / "rgb" / "020.jpg")
    assert rb.segment(rgb).mask.shape == rgb.shape[:2]
    assert 0.0 <= rb.classify(rgb).p_crack <= 1.0
    assert "batch_infer" in json.loads(cache.read_text())["provenance"]["backend"]


def test_cli_entry_point(cls_fixture, tmp_path):
    root, splits = cls_fixture
    out = tmp_path / "cli"
    r = subprocess.run([sys.executable, "-m", "holdpoint.vision.train_cls", "--dataset", "ozgenel_cls", "--root",
                        str(root), "--splits", str(splits), "--size", "64", "--out", str(out), *SMOKE],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert (out / "metrics.json").is_file()


@pytest.mark.skipif(not (common.RAW / "ozgenel_seg" / "extracted").is_dir()
                    or not (common.SPLITS / "ozgenel_seg").is_dir(), reason="Özgenel segmentation set not downloaded")
def test_train_seg_smoke_on_real_data(tmp_path):
    m = train_seg.main(["--size", "128", "--batch", "4", "--out", str(tmp_path / "real"), *SMOKE])
    assert m["meta"]["dataset"] == "ozgenel_seg" and m["val"]["n"] == 16


@pytest.mark.skipif(not (common.RAW / "ozgenel_cls" / "extracted").is_dir()
                    or not (common.SPLITS / "ozgenel_cls").is_dir(), reason="Özgenel classification set not downloaded")
def test_train_cls_smoke_on_real_data(tmp_path):
    m = train_cls.main(["--dataset", "ozgenel_cls", "--size", "96", "--batch", "8",
                        "--out", str(tmp_path / "real"), *SMOKE])
    assert m["val"]["n"] == 16 and m["val"]["positives"] == 8


CUDA_ONLY = re.compile(r"\.cuda\(|nvidia-smi|bitsandbytes|cudnn|pynvml|torch\.cuda\.(?!is_available|get_device_name)")


def test_no_cuda_only_calls():
    files = sorted((ROOT / "holdpoint" / "vision").glob("*.py")) + sorted((ROOT / "notebooks").glob("*.ipynb"))
    hits = [f"{f.name}:{i}" for f in files for i, line in enumerate(f.read_text().splitlines(), 1)
            if CUDA_ONLY.search(line)]
    assert not hits, hits
