"""Benchmark script: runs on CPU with 8 images, writes the full JSON schema, never fills in amd-smi or a price.

The 8 images are drawn here (grey noise with a dark line) only to exercise the code path; timing numbers from this
test say nothing about real hardware. Outputs go to tmp_path, never to results/.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from bench import bench
from tools import results_table

ROOT = Path(__file__).resolve().parents[1]
FAST = ["--batches", "1,8,32", "--warmup", "1", "--iters", "2", "--device", "cpu"]

TOP = {"schema", "provenance", "device", "amd_smi", "git", "backend", "model", "images", "settings", "rows", "cost"}
PROV = {"backend", "device", "model", "timestamp", "command", "git", "label"}
DEVICE = {"type", "name", "torch", "rocm", "python", "host_cpu", "torch_threads"}
ROW = {"batch", "iterations", "warmup", "images_timed", "img_per_s", "p50_ms", "p95_ms", "p50_ms_per_image", "mean_ms"}
COST = {"price_per_hour_usd", "source", "usd_per_1000_images"}


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    d = tmp_path_factory.mktemp("bench-images")
    for i in range(8):
        rng = np.random.default_rng(i)
        img = np.clip(rng.normal(150, 10, (96, 128)), 0, 255).astype(np.uint8)
        cv2.line(img, (5, 10 + i * 8), (120, 80 - i * 4), 60, 3)
        cv2.imwrite(str(d / f"{i:02d}.png"), img)
    return d


@pytest.fixture
def no_amd_smi(monkeypatch):
    real = bench.shutil.which
    monkeypatch.setattr(bench.shutil, "which", lambda name, *a, **k: None if name == "amd-smi" else real(name, *a, **k))


def _check_schema(r: dict, backend: str) -> None:
    assert TOP <= r.keys()
    assert r["schema"] == bench.SCHEMA and r["backend"] == backend
    assert PROV <= r["provenance"].keys()
    assert r["provenance"]["label"].startswith("computed on ")
    assert DEVICE <= r["device"].keys() and r["device"]["type"] == "cpu" and r["device"]["name"]
    assert r["git"]["commit"] == r["provenance"]["git"]
    assert r["images"]["count"] == 8 and len(r["images"]["sha256_of_file_hashes"]) == 64
    assert [row["batch"] for row in r["rows"]] == [1, 8, 32]
    for row in r["rows"]:
        assert ROW <= row.keys()
        assert row["img_per_s"] > 0 and 0 < row["p50_ms"] <= row["p95_ms"]
        assert row["images_timed"] == row["batch"] * 2
    assert COST <= r["cost"].keys()


def test_cpu_backend_8_images(images, tmp_path, no_amd_smi):
    r = bench.main(["--backend", "cpu", "--images", str(images), "--out-dir", str(tmp_path), *FAST])
    _check_schema(r, "cpu")
    assert r["amd_smi"] == "not available"
    assert r["model"]["weights"] == "classical baseline"
    path = Path(r["path"])
    assert path.parent == tmp_path and re.fullmatch(r"bench-cpu-\d{8}T\d{6}Z\.json", path.name)
    on_disk = json.loads(path.read_text())
    assert on_disk["rows"] == r["rows"] and on_disk["amd_smi"] == "not available"
    # no price given: no cost number
    assert on_disk["cost"]["usd_per_1000_images"] is None and on_disk["cost"]["source"] is None
    assert all("usd_per_1000_images" not in row for row in on_disk["rows"])


@pytest.mark.parametrize("backend", ["unet", "classifier"])
def test_torch_backends_untrained_on_cpu(images, tmp_path, no_amd_smi, backend):
    r = bench.main(["--backend", backend, "--images", str(images), "--size", "64", "--out-dir", str(tmp_path), *FAST])
    _check_schema(r, backend)
    assert r["amd_smi"] == "not available"
    assert r["device"]["rocm"] == "not available"  # CPU wheel: torch.version.hip is None
    assert r["model"]["weights"] == "untrained" and r["model"]["parameters"] > 0
    assert r["settings"]["precision"] == "fp32"


def test_trained_checkpoint(images, tmp_path, no_amd_smi):
    from holdpoint.vision import common, train_seg

    ck = tmp_path / "seg"
    ck.mkdir()
    net = train_seg.build_model("resnet18", pretrained=False)
    common.save_checkpoint(ck, net.state_dict(), {"task": "segment", "encoder": "resnet18"})
    r = bench.main(["--backend", "unet", "--checkpoint", str(ck / "model.pt"), "--images", str(images),
                    "--size", "64", "--out", str(tmp_path / "x.json"), *FAST])
    assert r["model"]["weights"] == "trained" and r["model"]["sha256"] == common.sha256_file(ck / "model.pt")
    with pytest.raises(SystemExit):  # a segmentation checkpoint is not a classifier
        bench.main(["--backend", "classifier", "--checkpoint", str(ck / "model.pt"), "--images", str(images),
                    "--out", str(tmp_path / "y.json"), *FAST])


def test_price_needs_a_source(images, tmp_path):
    with pytest.raises(SystemExit):
        bench.parse_args(["--backend", "cpu", "--images", str(images), "--price-per-hour", "2.0"])
    with pytest.raises(SystemExit):
        bench.parse_args(["--backend", "cpu", "--images", str(images), "--price-source", "somewhere"])


def test_cost_per_1000_images(images, tmp_path, no_amd_smi):
    # the price here is a test input, not a quote
    r = bench.main(["--backend", "cpu", "--images", str(images), "--out-dir", str(tmp_path),
                    "--price-per-hour", "3.6", "--price-source", "test input, not a real price", *FAST])
    cost = r["cost"]
    assert cost["price_per_hour_usd"] == 3.6 and cost["source"] == "test input, not a real price"
    for row in r["rows"]:
        assert row["usd_per_1000_images"] == pytest.approx(3.6 / (row["img_per_s"] * 3600) * 1000, rel=1e-3)
    best = max(r["rows"], key=lambda row: row["img_per_s"])
    assert cost["usd_per_1000_images"] == best["usd_per_1000_images"] and cost["at_batch"] == best["batch"]


def test_amd_smi_output_is_recorded_verbatim(tmp_path, monkeypatch):
    fake = tmp_path / "amd-smi"  # stand-in executable for the test; prints fixed JSON
    fake.write_text("#!/bin/sh\nif [ \"$1\" = version ]; then echo 'AMDSMI Tool: test'; "
                    "else echo '{\"fixture\": \"'$1'\"}'; fi\n")
    fake.chmod(0o755)
    monkeypatch.setattr(bench.shutil, "which", lambda name, *a, **k: str(fake) if name == "amd-smi" else None)
    out = bench.amd_smi()
    assert out["version"] == "AMDSMI Tool: test"
    assert out["static"] == {"fixture": "static"} and out["metric"] == {"fixture": "metric"}


def test_cli_and_results_table(images, tmp_path):
    r = subprocess.run([sys.executable, "bench/bench.py", "--backend", "cpu", "--images", str(images),
                        "--out-dir", str(tmp_path), *FAST], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    files = list(tmp_path.glob("bench-*.json"))
    assert len(files) == 1
    table = results_table.render(tmp_path)
    data = json.loads(files[0].read_text())
    rows = [ln for ln in table.splitlines() if ln.startswith(f"| {files[0].name} |")]
    assert len(rows) == 3 and all(data["provenance"]["device"] in ln for ln in rows)


CUDA_ONLY = re.compile(r"\.cuda\(|nvidia-smi|bitsandbytes|cudnn|pynvml|torch\.cuda\.(?!is_available|get_device_name)")


def test_no_cuda_only_calls():
    files = [ROOT / "bench" / "bench.py", ROOT / "notebooks" / "bench.ipynb"]
    hits = [f"{f.name}:{i}" for f in files if f.exists() for i, line in enumerate(f.read_text().splitlines(), 1)
            if CUDA_ONLY.search(line)]
    assert not hits, hits
