"""Data unit: deterministic disjoint splits, pseudo masks on a small drawn fixture, dataset attribution."""
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from data import download, split
from holdpoint.vision.pseudo_mask import iou, pseudo_mask

ROOT = Path(__file__).resolve().parents[1]

NAMES = [f"{folder}/{i:05d}.jpg" for folder in ("Positive", "Negative") for i in range(200)]


def test_split_is_deterministic_and_order_independent():
    shuffled = NAMES[:]
    random.Random(7).shuffle(shuffled)
    a = split.make_split(NAMES, seed=1)
    b = split.make_split(shuffled, seed=1)
    assert a == b
    assert split.make_split(NAMES, seed=2) != a


def test_split_is_disjoint_and_complete():
    s = split.make_split(NAMES, group=lambda r: r.split("/")[0])
    sets = {k: set(v) for k, v in s.items()}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
    assert sets["train"] | sets["val"] | sets["test"] == set(NAMES)
    assert len(s["train"]) == 280 and len(s["val"]) == 60 and len(s["test"]) == 60
    # stratified: both labels present in every part
    for part in s.values():
        assert {r.split("/")[0] for r in part} == {"Positive", "Negative"}


def test_split_files_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(split, "SPLITS", tmp_path)
    s = split.make_split(NAMES)
    split.write_split("fixture", s, {"seed": split.SEED})
    assert split.read_split("fixture", tmp_path) == s


def test_committed_split_files_are_disjoint():
    found = sorted(p.parent for p in (ROOT / "data" / "splits").glob("*/split.json"))
    if not found:
        pytest.skip("no committed split files yet")
    for d in found:
        s = split.read_split(d.name, d.parent)
        sets = [set(s[n]) for n in split.NAMES]
        assert sum(len(x) for x in sets) == len(set().union(*sets)), d.name
        assert all(len(x) == len(s[n]) for x, n in zip(sets, split.NAMES)), d.name


def _fixture(crack: bool, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """128x128 concrete-like gray noise; with crack=True a dark 2-3 px polyline is drawn (the expected mask)."""
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(150, 8, (128, 128)), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    truth = np.zeros_like(img)
    if crack:
        pts = np.array([[10, 20], [40, 45], [62, 50], [90, 85], [118, 110]], np.int32)
        cv2.polylines(truth, [pts], False, 255, 3)
        img[truth > 0] = 70
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), truth


def test_pseudo_mask_finds_drawn_crack():
    img, truth = _fixture(crack=True)
    m = pseudo_mask(img)
    assert m.dtype == np.uint8 and set(np.unique(m)) <= {0, 255}
    assert (m > 0).sum() > 0
    assert iou(m, truth) > 0.4


def test_pseudo_mask_empty_on_plain_surface():
    img, _ = _fixture(crack=False, seed=3)
    assert (pseudo_mask(img) > 0).sum() < 0.002 * img.shape[0] * img.shape[1]


def test_attribution_names_both_datasets_and_licenses():
    text = (ROOT / "ATTRIBUTION.md").read_text()
    for key in ("sdnet2018", "ozgenel_seg"):
        meta = download.DATASETS[key]
        assert meta["title"].split(":")[0] in text
        assert meta["doi"] in text
    assert text.count("CC BY 4.0") >= 2
    assert "Concrete Crack Segmentation Dataset" in text
