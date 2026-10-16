"""Recorded backend: replays results computed earlier (e.g. on AMD MI300X) from a JSON cache.

Cache format (one JSON file):
{
  "provenance": {"backend": "amd_notebook", "device": "AMD Instinct MI300X", "model": "...",
                 "timestamp": "...", "command": "..."},
  "results": {
    "<image sha256>": {
      "mask": {"shape": [H, W], "rle": [start, length, ...]},   # optional
      "width_masks": {"<input side>": {"shape": ..., "rle": ...}}, # optional: masks from segment_at(img, side)
      "p_crack": 0.93,                                            # optional
      "vlm": {"label": "crack", "sentence": "..."},               # optional
      "provenance": {...}                                         # optional per-item override
    }
  }
}
RLE runs index the row-major flattened mask.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .base import Backend, ClassifyResult, SegmentResult, VlmResult, image_sha256


class NotRecorded(KeyError):
    """The image has no recorded result for the requested task."""


def mask_to_rle(mask: np.ndarray) -> dict:
    flat = np.asarray(mask, dtype=bool).ravel()
    padded = np.concatenate([[False], flat, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    rle = np.empty(2 * len(starts), dtype=np.int64)
    rle[0::2], rle[1::2] = starts, ends - starts
    return {"shape": list(mask.shape), "rle": rle.tolist()}


def rle_to_mask(obj: dict) -> np.ndarray:
    shape = tuple(obj["shape"])
    flat = np.zeros(int(np.prod(shape)), dtype=bool)
    rle = obj["rle"]
    for start, length in zip(rle[0::2], rle[1::2]):
        flat[start : start + length] = True
    return flat.reshape(shape)


class RecordedBackend(Backend):
    name = "recorded"

    def __init__(self, path: str | os.PathLike | None = None):
        p = path or os.environ.get("HOLDPOINT_RECORDED")
        if not p:
            raise ValueError("RecordedBackend needs a cache path (argument or HOLDPOINT_RECORDED)")
        self.path = Path(p)
        data = json.loads(self.path.read_text())
        self._prov = dict(data.get("provenance", {}))
        self._results = data.get("results", {})

    def provenance(self) -> dict:
        prov = dict(self._prov)
        prov["recorded_backend"] = self._prov.get("backend")
        prov["backend"] = self.name
        prov["source"] = self.path.name
        prov["label"] = _label(self._prov)
        return prov

    def _item(self, img: np.ndarray, key: str) -> tuple[dict, dict]:
        sha = image_sha256(img)
        item = self._results.get(sha)
        if item is None or key not in item:
            raise NotRecorded(f"no recorded {key!r} for image {sha[:12]}")
        prov = self.provenance()
        prov.update(item.get("provenance", {}))
        prov["image_sha256"] = sha
        return item, prov

    def segment(self, img: np.ndarray) -> SegmentResult:
        item, prov = self._item(img, "mask")
        return SegmentResult(mask=rle_to_mask(item["mask"]), provenance=prov)

    def segment_at(self, img: np.ndarray, side: int) -> SegmentResult:
        item, prov = self._item(img, "width_masks")
        if str(side) not in item["width_masks"]:
            raise NotRecorded(f"no recorded width mask at side {side} for image {prov['image_sha256'][:12]}")
        return SegmentResult(mask=rle_to_mask(item["width_masks"][str(side)]), provenance={**prov, "input_side": side})

    def classify(self, img: np.ndarray) -> ClassifyResult:
        item, prov = self._item(img, "p_crack")
        return ClassifyResult(p_crack=float(item["p_crack"]), provenance=prov)

    def vlm_check(self, crop: np.ndarray) -> VlmResult:
        item, prov = self._item(crop, "vlm")
        return VlmResult(label=item["vlm"]["label"], sentence=item["vlm"]["sentence"], provenance=prov)


def _label(prov: dict) -> str:
    device = prov.get("device", "unknown device")
    when = prov.get("timestamp", "unknown date")
    cmd = prov.get("command", "unknown command")
    return f"computed on {device}, {when}, {cmd}"
