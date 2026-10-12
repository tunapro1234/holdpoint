"""Local CPU backend.

If torch, segmentation_models_pytorch and a weights file are available, the UNet runs on CPU.
Otherwise a classical baseline (black-hat morphology + Otsu threshold) produces the mask, and the
provenance says so. The baseline is a fallback, not the trained model.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .base import (
    Backend,
    ClassifyResult,
    SegmentResult,
    VlmResult,
    make_provenance,
    to_gray,
)

BASELINE_MODEL = "classical-blackhat-otsu-v0"


class CpuBackend(Backend):
    name = "cpu"

    def __init__(self, weights: str | os.PathLike | None = None, encoder: str = "resnet18"):
        w = weights or os.environ.get("HOLDPOINT_UNET_WEIGHTS")
        self.weights = Path(w) if w else None
        self.encoder = encoder
        self._net = None  # loaded lazily

    # -- model selection ------------------------------------------------------
    def _unet(self):
        """Return a loaded UNet or None. torch is imported only here."""
        if self._net is not None:
            return self._net
        if self.weights is None or not self.weights.is_file():
            return None
        try:
            import torch
            import segmentation_models_pytorch as smp
        except ImportError:
            return None
        net = smp.Unet(encoder_name=self.encoder, encoder_weights=None, in_channels=3, classes=1)
        state = torch.load(self.weights, map_location="cpu")
        net.load_state_dict(state.get("state_dict", state) if isinstance(state, dict) else state)
        net.eval()
        self._net = net
        return net

    def _model_name(self) -> str:
        if self._unet() is not None:
            return f"unet-{self.encoder}:{self.weights.name}"
        return BASELINE_MODEL

    def provenance(self) -> dict:
        return make_provenance(self.name, "cpu", self._model_name(), label="live on CPU")

    # -- tasks ----------------------------------------------------------------
    def segment(self, img: np.ndarray) -> SegmentResult:
        net = self._unet()
        mask = _unet_mask(net, img) if net is not None else baseline_mask(img)
        return SegmentResult(mask=mask, provenance=self.provenance())

    def classify(self, img: np.ndarray) -> ClassifyResult:
        # Without a trained classifier, p is the masked-area share squashed to [0, 1]; provenance names the method.
        mask = self.segment(img).mask
        frac = float(mask.mean()) if mask.size else 0.0
        p = float(1.0 - np.exp(-frac / 0.01))
        return ClassifyResult(p_crack=p, provenance=self.provenance())

    def vlm_check(self, crop: np.ndarray) -> VlmResult:
        prov = make_provenance(self.name, "cpu", "none")
        return VlmResult(label="unclear", sentence="No vision-language model on the CPU backend.", provenance=prov)


def baseline_mask(img: np.ndarray, kernel: int = 15, min_area: int = 20) -> np.ndarray:
    """Dark thin structures: black-hat top-hat, Otsu threshold, drop tiny blobs."""
    import cv2

    gray = to_gray(img)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    if int(bh.max()) == 0:
        return np.zeros(gray.shape, dtype=bool)
    _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def _unet_mask(net, img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    import torch

    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr[..., :3].astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    ph, pw = (-h) % 32, (-w) % 32  # UNet needs sides divisible by 32
    arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(net(x))[0, 0].numpy()
    return prob[:h, :w] >= threshold
