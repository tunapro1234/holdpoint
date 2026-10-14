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

    def __init__(self, weights: str | os.PathLike | None = None, encoder: str = "resnet18", side: int | None = None):
        w = weights or os.environ.get("HOLDPOINT_UNET_WEIGHTS")
        self.weights = Path(w) if w else None
        self.encoder = encoder
        # long side the photo is resized to for the UNet (trained on 256 px crops; full-size photos are off-scale)
        self.side = side or int(os.environ.get("HOLDPOINT_UNET_SIDE", "1024"))
        self._net = None  # loaded lazily
        self._norm = None  # (mean, std) from the checkpoint, if it has them

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
        state = torch.load(self.weights, map_location="cpu", weights_only=False)
        meta = state.get("meta", {}) if isinstance(state, dict) else {}
        # checkpoints from holdpoint.vision.train_seg carry their encoder and input normalization
        self.encoder = meta.get("encoder", self.encoder)
        norm = meta.get("input", {})
        self._norm = (norm["mean"], norm["std"]) if "mean" in norm else None
        net = smp.Unet(encoder_name=self.encoder, encoder_weights=None, in_channels=3, classes=1)
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
        mask = _unet_mask(net, img, norm=self._norm, side=self.side) if net is not None else baseline_mask(img)
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


def baseline_mask(img: np.ndarray, kernel: int = 15, min_contrast: int = 25, min_area: int = 20,
                  min_area_frac: float = 1e-4, min_elongation: float = 4.0) -> np.ndarray:
    """Dark thin structures: black-hat top-hat, threshold (Otsu, never below min_contrast grey levels),
    keep only blobs that are big enough and elongated (length^2 / area), like cracks."""
    import cv2

    gray = cv2.GaussianBlur(to_gray(img), (3, 3), 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    if int(bh.max()) < min_contrast:
        return np.zeros(gray.shape, dtype=bool)
    otsu, _ = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = (bh >= max(otsu, min_contrast)).astype(np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    diag2 = stats[:, cv2.CC_STAT_WIDTH].astype(np.float64) ** 2 + stats[:, cv2.CC_STAT_HEIGHT].astype(np.float64) ** 2
    keep = (area >= max(min_area, min_area_frac * gray.size)) & (diag2 / np.maximum(area, 1) >= min_elongation)
    keep[0] = False
    return keep[labels]


def _unet_mask(net, img: np.ndarray, threshold: float = 0.5, norm: tuple | None = None,
               side: int | None = None) -> np.ndarray:
    """Probability map at long side `side` (None = native size), resized back to the photo."""
    import cv2
    import torch

    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr[..., :3].astype(np.float32) / 255.0
    if norm is not None:
        arr = (arr - np.asarray(norm[0], np.float32)) / np.asarray(norm[1], np.float32)
    h0, w0 = arr.shape[:2]
    if side and max(h0, w0) != side:
        s = side / max(h0, w0)
        arr = cv2.resize(arr, (max(32, round(w0 * s)), max(32, round(h0 * s))), interpolation=cv2.INTER_AREA)
    h, w = arr.shape[:2]
    ph, pw = (-h) % 32, (-w) % 32  # UNet needs sides divisible by 32
    arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(net(x))[0, 0].numpy()[:h, :w]
    if (h, w) != (h0, w0):
        prob = cv2.resize(prob, (w0, h0), interpolation=cv2.INTER_LINEAR)
    return prob >= threshold
