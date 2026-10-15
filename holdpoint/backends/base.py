"""Backend interface shared by every model runner.

A backend answers three questions about an image and says where the answer came from:
segment -> crack mask, classify -> crack probability, vlm_check -> label from a fixed vocabulary.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

VLM_LABELS = (
    "crack",
    "form_oil_stain",
    "shadow",
    "joint_edge",
    "air_void",
    "exposed_rebar",
    "unclear",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def image_sha256(img: np.ndarray) -> str:
    """Hash of the pixel array (shape + dtype + bytes); stable key for cached results."""
    arr = np.ascontiguousarray(img)
    h = hashlib.sha256()
    h.update(f"{arr.shape}|{arr.dtype}".encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def make_provenance(backend: str, device: str, model: str, **extra: Any) -> dict:
    prov = {"backend": backend, "device": device, "model": model, "timestamp": utc_now()}
    prov.update(extra)
    return prov


@dataclass
class SegmentResult:
    mask: np.ndarray  # bool, HxW
    provenance: dict = field(default_factory=dict)


@dataclass
class ClassifyResult:
    p_crack: float
    provenance: dict = field(default_factory=dict)


@dataclass
class VlmResult:
    label: str
    sentence: str
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in VLM_LABELS:
            raise ValueError(f"label {self.label!r} not in vocabulary {VLM_LABELS}")


class Backend(ABC):
    name: str = "base"

    @abstractmethod
    def segment(self, img: np.ndarray) -> SegmentResult: ...

    @abstractmethod
    def classify(self, img: np.ndarray) -> ClassifyResult: ...

    @abstractmethod
    def vlm_check(self, crop: np.ndarray) -> VlmResult: ...

    @abstractmethod
    def provenance(self) -> dict: ...

    def segment_at(self, img: np.ndarray, side: int) -> SegmentResult:
        """Mask with the photo resized to long side `side` first. Backends without a resolution knob do not
        implement it; the pipeline then measures width on the `segment` mask."""
        raise NotImplementedError


def to_gray(img: np.ndarray) -> np.ndarray:
    """uint8 grayscale from HxW or HxWx3/4 (RGB) input."""
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr
