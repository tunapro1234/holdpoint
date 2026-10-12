"""Model backends behind one interface: cpu | recorded | amd_vllm."""
from __future__ import annotations

from .base import (
    VLM_LABELS,
    Backend,
    ClassifyResult,
    SegmentResult,
    VlmResult,
    image_sha256,
    make_provenance,
)


def get_backend(name: str, **kwargs) -> Backend:
    if name == "cpu":
        from .cpu import CpuBackend

        return CpuBackend(**kwargs)
    if name == "recorded":
        from .recorded import RecordedBackend

        return RecordedBackend(**kwargs)
    if name == "amd_vllm":
        from .amd_vllm import AmdVllmBackend

        return AmdVllmBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r} (cpu | recorded | amd_vllm)")


__all__ = [
    "VLM_LABELS",
    "Backend",
    "ClassifyResult",
    "SegmentResult",
    "VlmResult",
    "get_backend",
    "image_sha256",
    "make_provenance",
]
