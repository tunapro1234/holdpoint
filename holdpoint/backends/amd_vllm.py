"""AMD vLLM backend: talks to an OpenAI-compatible vLLM server running on AMD hardware.

Only vlm_check is served live (see holdpoint.vlm); segmentation and classification on AMD run as notebook
batch jobs and reach the app through the recorded backend.

Environment:
  HOLDPOINT_VLLM_BASE_URL   e.g. http://host:8000/v1   (required)
  HOLDPOINT_VLLM_MODEL      default Qwen/Qwen2.5-VL-7B-Instruct
  HOLDPOINT_VLLM_API_KEY    default "EMPTY" (vLLM ignores it unless started with --api-key)
  HOLDPOINT_VLLM_DEVICE     shown in provenance, default "AMD GPU (vLLM)"
"""
from __future__ import annotations

import numpy as np

from ..vlm import DEFAULT_MODEL, VlmClient, parse_reply
from .base import Backend, ClassifyResult, SegmentResult, VlmResult, make_provenance

__all__ = ["AmdVllmBackend", "DEFAULT_MODEL", "parse_vlm_reply"]


class AmdVllmBackend(Backend):
    name = "amd_vllm"

    def __init__(self, base_url: str | None = None, model: str | None = None, client=None):
        self._vlm = VlmClient(base_url=base_url, model=model, client=client, backend=self.name)
        if not self._vlm.base_url:
            raise ValueError("AmdVllmBackend needs base_url (argument or HOLDPOINT_VLLM_BASE_URL)")
        self.base_url, self.model, self.device = self._vlm.base_url, self._vlm.model, self._vlm.device

    def provenance(self) -> dict:
        return make_provenance(self.name, self.device, self.model, base_url=self.base_url, label="live on AMD")

    def segment(self, img: np.ndarray) -> SegmentResult:
        raise NotImplementedError("segmentation on AMD runs as a notebook batch job; use the recorded backend")

    def classify(self, img: np.ndarray) -> ClassifyResult:
        raise NotImplementedError("classification on AMD runs as a notebook batch job; use the recorded backend")

    def vlm_check(self, crop: np.ndarray) -> VlmResult:
        return self._vlm.check(crop)


def parse_vlm_reply(text: str) -> tuple[str, str]:
    p = parse_reply(text)
    return p.label, p.sentence
