"""AMD vLLM backend: talks to an OpenAI-compatible vLLM server running on AMD hardware.

Only vlm_check is served live; segmentation and classification on AMD run as notebook batch jobs
and reach the app through the recorded backend.

Environment:
  HOLDPOINT_VLLM_BASE_URL   e.g. http://host:8000/v1   (required)
  HOLDPOINT_VLLM_MODEL      default Qwen/Qwen2.5-VL-7B-Instruct
  HOLDPOINT_VLLM_API_KEY    default "EMPTY" (vLLM ignores it unless started with --api-key)
  HOLDPOINT_VLLM_DEVICE     shown in provenance, default "AMD GPU (vLLM)"
"""
from __future__ import annotations

import base64
import json
import os

import numpy as np

from .base import VLM_LABELS, Backend, ClassifyResult, SegmentResult, VlmResult, make_provenance

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT = (
    "You are checking one crop of a concrete surface photo. Answer with JSON only: "
    '{"label": <one of ' + ", ".join(VLM_LABELS) + '>, "sentence": <one short sentence, no numbers>}.'
)


class AmdVllmBackend(Backend):
    name = "amd_vllm"

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 client=None):
        self.base_url = base_url or os.environ.get("HOLDPOINT_VLLM_BASE_URL")
        if not self.base_url:
            raise ValueError("AmdVllmBackend needs base_url (argument or HOLDPOINT_VLLM_BASE_URL)")
        self.model = model or os.environ.get("HOLDPOINT_VLLM_MODEL", DEFAULT_MODEL)
        self._api_key = api_key or os.environ.get("HOLDPOINT_VLLM_API_KEY", "EMPTY")
        self.device = os.environ.get("HOLDPOINT_VLLM_DEVICE", "AMD GPU (vLLM)")
        self._client = client

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self._api_key)
        return self._client

    def provenance(self) -> dict:
        return make_provenance(self.name, self.device, self.model, base_url=self.base_url, label="live on AMD")

    def segment(self, img: np.ndarray) -> SegmentResult:
        raise NotImplementedError("segmentation on AMD runs as a notebook batch job; use the recorded backend")

    def classify(self, img: np.ndarray) -> ClassifyResult:
        raise NotImplementedError("classification on AMD runs as a notebook batch job; use the recorded backend")

    def vlm_check(self, crop: np.ndarray) -> VlmResult:
        url = "data:image/png;base64," + base64.b64encode(_png_bytes(crop)).decode()
        resp = self._get_client().chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }],
        )
        label, sentence = parse_vlm_reply(resp.choices[0].message.content or "")
        return VlmResult(label=label, sentence=sentence, provenance=self.provenance())


def parse_vlm_reply(text: str) -> tuple[str, str]:
    """Pull label + sentence out of the model reply; anything off-vocabulary becomes 'unclear'."""
    start, end = text.find("{"), text.rfind("}")
    try:
        obj = json.loads(text[start : end + 1]) if start != -1 and end > start else {}
    except json.JSONDecodeError:
        obj = {}
    label = str(obj.get("label", "")).strip().lower()
    sentence = str(obj.get("sentence", "")).strip()
    if label not in VLM_LABELS:
        return "unclear", "Model reply did not match the label vocabulary."
    return label, sentence


def _png_bytes(img: np.ndarray) -> bytes:
    import cv2

    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("could not encode crop as PNG")
    return buf.tobytes()
