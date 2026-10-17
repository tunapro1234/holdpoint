"""VLM second check: one crop in, one label from a fixed vocabulary plus one sentence out.

The model is asked for JSON that matches SCHEMA (vLLM enforces it with structured outputs), and the reply
is validated again here, because a server may ignore the schema:
  - anything that is not exactly {"label", "sentence"} with string values -> unclear
  - a label outside the vocabulary -> unclear
  - a sentence with numeric content (digits, number words, units) -> sentence dropped, flagged.
    Widths come from the measuring code, never from the language model.
"""
from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass

import numpy as np

from .backends.base import VLM_LABELS, VlmResult, make_provenance

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_SENTENCE = 200
# the app waits on these calls, so a dead endpoint has to fail fast (the openai default is 600 s, 2 retries)
TIMEOUT_S = 30.0
MAX_RETRIES = 1

SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": list(VLM_LABELS)},
        "sentence": {"type": "string", "maxLength": MAX_SENTENCE},
    },
    "required": ["label", "sentence"],
    "additionalProperties": False,
}

PROMPT = (
    "This is a crop from a photo of a concrete surface. Say what the dark mark is. "
    "Answer with JSON only, matching this schema: "
    + json.dumps(SCHEMA)
    + ". Labels: crack = a real crack in the concrete; form_oil_stain = release-agent or oil mark; "
    "shadow = shading, no surface break; joint_edge = construction or panel joint; air_void = surface bug hole; "
    "exposed_rebar = steel visible; unclear = cannot tell. "
    "The sentence describes what you see in plain words. Do not give any number, size or measurement."
)

_NUM_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|"
    "thousand|half|quarter|dozen|single|double|twice"
)
_UNITS = r"mm|cm|m|millimet\w*|centimet\w*|met(?:er|re)s?|inch\w*|percent|%"
NUMERIC_RE = re.compile(rf"\d|\b(?:{_NUM_WORDS})\b|\b(?:{_UNITS})\b|%", re.IGNORECASE)

UNPARSEABLE = "Model reply was not valid JSON for the schema."
OFF_VOCAB = "Model label was outside the vocabulary."
NUMERIC_DROPPED = "Sentence removed: it contained numeric content."


def api_errors() -> tuple[type[BaseException], ...]:
    """Errors that mean the endpoint did not give an answer (network, timeout, HTTP status), not a bug here."""
    try:
        from openai import OpenAIError
    except ImportError:
        return (OSError,)
    return (OSError, OpenAIError)


@dataclass
class ParsedReply:
    label: str
    sentence: str
    flags: list[str]


def has_numeric_content(text: str) -> bool:
    return bool(NUMERIC_RE.search(text))


def parse_reply(text: str) -> ParsedReply:
    """Validate a raw model reply against SCHEMA and the numeric ban."""
    start, end = text.find("{"), text.rfind("}")
    try:
        obj = json.loads(text[start : end + 1]) if start != -1 and end > start else None
    except json.JSONDecodeError:
        obj = None
    if (not isinstance(obj, dict) or set(obj) != {"label", "sentence"}
            or not all(isinstance(obj[k], str) for k in obj)):
        return ParsedReply("unclear", UNPARSEABLE, ["schema_violation"])
    label, sentence = obj["label"].strip().lower(), obj["sentence"].strip()
    flags = []
    if start > 0 or end < len(text.rstrip()) - 1:
        flags.append("extra_text")
    if label not in VLM_LABELS:
        return ParsedReply("unclear", OFF_VOCAB, flags + ["off_vocabulary"])
    if has_numeric_content(sentence):
        return ParsedReply(label, NUMERIC_DROPPED, flags + ["numeric_content"])
    if len(sentence) > MAX_SENTENCE:
        sentence = sentence[:MAX_SENTENCE].rsplit(" ", 1)[0]
        flags.append("truncated")
    return ParsedReply(label, sentence, flags)


def png_data_url(img: np.ndarray) -> str:
    import cv2

    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr[..., :3].astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("could not encode crop as PNG")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def messages_for(image_url: str) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }]


class VlmClient:
    """OpenAI-compatible chat client (vLLM serve). The JSON schema is sent as a structured-output constraint."""

    def __init__(self, base_url: str | None = None, model: str | None = None, client=None,
                 device: str | None = None, backend: str = "amd_vllm"):
        self.base_url = base_url or os.environ.get("HOLDPOINT_VLLM_BASE_URL")
        if client is None and not self.base_url:
            raise ValueError("VlmClient needs base_url (HOLDPOINT_VLLM_BASE_URL) or a client")
        self.model = model or os.environ.get("HOLDPOINT_VLLM_MODEL", DEFAULT_MODEL)
        self.device = device or os.environ.get("HOLDPOINT_VLLM_DEVICE", "AMD GPU (vLLM)")
        self.backend = backend
        self._client = client

    @classmethod
    def from_env(cls) -> VlmClient | None:
        """The client the app uses when HOLDPOINT_VLLM_BASE_URL is set, else None."""
        return cls() if os.environ.get("HOLDPOINT_VLLM_BASE_URL") else None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=os.environ.get("HOLDPOINT_VLLM_API_KEY", "EMPTY"),
                                  timeout=TIMEOUT_S, max_retries=MAX_RETRIES)
        return self._client

    def check(self, crop: np.ndarray) -> VlmResult:
        create = self._get_client().chat.completions.create
        kw = dict(model=self.model, temperature=0, max_tokens=120, messages=messages_for(png_data_url(crop)))
        flags = []
        try:
            resp = create(**kw, response_format={
                "type": "json_schema", "json_schema": {"name": "vlm_check", "schema": SCHEMA, "strict": True}})
        except Exception as e:
            if getattr(e, "status_code", None) != 400:
                raise
            # some OpenAI-compatible servers refuse structured outputs; parse_reply still enforces the schema
            resp = create(**kw)
            flags.append("response_format_rejected")
        p = parse_reply(resp.choices[0].message.content or "")
        prov = make_provenance(self.backend, self.device, self.model, base_url=self.base_url,
                               label="live on AMD", flags=flags + p.flags)
        return VlmResult(label=p.label, sentence=p.sentence, provenance=prov)

    def check_or_unclear(self, crop: np.ndarray) -> VlmResult:
        """check(), except that an endpoint which does not answer gives "unclear" with the reason, so the
        measuring flow goes on without the second check."""
        try:
            return self.check(crop)
        except api_errors() as e:
            err = type(e).__name__
            prov = make_provenance(self.backend, self.device, self.model, base_url=self.base_url,
                                   label=f"live on AMD, no answer ({err})", error=err)
            return VlmResult(label="unclear", sentence=f"Second check unavailable: the endpoint did not answer ({err}).",
                             provenance=prov)
