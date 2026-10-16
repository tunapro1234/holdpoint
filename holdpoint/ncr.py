"""Defect report (NCR) draft with a number guard.

The report fields come from deterministic code. A text writer (template, or an LLM behind an
OpenAI-compatible endpoint) turns them into prose. Every number in the prose must already appear in the
fields; otherwise the text is rejected and the deterministic template is used instead. Numbers written as
words ("two", "half a millimetre") count too, as in the VLM check (holdpoint.vlm): a number word must appear
in the fields, and a unit must follow a number.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

from .vlm import _NUM_WORDS, _UNITS

NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?")
NUM_WORD_RE = re.compile(rf"\b(?:{_NUM_WORDS})\b", re.IGNORECASE)
UNIT_RE = re.compile(rf"\b(?:{_UNITS})\b|%", re.IGNORECASE)
AFTER_NUMBER_RE = re.compile(r"\d\s*$")  # a unit is only allowed right after a number ("0.35 mm", "0.35mm")


class NumberGuardError(ValueError):
    def __init__(self, invented: list[str]):
        super().__init__(f"text contains numbers not in the input fields: {invented}")
        self.invented = invented


def _norm(tok: str) -> str:
    v = float(tok.replace(",", "."))
    return f"{v:.6f}".rstrip("0").rstrip(".")


def numbers_in(text: str) -> list[str]:
    return NUMBER_RE.findall(text)


def _is_hash_key(key) -> bool:
    k = str(key).lower()
    return "sha" in k or "hash" in k


def _field_numbers(obj) -> set[str]:
    if isinstance(obj, dict):
        # hashes are not quantities: digit runs inside a hex digest must not whitelist numbers
        vals = [v for k, v in obj.items() if not _is_hash_key(k)]
        return set().union(*(_field_numbers(v) for v in vals)) if vals else set()
    if isinstance(obj, (list, tuple)):
        return set().union(*(_field_numbers(v) for v in obj)) if obj else set()
    if isinstance(obj, bool) or obj is None:
        return set()
    if isinstance(obj, (int, float)):
        return {_norm(repr(obj))}
    return {_norm(t) for t in numbers_in(str(obj))}


def _field_words(obj) -> set[str]:
    """Number words that appear in the fields' own text (outside hashes)."""
    if isinstance(obj, dict):
        return set().union(set(), *(_field_words(v) for k, v in obj.items() if not _is_hash_key(k)))
    if isinstance(obj, (list, tuple)):
        return set().union(set(), *(_field_words(v) for v in obj))
    return {w.lower() for w in NUM_WORD_RE.findall(obj)} if isinstance(obj, str) else set()


def check_numbers(text: str, fields: dict) -> None:
    """Raise NumberGuardError if `text` has a number that is not in `fields`: digits, number words, or a unit
    with no number in front of it."""
    allowed, words = _field_numbers(fields), _field_words(fields)
    found = [(m.start(), m.group()) for m in NUMBER_RE.finditer(text) if _norm(m.group()) not in allowed]
    found += [(m.start(), m.group()) for m in NUM_WORD_RE.finditer(text) if m.group().lower() not in words]
    found += [(m.start(), m.group()) for m in UNIT_RE.finditer(text) if not AFTER_NUMBER_RE.search(text[:m.start()])]
    if found:
        raise NumberGuardError([t for _, t in sorted(found)])


@dataclass
class DefectReport:
    element_id: str
    photo_sha256: str
    band_mm: tuple[float, float] | None
    suggestion: str
    limit_mm: float | None
    vlm_label: str
    vlm_sentence: str = ""
    model_version: str = ""
    text: str = ""
    text_source: str = ""
    provenance: dict = field(default_factory=dict)

    def fields(self) -> dict:
        """The facts a writer may use (no text)."""
        d = asdict(self)
        for k in ("text", "text_source", "provenance"):
            d.pop(k)
        return d

    def to_dict(self) -> dict:
        return asdict(self)


class TextWriter(Protocol):
    name: str

    def write(self, fields: dict) -> str: ...


SUGGESTION_TEXT = {"measure_with_gauge": "measure with a gauge"}  # as the app shows it; others read fine as is


def _band_text(band) -> str:
    return "width not measurable (no scale card)" if band is None else f"{band[0]}-{band[1]} mm"


class TemplateWriter:
    name = "template"

    def write(self, f: dict) -> str:
        limit = "" if f["limit_mm"] is None else f" against a limit of {f['limit_mm']} mm from the user table"
        return (
            f"Element {f['element_id']}: crack width {_band_text(f['band_mm'])}{limit}. "
            f"Suggestion: {SUGGESTION_TEXT.get(f['suggestion'], f['suggestion'].replace('_', ' '))}. "
            f"Second check: {f['vlm_label'].replace('_', ' ')}. "
            "An inspector must confirm before this report is opened."
        )


SYSTEM_PROMPT = (
    "You write short defect-report paragraphs for concrete crack inspections. Use only the facts in the JSON. "
    "Do not add any number that is not in the JSON. Do not give a final verdict: an inspector decides. "
    "Two to three plain English sentences."
)


class LlmWriter:
    """Any OpenAI-compatible chat endpoint (vLLM on AMD, or another server)."""

    name = "llm"

    def __init__(self, base_url: str | None = None, model: str | None = None, client=None):
        self.base_url = base_url or os.environ.get("HOLDPOINT_LLM_BASE_URL") or os.environ.get("HOLDPOINT_VLLM_BASE_URL")
        self.model = model or os.environ.get("HOLDPOINT_LLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
        self._client = client
        if self._client is None and not self.base_url:
            raise ValueError("LlmWriter needs base_url (HOLDPOINT_LLM_BASE_URL) or a client")

    def write(self, f: dict) -> str:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=os.environ.get("HOLDPOINT_LLM_API_KEY", "EMPTY"))
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(f)},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


def draft_report(report: DefectReport, writer: TextWriter | None = None) -> DefectReport:
    """Fill report.text. LLM text that fails the number guard is replaced by the template text."""
    fields = report.fields()
    fallback = TemplateWriter()
    writer = writer or fallback
    try:
        text = writer.write(fields)
        check_numbers(text, fields)
        source = writer.name
    except NumberGuardError as e:
        if writer is fallback:
            raise
        text = fallback.write(fields)
        check_numbers(text, fields)
        source = f"template ({writer.name} text rejected: invented numbers {e.invented})"
    report.text, report.text_source = text, source
    return report
