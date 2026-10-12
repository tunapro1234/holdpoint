import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from holdpoint.backends import VLM_LABELS
from holdpoint.vlm import (
    NUMERIC_DROPPED,
    PROMPT,
    SCHEMA,
    VlmClient,
    has_numeric_content,
    parse_reply,
)

FIXTURE = json.loads((Path(__file__).parent / "data" / "vlm_replies_synthetic.json").read_text())
CASES = FIXTURE["cases"]


def test_fixture_is_marked_synthetic():
    assert FIXTURE["synthetic"] is True and "Not model output" in FIXTURE["note"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_recorded_replies(case):
    p = parse_reply(case["reply"])
    assert p.label == case["label"]
    assert p.flags == case["flags"]
    assert p.label in VLM_LABELS
    assert not has_numeric_content(p.sentence)


def test_schema_is_closed_vocabulary():
    assert SCHEMA["properties"]["label"]["enum"] == list(VLM_LABELS)
    assert SCHEMA["additionalProperties"] is False
    assert set(SCHEMA["required"]) == {"label", "sentence"}
    assert "Do not give any number" in PROMPT


def test_numeric_ban_drops_sentence_but_keeps_label():
    p = parse_reply('{"label": "crack", "sentence": "Roughly 15 cm long."}')
    assert p.label == "crack" and p.sentence == NUMERIC_DROPPED


@pytest.mark.parametrize("text,numeric", [
    ("A thin line with sharp edges.", False),
    ("Someone left a stain.", False),  # 'one' inside a word is not a number
    ("Width 0.2", True),
    ("three branches", True),
    ("50% of the face", True),
    ("several mm", True),
])
def test_has_numeric_content(text, numeric):
    assert has_numeric_content(text) is numeric


def test_long_sentence_truncated():
    p = parse_reply(json.dumps({"label": "crack", "sentence": "line " * 100}))
    assert len(p.sentence) <= SCHEMA["properties"]["sentence"]["maxLength"] and "truncated" in p.flags


def test_client_sends_schema_and_image(monkeypatch):
    sent = {}
    reply = CASES[0]["reply"]

    def create(**kw):
        sent.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.delenv("HOLDPOINT_VLLM_BASE_URL", raising=False)
    v = VlmClient(client=client).check(np.full((16, 16, 3), 120, np.uint8))
    rf = sent["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"] == SCHEMA
    img = sent["messages"][0]["content"][1]["image_url"]["url"]
    assert img.startswith("data:image/png;base64,")
    assert sent["temperature"] == 0
    assert v.label == "crack" and v.provenance["flags"] == []
    assert {"backend", "device", "model", "timestamp"} <= v.provenance.keys()


def test_client_needs_url_or_client(monkeypatch):
    monkeypatch.delenv("HOLDPOINT_VLLM_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        VlmClient()
