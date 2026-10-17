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


# -- the real openai client against a fake endpoint (tests/conftest.py: no network) --------------------------

from conftest import chat_reply, error_reply, unreachable  # noqa: E402

CROP = np.full((16, 16, 3), 120, np.uint8)
URL = "http://amd.invalid/v1"


def test_400_on_response_format_retried_once_without_it(fake_endpoint):
    bodies = fake_endpoint(lambda r: error_reply(400) if b"response_format" in r.content
                           else chat_reply('{"label": "shadow", "sentence": "Soft shading, no break."}'))
    v = VlmClient(base_url=URL).check(CROP)
    assert len(bodies) == 2 and "response_format" in bodies[0] and "response_format" not in bodies[1]
    assert v.label == "shadow" and v.provenance["flags"] == ["response_format_rejected"]
    assert v.provenance["label"] == "live on AMD"


def test_retry_without_schema_still_checks_schema_and_numbers(fake_endpoint):
    fake_endpoint(lambda r: error_reply(400) if b"response_format" in r.content
                  else chat_reply('{"label": "crack", "sentence": "About two millimetres wide."}'))
    v = VlmClient(base_url=URL).check(CROP)
    assert v.label == "crack" and v.sentence == NUMERIC_DROPPED
    fake_endpoint(lambda r: error_reply(400) if b"response_format" in r.content else chat_reply("It is a crack."))
    assert VlmClient(base_url=URL).check(CROP).label == "unclear"


def test_other_errors_are_not_retried(fake_endpoint):
    bodies = fake_endpoint(lambda r: error_reply(500))
    with pytest.raises(Exception) as e:
        VlmClient(base_url=URL).check(CROP)
    assert getattr(e.value, "status_code", None) == 500 and len(bodies) == 1


@pytest.mark.parametrize("handler,err", [(unreachable, "APIConnectionError"),
                                         (lambda r: error_reply(503), "InternalServerError")])
def test_check_or_unclear_when_endpoint_does_not_answer(fake_endpoint, handler, err):
    fake_endpoint(handler)
    v = VlmClient(base_url=URL, device="AMD Instinct MI300X").check_or_unclear(CROP)
    assert v.label == "unclear" and err in v.sentence
    assert v.provenance["error"] == err and v.provenance["device"] == "AMD Instinct MI300X"
    assert v.provenance["label"].startswith("live on AMD, no answer")


def test_from_env(monkeypatch):
    monkeypatch.delenv("HOLDPOINT_VLLM_BASE_URL", raising=False)
    assert VlmClient.from_env() is None
    monkeypatch.setenv("HOLDPOINT_VLLM_BASE_URL", URL)
    monkeypatch.setenv("HOLDPOINT_VLLM_DEVICE", "AMD Instinct MI300X")
    c = VlmClient.from_env()
    assert c.base_url == URL and c.device == "AMD Instinct MI300X"
