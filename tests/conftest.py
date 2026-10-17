"""A fake OpenAI-compatible endpoint: requests from the real openai client go to an in-process MockTransport.
No network, no port."""
import json

import pytest

try:
    import httpx2 as httpx  # the HTTP library of openai >= 3
except ImportError:
    try:
        import httpx
    except ImportError:
        httpx = None


def chat_reply(content: str):
    return httpx.Response(200, json={
        "id": "fake", "object": "chat.completion", "created": 0, "model": "fake",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
    })


def error_reply(status: int):
    return httpx.Response(status, json={"error": {"message": f"fake {status}", "type": "fake"}})


def unreachable(request):
    raise httpx.ConnectError("connection refused", request=request)


@pytest.fixture
def fake_endpoint(monkeypatch):
    """install(handler) routes every openai.OpenAI(...) to `handler(request) -> Response` and returns the list
    of JSON bodies sent. Retries are off so a test sees each request once."""
    openai = pytest.importorskip("openai")
    if httpx is None:
        pytest.skip("no httpx")
    real = openai.OpenAI

    def install(handler):
        bodies = []

        def record(request):
            bodies.append(json.loads(request.content or b"{}"))
            return handler(request)

        def make(**kw):
            kw.update(http_client=httpx.Client(transport=httpx.MockTransport(record)), max_retries=0)
            return real(**kw)

        monkeypatch.setattr(openai, "OpenAI", make)
        return bodies

    return install
