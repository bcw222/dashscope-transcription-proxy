"""Endpoint tests for ``backend=vllm`` — routing, validation and auth behaviour.

The real HTTP call to the local ASR server is stubbed; these tests cover the
FastAPI layer only (``test_local_client.py`` covers the client itself).
"""

from __future__ import annotations

import base64
from importlib import import_module

import pytest
from fastapi.testclient import TestClient

from dashscope_transcription_proxy.app import app

# 注意：`from dashscope_transcription_proxy import app` 会拿到 **FastAPI 实例**
# （因为 __init__.py 里 `from .app import app` 遮蔽了同名子模块），
# 所以要改 `_client` 全局必须用 import_module 拿到真正的模块对象。
app_module = import_module("dashscope_transcription_proxy.app")


class _StubClient:
    """Mimics both clients' contract; returns a DashScope-shaped raw dict."""

    def __init__(self, text: str = "本地识别结果", language: str = "zh") -> None:
        self.text = text
        self.language = language
        self.calls: list[dict] = []

    async def start(self) -> None:  # pragma: no cover - trivial
        return None

    async def stop(self) -> None:  # pragma: no cover - trivial
        return None

    async def transcribe(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {
            "status_code": 200,
            "code": None,
            "message": None,
            "request_id": "stub-1",
            "output": {"choices": [{"message": {
                "role": "assistant",
                "content": self.text,
                "annotations": [{"type": "audio_info", "language": self.language, "words": []}],
            }}]},
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }


@pytest.fixture
def vllm_env(monkeypatch):
    """backend=vllm, real (non-mock) code path, downstream key still required."""
    monkeypatch.setenv("DASHSCOPE_PROXY_BACKEND", "vllm")
    monkeypatch.setenv("DASHSCOPE_PROXY_LOCAL_MODE", "false")
    monkeypatch.setenv("DASHSCOPE_PROXY_VLLM_MODEL", "qwen3-asr")
    monkeypatch.setenv("DASHSCOPE_PROXY_VLLM_BASE_URL", "http://asr:8000/v1")
    from dashscope_transcription_proxy.settings import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def stub_client(monkeypatch):
    """Install the stub as the singleton client (bypasses _get_client)."""
    stub = _StubClient()
    monkeypatch.setattr(app_module, "_client", stub)
    yield stub
    monkeypatch.setattr(app_module, "_client", None)


def _wav_bytes() -> bytes:
    return b"\x00\x01fake-audio"


def test_health_reports_vllm_backend(vllm_env):
    with TestClient(app) as c:
        body = c.get("/health").json()
    assert body["backend"] == "vllm"
    assert body["model"] == "qwen3-asr"          # vllm_model, not the DashScope id
    assert body["vllm_base_url"] == "http://asr:8000/v1"


def test_v1_alias_route_exists(vllm_env, stub_client):
    with TestClient(app) as c:
        r = c.post(
            "/v1/audio/transcriptions",
            files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
            data={"response_format": "json"},
            headers={"Authorization": "Bearer ***"},
        )
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "本地识别结果"


def test_bare_route_still_works(vllm_env, stub_client):
    with TestClient(app) as c:
        r = c.post(
            "/audio/transcriptions",
            files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
            data={"response_format": "json"},
            headers={"Authorization": "Bearer ***"},
        )
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "本地识别结果"


def test_srt_rejected_on_vllm_backend(vllm_env, stub_client):
    """No word-level timestamps locally ⇒ must fail loudly, not emit empty subtitles."""
    with TestClient(app) as c:
        r = c.post(
            "/audio/transcriptions",
            files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
            data={"response_format": "srt"},
            headers={"Authorization": "Bearer ***"},
        )
    assert r.status_code == 400
    assert "word-level timestamps" in r.json()["detail"]


def test_vtt_rejected_on_vllm_backend(vllm_env, stub_client):
    with TestClient(app) as c:
        r = c.post(
            "/audio/transcriptions",
            files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
            data={"response_format": "vtt"},
            headers={"Authorization": "Bearer ***"},
        )
    assert r.status_code == 400


def test_verbose_json_reports_language_and_token_usage(vllm_env, stub_client):
    with TestClient(app) as c:
        r = c.post(
            "/audio/transcriptions",
            files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
            data={"response_format": "verbose_json"},
            headers={"Authorization": "Bearer ***"},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "本地识别结果"
    assert body["language"] == "zh"
    assert body["words"] == []
    assert body["usage"] == {"type": "tokens", "input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_text_format_returns_plain_text(vllm_env, stub_client):
    with TestClient(app) as c:
        r = c.post(
            "/audio/transcriptions",
            files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
            data={"response_format": "text"},
            headers={"Authorization": "Bearer ***"},
        )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert r.text == "本地识别结果"


def test_json_body_input_audio_path(vllm_env, stub_client):
    b64 = base64.b64encode(_wav_bytes()).decode()
    with TestClient(app) as c:
        r = c.post(
            "/audio/transcriptions",
            json={"input_audio": {"data": f"data:audio/wav;base64,{b64}"}, "response_format": "json"},
            headers={"Authorization": "Bearer ***"},
        )
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "本地识别结果"
    # data: URI must be passed through untouched
    assert stub_client.calls[0]["audio_ref"].startswith("data:audio/wav;base64,")


def test_prompt_forwarded_as_context(vllm_env, stub_client):
    with TestClient(app) as c:
        r = c.post(
            "/audio/transcriptions",
            files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
            data={"prompt": "傅里叶变换 频谱", "response_format": "json"},
            headers={"Authorization": "Bearer ***"},
        )
    assert r.status_code == 200, r.text
    assert stub_client.calls[0]["context"] == "傅里叶变换 频谱"


def test_missing_key_still_401_by_default(vllm_env):
    with TestClient(app) as c:
        r = c.post(
            "/audio/transcriptions",
            files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
        )
    assert r.status_code == 401


def test_anonymous_allowed_when_key_not_required(vllm_env, stub_client, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_PROXY_REQUIRE_DOWNSTREAM_KEY", "false")
    from dashscope_transcription_proxy.settings import get_settings
    get_settings.cache_clear()
    with TestClient(app) as c:
        r = c.post(
            "/audio/transcriptions",
            files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
            data={"response_format": "json"},
        )
    assert r.status_code == 200, r.text
    assert r.json()["text"] == "本地识别结果"


def test_stream_still_501(vllm_env, stub_client):
    with TestClient(app) as c:
        r = c.post(
            "/audio/transcriptions",
            files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
            data={"stream": "true"},
            headers={"Authorization": "Bearer ***"},
        )
    assert r.status_code == 501
