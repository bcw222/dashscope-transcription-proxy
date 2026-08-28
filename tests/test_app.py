"""Endpoint tests using FastAPI TestClient with ``LOCAL_MODE``.

These tests exercise the HTTP layer without hitting DashScope.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from dashscope_transcription_proxy.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer sk-test-key"}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["local_mode"] is True


def test_missing_auth(client):
    r = client.post("/audio/transcriptions", files={"file": ("a.wav", b"x", "audio/wav")})
    assert r.status_code == 401


def test_multipart_json_format(client):
    files = {"file": ("a.wav", b"\x00\x01fake", "audio/wav")}
    data = {"model": "qwen3-asr-flash", "language": "zh", "response_format": "json"}
    r = client.post("/audio/transcriptions", files=files, data=data, headers=_auth_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["text"].startswith("[mock]")


def test_multipart_text_format(client):
    files = {"file": ("a.mp3", b"\x00\x01fake", "audio/mpeg")}
    data = {"response_format": "text"}
    r = client.post("/audio/transcriptions", files=files, data=data, headers=_auth_headers())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert r.text.startswith("[mock]")


def test_multipart_verbose_json_format(client):
    files = {"file": ("a.wav", b"\x00\x01fake", "audio/wav")}
    data = {"response_format": "verbose_json"}
    r = client.post("/audio/transcriptions", files=files, data=data, headers=_auth_headers())
    assert r.status_code == 200
    body = r.json()
    assert "text" in body
    assert "words" in body
    assert "segments" in body


def test_json_body_with_data_uri(client):
    b64 = base64.b64encode(b"\x00\x01fake").decode()
    payload = {
        "model": "qwen3-asr-flash",
        "language": "zh",
        "response_format": "json",
        "input_audio": {"data": f"data:audio/wav;base64,{b64}"},
    }
    r = client.post("/audio/transcriptions", json=payload, headers=_auth_headers())
    assert r.status_code == 200
    assert r.json()["text"].startswith("[mock]")


def test_json_body_raw_base64(client):
    b64 = base64.b64encode(b"\x00\x01fake").decode()
    payload = {
        "input_audio": {"data": b64, "format": "wav"},
    }
    r = client.post("/audio/transcriptions", json=payload, headers=_auth_headers())
    assert r.status_code == 200
    assert r.json()["text"].startswith("[mock]")


def test_json_body_missing_data(client):
    payload = {"input_audio": {"format": "wav"}}
    r = client.post("/audio/transcriptions", json=payload, headers=_auth_headers())
    assert r.status_code == 400


def test_no_body(client):
    r = client.post("/audio/transcriptions", headers=_auth_headers())
    assert r.status_code == 400


def test_invalid_json_body(client):
    r = client.post(
        "/audio/transcriptions",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        content="{not valid json",
    )
    assert r.status_code == 400


def test_unsupported_response_format(client):
    files = {"file": ("a.wav", b"x", "audio/wav")}
    data = {"response_format": "bogus"}
    r = client.post("/audio/transcriptions", files=files, data=data, headers=_auth_headers())
    assert r.status_code == 400


def test_stream_not_supported(client):
    files = {"file": ("a.wav", b"x", "audio/wav")}
    data = {"stream": "true"}
    r = client.post("/audio/transcriptions", files=files, data=data, headers=_auth_headers())
    assert r.status_code == 501


def test_upload_too_large(client, monkeypatch):
    # Force a tiny max upload size (in MB) for the test
    monkeypatch.setenv("DASHSCOPE_PROXY_MAX_UPLOAD_MB", "0.000001")
    from dashscope_transcription_proxy.settings import get_settings

    get_settings.cache_clear()
    files = {"file": ("a.wav", b"x" * 2048, "audio/wav")}
    r = client.post("/audio/transcriptions", files=files, headers=_auth_headers())
    get_settings.cache_clear()
    assert r.status_code == 413


def test_srt_format(client):
    files = {"file": ("a.wav", b"x", "audio/wav")}
    data = {"response_format": "srt"}
    r = client.post("/audio/transcriptions", files=files, data=data, headers=_auth_headers())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-subrip")


def test_vtt_format(client):
    files = {"file": ("a.wav", b"x", "audio/wav")}
    data = {"response_format": "vtt"}
    r = client.post("/audio/transcriptions", files=files, data=data, headers=_auth_headers())
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/vtt")
    assert r.text.startswith("WEBVTT")
