"""Unit tests for the local (vLLM) backend: output parsing + response mapping."""

from __future__ import annotations

import httpx
import pytest

from dashscope_transcription_proxy.local_client import (
    VLLMTranscriptionClient,
    parse_qwen_output,
)
from dashscope_transcription_proxy.settings import Settings


# --- parse_qwen_output -------------------------------------------------------

def test_parse_empty():
    assert parse_qwen_output("", "auto") == ("", "")


def test_parse_none_mode_passthrough():
    assert parse_qwen_output("<|zh|>你好", "none") == ("", "<|zh|>你好")


def test_parse_plain_text_has_no_language():
    assert parse_qwen_output("今天讲傅里叶变换。", "auto") == ("", "今天讲傅里叶变换。")


def test_parse_strips_special_token_language():
    lang, text = parse_qwen_output("<|zh|>今天讲傅里叶变换。", "auto")
    assert lang == "zh"
    assert text == "今天讲傅里叶变换。"


def test_parse_strips_explicit_language_prefix():
    lang, text = parse_qwen_output("language: zh\n今天讲傅里叶变换。", "auto")
    assert lang == "zh"
    assert text == "今天讲傅里叶变换。"


def test_parse_heuristic_is_inert_on_ambiguous_input():
    """A long/spacey leading marker is stripped as a format token but must NOT
    be interpreted as a language tag."""
    lang, text = parse_qwen_output("<|this is not a language tag at all|>正文内容", "auto")
    assert lang == ""
    assert text == "正文内容"


def test_parse_official_mode_raises_when_unavailable():
    with pytest.raises(ImportError):
        parse_qwen_output("你好", "official")


# --- VLLMTranscriptionClient -------------------------------------------------

def _completion(content: str, rid: str = "chatcmpl-abc") -> dict:
    return {
        "id": rid,
        "object": "chat.completion",
        "model": "qwen3-asr",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": content},
        }],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    }


def _make_client(handler, **over) -> VLLMTranscriptionClient:
    s = Settings(backend="vllm", qwen_output_parser="none", **over)
    c = VLLMTranscriptionClient(s)
    # Inject a mock transport instead of start() (which opens a real client)
    c._http = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=s.vllm_base_url.rstrip("/"),
    )
    return c


@pytest.mark.asyncio
async def test_transcribe_maps_to_dashscope_shape():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion("识别结果"))

    c = _make_client(handler)
    raw = await c.transcribe(audio_ref="data:audio/wav;base64,AAA", api_key="sk-downstream")
    await c.stop()

    # Reused by app._extract_text (shape 1) without modification
    assert raw["output"]["choices"][0]["message"]["content"] == "识别结果"
    assert raw["status_code"] == 200
    assert raw["request_id"] == "chatcmpl-abc"
    # Reused by models.build_response for verbose_json usage
    assert raw["usage"]["input_tokens"] == 120
    assert raw["usage"]["output_tokens"] == 30
    assert raw["usage"]["total_tokens"] == 150
    # Annotations synthesised so build_response can report language
    ann = raw["output"]["choices"][0]["message"]["annotations"][0]
    assert ann["type"] == "audio_info"
    assert ann["words"] == []

    # httpx joins base_url (…/v1) with the relative path ⇒ full path carries /v1
    assert seen["path"] == "/v1/chat/completions"
    assert seen["body"]["model"] == "qwen3-asr"
    assert seen["body"]["stream"] is False
    part = seen["body"]["messages"][-1]["content"][0]
    assert part["type"] == "audio_url"
    assert part["audio_url"]["url"].startswith("data:audio/wav;base64,")


@pytest.mark.asyncio
async def test_downstream_key_is_not_forwarded():
    """The caller's DashScope key must not leak to the local server."""
    headers: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        headers.update(dict(request.headers))
        return httpx.Response(200, json=_completion("x"))

    c = _make_client(handler)
    await c.transcribe(audio_ref="data:audio/wav;base64,AAA", api_key="sk-secret-downstream")
    await c.stop()

    assert "authorization" not in headers  # vllm_api_key empty → no auth header


@pytest.mark.asyncio
async def test_configured_vllm_key_is_used():
    headers: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        headers.update(dict(request.headers))
        return httpx.Response(200, json=_completion("x"))

    c = _make_client(handler, vllm_api_key="sk-local")
    await c.transcribe(audio_ref="data:audio/wav;base64,AAA", api_key="sk-downstream")
    await c.stop()

    assert headers.get("authorization") == "Bearer sk-local"


@pytest.mark.asyncio
async def test_prompt_becomes_system_message():
    import json
    body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body.update(json.loads(request.content))
        return httpx.Response(200, json=_completion("x"))

    c = _make_client(handler)
    await c.transcribe(
        audio_ref="data:audio/wav;base64,AAA", api_key="k",
        context="傅里叶变换 频谱 卷积",
    )
    await c.stop()

    assert body["messages"][0] == {"role": "system", "content": "傅里叶变换 频谱 卷积"}


@pytest.mark.asyncio
async def test_context_can_be_disabled():
    import json
    body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body.update(json.loads(request.content))
        return httpx.Response(200, json=_completion("x"))

    c = _make_client(handler, vllm_passthrough_context=False)
    await c.transcribe(audio_ref="data:audio/wav;base64,AAA", api_key="k", context="热词")
    await c.stop()

    assert all(m["role"] != "system" for m in body["messages"])


@pytest.mark.asyncio
async def test_http_error_raises_runtimeerror():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="model loading")

    c = _make_client(handler)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        await c.transcribe(audio_ref="data:audio/wav;base64,AAA", api_key="k")
    await c.stop()


@pytest.mark.asyncio
async def test_unreachable_raises_runtimeerror():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    c = _make_client(handler)
    with pytest.raises(RuntimeError, match="cannot reach local ASR server"):
        await c.transcribe(audio_ref="data:audio/wav;base64,AAA", api_key="k")
    await c.stop()


@pytest.mark.asyncio
async def test_list_content_is_joined():
    """Defensive: some servers return content as a list of parts."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_completion([{"text": "你"}, {"text": "好"}]))

    c = _make_client(handler)
    raw = await c.transcribe(audio_ref="data:audio/wav;base64,AAA", api_key="k")
    await c.stop()

    assert raw["output"]["choices"][0]["message"]["content"] == "你好"


def test_extract_text_accepts_local_backend_shape():
    """The existing DashScope-shaped extractor must work unchanged."""
    from dashscope_transcription_proxy.app import _extract_text

    raw = {
        "output": {"choices": [{"message": {
            "role": "assistant",
            "content": "本地识别结果",
            "annotations": [{"type": "audio_info", "language": "zh", "words": []}],
        }}]},
    }
    assert _extract_text(raw) == "本地识别结果"
