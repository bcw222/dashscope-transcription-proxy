"""FastAPI proxy — OpenAI-compatible ``POST /audio/transcriptions`` endpoint.

Backed by DashScope (阿里云百炼) ASR via ``dashscope.MultiModalConversation``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import PlainTextResponse, Response

from .client import TranscriptionClient
from .models import SUPPORTED_RESPONSE_FORMATS, build_response
from .settings import get_settings

logger = logging.getLogger("dashscope-transcription-proxy")


def _get_client() -> TranscriptionClient:
    """Create the singleton client lazily (settings are read on first use)."""
    return TranscriptionClient(get_settings())


_client: TranscriptionClient | None = None


def _client_or_init() -> TranscriptionClient:
    global _client
    if _client is None:
        _client = _get_client()
    return _client


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = _client_or_init()
    s = get_settings()
    logger.info("Starting (model=%s local_mode=%s) …", s.model, s.local_mode)
    await client.start()
    yield
    await client.stop()
    logger.info("Shut down.")


app = FastAPI(title="DashScope Transcription Proxy", version="0.1.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# POST /audio/transcriptions
# ---------------------------------------------------------------------------

@app.post("/audio/transcriptions")
async def transcribe(
    request: Request,
    # multipart fields
    file: Annotated[UploadFile | None, File()] = None,
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[str | None, Form()] = None,
    stream: Annotated[bool | None, Form()] = None,
):
    """OpenAI-compatible transcription endpoint.

    Supports:
    - **multipart/form-data** — ``file`` field (audio upload)
    - **application/json** — ``input_audio`` object with ``data``
    """
    if stream:
        raise HTTPException(501, "Streaming is not supported")

    api_key = _extract_api_key(request)
    if not api_key:
        raise HTTPException(401, "Missing Authorization: Bearer <key> header")

    content_type: str = request.headers.get("content-type", "")

    if "application/json" in content_type:
        return await _handle_json(
            request, api_key=api_key, model=model, language=language,
            prompt=prompt, response_format=response_format, stream=stream,
        )

    if file is not None:
        return await _handle_multipart(
            file=file, api_key=api_key, model=model, language=language,
            prompt=prompt, response_format=response_format,
        )

    raise HTTPException(400, "Provide file (multipart) or JSON body with input_audio")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    s = get_settings()
    return {"status": "ok", "model": s.model, "local_mode": s.local_mode}


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def _handle_json(
    request: Request,
    *,
    api_key: str,
    model: str | None,
    language: str | None,
    prompt: str | None,
    response_format: str | None,
    stream: bool | None,
) -> Any:
    try:
        body: dict = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"Invalid JSON body: {exc}") from exc

    model = body.get("model", model)
    language = body.get("language", language)
    prompt = body.get("prompt", prompt)
    response_format = body.get("response_format") or response_format
    if body.get("stream") or stream:
        raise HTTPException(501, "Streaming is not supported")

    input_audio: dict = body.get("input_audio", {})
    data: str = input_audio.get("data", "")
    if not data:
        raise HTTPException(400, "Missing input_audio.data")

    # DashScope accepts data: URIs and http(s) URLs as-is.
    # Wrap raw base64 strings.
    if not data.startswith(("data:", "http://", "https://")):
        fmt = input_audio.get("format", "wav")
        data = f"data:audio/{fmt};base64,{data}"

    return await _do_transcribe(
        audio_ref=data,
        api_key=api_key,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
    )


async def _handle_multipart(
    file: UploadFile,
    *,
    api_key: str,
    model: str | None,
    language: str | None,
    prompt: str | None,
    response_format: str | None,
) -> Any:
    s = get_settings()
    audio_bytes = await file.read()
    if s.max_upload_mb > 0 and len(audio_bytes) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"Audio file exceeds {s.max_upload_mb}MB limit")

    mime = _safe_mime(file.content_type, file.filename)
    b64 = base64.b64encode(audio_bytes).decode()
    data_uri = f"data:{mime};base64,{b64}"

    return await _do_transcribe(
        audio_ref=data_uri,
        api_key=api_key,
        model=model,
        language=language,
        prompt=prompt,
        response_format=response_format,
    )


async def _do_transcribe(
    *,
    audio_ref: str,
    api_key: str,
    model: str | None,
    language: str | None,
    prompt: str | None,
    response_format: str | None,
) -> Any:
    s = get_settings()
    lang = language or s.default_language
    fmt = response_format or "json"

    if fmt not in SUPPORTED_RESPONSE_FORMATS:
        raise HTTPException(400, f"Unsupported response_format: {fmt}")

    if s.local_mode:
        response = _mock_result(fmt)
    else:
        try:
            ds_raw = await _client_or_init().transcribe(
                audio_ref=audio_ref,
                api_key=api_key,
                context=prompt,
                language=lang,
                model=model,
            )
        except Exception as exc:
            logger.exception("DashScope call failed")
            raise HTTPException(502, f"DashScope error: {exc}") from exc

        # Extract text (handle multiple DashScope response shapes)
        text = _extract_text(ds_raw)
        if not text:
            logger.warning(
                "Cannot extract text from DashScope response: status_code=%s code=%s message=%s request_id=%s output=%s",
                ds_raw.get("status_code"),
                ds_raw.get("code"),
                ds_raw.get("message"),
                ds_raw.get("request_id"),
                json.dumps(ds_raw.get("output"), ensure_ascii=False)[:2000],
            )

        response = build_response(text=text, response_format=fmt, dashscope_output=ds_raw)

        # Persist raw result
        if s.results_dir:
            os.makedirs(s.results_dir, exist_ok=True)
            rid = str(uuid.uuid4())
            with open(os.path.join(s.results_dir, f"{rid}.json"), "w", encoding="utf-8") as f:
                json.dump(ds_raw, f, ensure_ascii=False, indent=2)

    if fmt == "text":
        return PlainTextResponse(str(response))

    if fmt in ("srt", "vtt"):
        media = "application/x-subrip" if fmt == "srt" else "text/vtt"
        return Response(content=str(response), media_type=media)

    return response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text(ds_raw: dict[str, Any]) -> str:
    """Extract recognised text across DashScope response shapes.

    - qwen3-asr-flash: ``output.choices[0].message.content`` (str or list)
    - fun-asr-flash / qwen-audio-3.0-asr-flash: ``output.output.sentence.text``
      or ``output.text``
    """
    output = ds_raw.get("output") or {}

    # Shape 1: choices[].message.content
    choices = output.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
            return "".join(parts)

    # Shape 2: output.output.sentence.text
    inner = output.get("output")
    if isinstance(inner, dict):
        sentence = inner.get("sentence")
        if isinstance(sentence, dict):
            return str(sentence.get("text", ""))

    # Shape 3: output.text (top-level of output)
    text = output.get("text")
    if isinstance(text, str):
        return text

    return ""


def _extract_api_key(request: Request) -> str:
    """Extract ``Bearer <key>`` from the Authorization header."""
    auth: str | None = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    # Fallback: check X-API-Key header
    key = request.headers.get("x-api-key", "")
    if auth and not key:
        logger.warning("Authorization header present but not Bearer; falling back to X-API-Key")
    return key


def _safe_mime(content_type: str | None, filename: str | None) -> str:
    """Prefer filename-based MIME when content_type is vague."""
    if content_type and content_type != "application/octet-stream":
        return content_type
    return _guess_mime(filename)


def _guess_mime(filename: str | None) -> str:
    if not filename:
        return "audio/wav"
    ext = os.path.splitext(filename)[1].lower()
    return {
        ".wav": "audio/wav", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
        ".oga": "audio/ogg", ".flac": "audio/flac", ".m4a": "audio/mp4",
        ".aac": "audio/aac", ".webm": "audio/webm",
    }.get(ext, "audio/wav")


def _mock_result(fmt: str) -> Any:
    text = "[mock] 这是模拟的语音识别结果。"
    return build_response(text=text, response_format=fmt)
