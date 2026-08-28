"""DashScope ASR client — wraps ``dashscope.MultiModalConversation.call``."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import dashscope
from dashscope import MultiModalConversation

from .settings import Settings

logger = logging.getLogger(__name__)


class TranscriptionClient:
    """Calls DashScope ``MultiModalConversation`` for ASR.

    DashScope SDK is synchronous, so each call runs in a dedicated thread
    pool to avoid blocking the asyncio event loop.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._executor: ThreadPoolExecutor | None = None
        if settings.base_http_api_url:
            dashscope.base_http_api_url = settings.base_http_api_url

    async def start(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=4)

    async def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    # -- public API ----------------------------------------------------------

    async def transcribe(
        self,
        *,
        audio_ref: str,
        api_key: str,
        context: str | None = None,
        language: str | None = None,
        model: str | None = None,
        enable_itn: bool | None = None,
        enable_lid: bool | None = None,
    ) -> dict[str, Any]:
        """Transcribe audio.

        ``audio_ref`` is what DashScope accepts in ``{"audio": ...}``:
        a data: URI, a ``file://`` path, or a public URL.
        ``api_key`` **must** be provided — it comes from the downstream
        request's ``Authorization: Bearer <key>`` header.

        Returns the DashScope **raw dict** so callers can inspect annotations.
        """
        asr_opts: dict[str, Any] = {}
        if enable_itn is None:
            enable_itn = self._settings.enable_itn
        if enable_lid is None:
            enable_lid = self._settings.enable_lid

        asr_opts["enable_itn"] = enable_itn
        if enable_lid:
            asr_opts["enable_lid"] = True
        if language:
            asr_opts["language"] = language

        messages: list[dict[str, Any]] = []
        if context:
            messages.append({"role": "system", "content": [{"text": context}]})
        messages.append({"role": "user", "content": [{"audio": audio_ref}]})

        model_id = model or self._settings.model
        logger.info(
            "DashScope transcribe: model=%s language=%s audio_ref_len=%d",
            model_id, language, len(audio_ref),
        )

        raw = await self._run_in_thread(
            MultiModalConversation.call,
            model=model_id,
            messages=messages,
            result_format="message",
            asr_options=asr_opts,
            api_key=api_key,
        )

        status_code = raw.get("status_code")
        if status_code and status_code != 200:
            logger.error(
                "DashScope error: status_code=%s code=%s message=%s request_id=%s",
                status_code, raw.get("code"), raw.get("message"), raw.get("request_id"),
            )
            detail = f"{raw.get('code') or ''} {raw.get('message') or ''}".strip()
            raise RuntimeError(f"DashScope HTTP {status_code}: {detail}")

        return raw

    # -- internal ------------------------------------------------------------

    async def _run_in_thread(self, fn, **kwargs: Any) -> dict[str, Any]:
        """Run a sync DashScope call in the thread pool."""
        assert self._executor is not None, "client.start() must be called first"
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(self._executor, lambda: fn(**kwargs))
        return _dashscope_response_to_dict(response)


def _dashscope_response_to_dict(response: Any) -> dict[str, Any]:
    """Convert DashScope response object → plain dict (preserve diagnostics)."""
    result: dict[str, Any] = {
        "status_code": getattr(response, "status_code", None),
        "code": getattr(response, "code", None),
        "message": getattr(response, "message", None),
        "request_id": getattr(response, "request_id", ""),
        "output": {},
        "usage": {},
    }

    output = getattr(response, "output", None)
    if output is not None and hasattr(output, "choices"):
        choices = []
        for c in output.choices:
            cdict: dict[str, Any] = {
                "finish_reason": getattr(c, "finish_reason", None),
                "index": getattr(c, "index", 0),
            }
            msg = getattr(c, "message", None)
            if msg:
                mdict: dict[str, Any] = {
                    "role": getattr(msg, "role", "assistant"),
                    "content": _normalize_content(getattr(msg, "content", "")),
                }
                anns = getattr(msg, "annotations", None)
                if anns:
                    mdict["annotations"] = [_annotation_to_dict(a) for a in anns]
                cdict["message"] = mdict
            choices.append(cdict)
        result["output"] = {"choices": choices}
    elif isinstance(output, dict):
        result["output"] = output
    elif output is not None:
        # fun-asr-flash / qwen-audio-3.0-asr-flash:
        # output.output.sentence.text  /  output.text
        inner = getattr(output, "output", None)
        out_dict: dict[str, Any] = {}
        if inner is not None:
            out_dict["output"] = _to_plain(inner)
        text = getattr(output, "text", None)
        if text is not None:
            out_dict["text"] = text
        result["output"] = out_dict

    result["usage"] = _dict_usage(response)
    return result


def _dict_usage(response: Any) -> dict[str, Any]:
    u = getattr(response, "usage", None)
    if u is None:
        return {}
    result: dict[str, Any] = {
        "input_tokens": getattr(u, "input_tokens", 0),
        "output_tokens": getattr(u, "output_tokens", 0),
        "total_tokens": getattr(u, "total_tokens", 0),
    }
    audio_seconds = getattr(u, "audio_seconds", None)
    if audio_seconds is not None:
        result["audio_seconds"] = audio_seconds
    return result


def _annotation_to_dict(ann: Any) -> dict[str, Any]:
    if hasattr(ann, "__dict__"):
        return ann.__dict__
    return ann


def _to_plain(obj: Any) -> Any:
    """Recursively convert SDK objects to plain dict/list/str."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: _to_plain(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def _normalize_content(content: Any) -> str:
    """DashScope message content may be a str or a list of parts.

    Examples:
        "识别文本"
        [{"text": "识别文本"}]
        [{"text": "你"}, {"text": "好"}]

    Normalise to a single plain string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                t = getattr(item, "text", None)
                if t is not None:
                    parts.append(str(t))
        return "".join(parts)
    # fallback: coerce
    return str(content)
