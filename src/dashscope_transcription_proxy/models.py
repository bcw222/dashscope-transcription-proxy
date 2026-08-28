"""OpenAI-compatible response models and DashScope → OpenAI mapping."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Supported response formats
# ---------------------------------------------------------------------------

SUPPORTED_RESPONSE_FORMATS = frozenset({"json", "text", "verbose_json", "srt", "vtt"})
"""Formats accepted by the endpoint. Unsupported values are rejected with 400."""


class TranscriptionWord(BaseModel):
    word: str
    start: float
    end: float


class TranscriptionSegment(BaseModel):
    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: list[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def _ms_to_s(ms: float) -> float:
    return round(ms / 1000.0, 3)


def _format_timestamp_srt(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    return _format_timestamp_srt(seconds).replace(",", ".")


def build_response(
    *,
    text: str,
    response_format: str = "json",
    dashscope_output: dict[str, Any] | None = None,
) -> str | dict[str, Any]:
    """Map DashScope result → OpenAI-compatible response.

    ``text``            → plain string
    ``json``            → ``{"text": ...}``
    ``verbose_json``    → ``text`` + words/segments/duration/language/usage
    ``srt`` / ``vtt``   → subtitle text (word-level timestamps required)
    """
    if response_format == "text":
        return text

    result: dict[str, Any] = {"text": text}

    if response_format in ("json", None, ""):
        return result

    ds = dashscope_output or {}
    annotations = _extract_annotations(ds)
    words_raw = annotations.get("words", [])
    words = [
        TranscriptionWord(
            word=w.get("text", ""),
            start=_ms_to_s(w.get("begin_time", 0)),
            end=_ms_to_s(w.get("end_time", 0)),
        )
        for w in words_raw
    ]

    # ---- srt / vtt ---------------------------------------------------------
    if response_format in ("srt", "vtt"):
        ts_fmt = _format_timestamp_srt if response_format == "srt" else _format_timestamp_vtt
        blocks: list[str] = []
        for i, w in enumerate(words, start=1):
            blocks.append(
                f"{i}\n{ts_fmt(w.start)} --> {ts_fmt(w.end)}\n{w.word}"
            )
        if response_format == "srt":
            return "\n\n".join(blocks)
        return "WEBVTT\n\n" + "\n\n".join(blocks)

    # ---- verbose_json ------------------------------------------------------
    result["language"] = annotations.get("language", "")
    result["duration"] = 0.0
    result["segments"] = []
    result["words"] = [w.model_dump() for w in words]

    usage = ds.get("usage", {})
    audio_seconds = usage.get("audio_seconds")
    if audio_seconds is not None and audio_seconds > 0:
        result["usage"] = {"type": "duration", "seconds": audio_seconds}
    elif usage:
        result["usage"] = {
            "type": "tokens",
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    if words:
        seg = TranscriptionSegment(
            id=0,
            start=words[0].start,
            end=words[-1].end,
            text=text,
        )
        result["segments"] = [seg.model_dump()]
        result["duration"] = words[-1].end

    return result


def _extract_annotations(dashscope_output: dict[str, Any]) -> dict[str, Any]:
    """Walk DashScope response to find the ``audio_info`` annotation."""
    try:
        anns = dashscope_output["output"]["choices"][0]["message"]["annotations"]
        for a in anns:
            if isinstance(a, dict) and a.get("type") == "audio_info":
                return a
    except (KeyError, IndexError, TypeError):
        pass
    return {}
