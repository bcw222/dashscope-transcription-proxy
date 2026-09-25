"""Local Qwen3-ASR backend — talks to ``qwen-asr-serve`` (vLLM) over its
OpenAI-compatible ``POST /v1/chat/completions``.

Design contract: returns the **same raw-dict shape** as :mod:`.client`
(DashScope), so ``app._extract_text`` and ``models.build_response`` are reused
unchanged::

    {"status_code", "code", "message", "request_id",
     "output": {"choices": [{"message": {"role", "content", "annotations"}}]},
     "usage": {...}}

Known limitations versus the DashScope backend (see README):

* **No word-level timestamps** — the vLLM chat surface does not return them, so
  ``response_format=srt|vtt`` is rejected upstream in ``app`` with 400.
* ``language`` **cannot be forced** — Qwen3-ASR auto-detects. The parameter is
  accepted (for signature compatibility) and logged, not forwarded.
* The caller's ``model`` field is **ignored**; ``vllm_model`` is always used,
  because downstream clients typically send a DashScope model id
  (``qwen3-asr-flash``) that does not exist on the local server.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from .settings import Settings

logger = logging.getLogger(__name__)

# Qwen3-ASR special tokens, e.g. "<|zh|>识别文本"
_SPECIAL_TOKENS_RE = re.compile(r"^(?:<\|[^|>]*\|>)+\s*")
# Explicit "language: zh" style prefixes
_LANG_PREFIX_RE = re.compile(
    r"^\s*(?:language|lang|语种|语言)\s*[:：]\s*([A-Za-z\u4e00-\u9fff_-]+)\s*[\n,，]?\s*",
    re.IGNORECASE,
)


def parse_qwen_output(content: str, mode: str = "auto") -> tuple[str, str]:
    """Split Qwen3-ASR's raw completion into ``(language, text)``.

    ``mode``:
      ``auto``     — official ``qwen_asr.parse_asr_output`` if importable,
                     else the conservative heuristic below, else passthrough.
      ``official`` — require the official parser (raises if unavailable).
      ``none``     — passthrough: ``("", content)``.

    The heuristic only strips **unambiguous** markers (leading ``<|…|>`` special
    tokens, or an explicit ``language: X`` prefix). Anything else is returned
    untouched — a wrong guess here would silently corrupt transcripts, so the
    fallback is deliberately inert.
    """
    if not content:
        return "", ""

    if mode == "none":
        return "", content

    if mode in ("auto", "official"):
        try:
            from qwen_asr import parse_asr_output  # type: ignore[import-not-found]
        except ImportError:
            if mode == "official":
                raise
        else:
            try:
                lang, text = parse_asr_output(content)
                return (lang or ""), (text if text is not None else content)
            except Exception:  # noqa: BLE001 — never let parsing kill a request
                logger.warning("qwen_asr.parse_asr_output failed; using heuristic", exc_info=True)
                if mode == "official":
                    raise

    # --- conservative heuristic -------------------------------------------
    text = content
    lang = ""

    m = _SPECIAL_TOKENS_RE.match(text)
    if m:
        token = m.group(0).strip()
        inner = token.strip("<|>").strip()
        # Only treat it as a language tag if it looks like one (short, no spaces)
        if inner and len(inner) <= 24 and " " not in inner:
            lang = inner
        text = text[m.end():]

    m = _LANG_PREFIX_RE.match(text)
    if m:
        lang = lang or m.group(1)
        text = text[m.end():]

    return lang, text.strip() or content.strip()


class VLLMTranscriptionClient:
    """OpenAI-compatible client for a locally served Qwen3-ASR (vLLM)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._http: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._settings.vllm_base_url.rstrip("/"),
                timeout=httpx.Timeout(self._settings.vllm_timeout, connect=15.0),
            )
            logger.info(
                "vLLM backend ready: base_url=%s model=%s",
                self._settings.vllm_base_url, self._settings.vllm_model,
            )

    async def stop(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # -- public API (same signature as TranscriptionClient) ------------------

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
        """Transcribe via the local server; return a DashScope-shaped raw dict.

        ``api_key`` is the **downstream** caller's key and is not forwarded —
        the local server is authenticated with ``settings.vllm_api_key``.
        ``language`` / ``model`` / ``enable_*`` are accepted for signature
        compatibility; see module docstring for what is actually honoured.
        """
        assert self._http is not None, "client.start() must be called first"
        s = self._settings

        if language:
            logger.debug("language=%r ignored on vllm backend (auto-detect only)", language)
        if model and model != s.vllm_model:
            logger.debug("caller model=%r overridden by vllm_model=%r", model, s.vllm_model)
        if enable_itn is not None or enable_lid:
            logger.debug("enable_itn/enable_lid are DashScope-only; ignored on vllm backend")

        messages: list[dict[str, Any]] = []
        if context and s.vllm_passthrough_context:
            messages.append({"role": "system", "content": context})
        messages.append({
            "role": "user",
            "content": [{"type": "audio_url", "audio_url": {"url": audio_ref}}],
        })

        payload = {
            "model": s.vllm_model,
            "messages": messages,
            "max_tokens": s.vllm_max_tokens,
            "stream": False,
        }

        headers = {"Content-Type": "application/json"}
        if s.vllm_api_key:
            headers["Authorization"] = f"Bearer {s.vllm_api_key}"

        logger.info(
            "vLLM transcribe: model=%s audio_ref_len=%d context=%s",
            s.vllm_model, len(audio_ref), bool(context),
        )

        try:
            resp = await self._http.post("/chat/completions", json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"cannot reach local ASR server at {s.vllm_base_url}: {exc}") from exc

        if resp.status_code != 200:
            detail = resp.text[:500]
            logger.error("vLLM HTTP %s: %s", resp.status_code, detail)
            raise RuntimeError(f"local ASR server HTTP {resp.status_code}: {detail}")

        data = resp.json()
        return self._to_dashscope_shape(data)

    # -- internal ------------------------------------------------------------

    def _to_dashscope_shape(self, data: dict[str, Any]) -> dict[str, Any]:
        """Map an OpenAI chat completion → the DashScope raw dict shape."""
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        raw_content = message.get("content", "")
        if isinstance(raw_content, list):  # defensive: some servers return parts
            raw_content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in raw_content
            )

        lang, text = parse_qwen_output(raw_content, self._settings.qwen_output_parser)

        u = data.get("usage") or {}
        usage: dict[str, Any] = {
            "input_tokens": u.get("prompt_tokens", u.get("input_tokens", 0)),
            "output_tokens": u.get("completion_tokens", u.get("output_tokens", 0)),
            "total_tokens": u.get("total_tokens", 0),
        }

        return {
            "status_code": 200,
            "code": None,
            "message": None,
            "request_id": data.get("id", ""),
            "output": {
                "choices": [{
                    "finish_reason": choice.get("finish_reason"),
                    "index": choice.get("index", 0),
                    "message": {
                        "role": message.get("role", "assistant"),
                        "content": text,
                        # Synthesised so models.build_response can report the
                        # detected language. `words` stays empty — the vLLM chat
                        # surface gives no word-level timestamps.
                        "annotations": [{
                            "type": "audio_info",
                            "language": lang,
                            "words": [],
                        }],
                    },
                }],
            },
            "usage": usage,
        }
