"""Streaming transcription bridge — OpenAI Realtime dialect ↔ DashScope Qwen-ASR-Realtime.

``POST /audio/transcriptions`` covers one-shot files. This module adds the
streaming half: a WebSocket endpoint that downstream clients can drive with the
OpenAI Realtime *transcription* event dialect, while the proxy speaks
DashScope's ``qwen3-asr-flash-realtime`` protocol upstream.

Why a translation layer is needed (the two dialects are close but not equal):

=============================== ========================================== =========================================
Direction                       Client (OpenAI Realtime dialect)             Upstream (DashScope Qwen-ASR-Realtime)
=============================== ========================================== =========================================
connect                         —                                          ``wss://…/api-ws/v1/realtime?model=…``
configure                       ``session.update``                         ``session.update`` (+ mandatory ``event_id``)
audio                           ``input_audio_buffer.append``              ``input_audio_buffer.append`` (PCM only)
manual segmentation             ``input_audio_buffer.commit``              ``input_audio_buffer.commit``
partial transcript              ``…transcription.delta`` (incremental)     ``…transcription.text`` (``text``+``stash``)
final transcript                ``…transcription.completed``               ``…transcription.completed``
flush before close              ``session.finish``                         ``session.finish`` → ``session.finished``
=============================== ========================================== =========================================

Two behaviours deserve emphasis:

* **Partials are re-shaped.** DashScope sends a confirmed prefix (``text``) plus
  a revisable draft (``stash``); OpenAI clients expect an incremental ``delta``.
  The bridge tracks the cumulative prefix per item and emits only the new tail
  as ``delta``, while also passing the cumulative string through as ``text`` so
  clients can resync after a revision.
* **``session.finish`` is not optional.** DashScope discards the in-progress
  item when the socket closes without it, so the bridge forwards a finish on
  client disconnect too and waits (bounded) for ``session.finished``.

Audio format: telephony sources such as OpenClaw's Talk/dictation relay send
8 kHz G.711 µ-law. DashScope accepts PCM at 16 kHz or 8 kHz, so G.711 payloads
are decoded to PCM16 here (see :mod:`.audio`) and the session keeps the
caller's sample rate.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import uuid
from typing import Any, Protocol
from urllib.parse import urlencode, urlparse, urlunparse

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .audio import decode_g711, normalize_input_format
from .settings import Settings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Client-side event names (OpenAI Realtime transcription dialect).
EV_APPEND = "input_audio_buffer.append"
EV_COMMIT = "input_audio_buffer.commit"
EV_SESSION_UPDATE = "session.update"
EV_SESSION_FINISH = "session.finish"
EV_SESSION_CLOSE = "session.close"

# Upstream transcription event names shared by both dialects.
_UPSTREAM_TEXT = "conversation.item.input_audio_transcription.text"
_DOWNSTREAM_DELTA = "conversation.item.input_audio_transcription.delta"
_COMPLETED = "conversation.item.input_audio_transcription.completed"

# Sentinel distinguishing "client never set turn_detection" from "client set it to null".
_UNSET = object()


def _event_id() -> str:
    return f"event_{uuid.uuid4().hex}"


def derive_realtime_ws_url(base_http_api_url: str) -> str:
    """Derive the realtime WebSocket URL from the HTTP API base URL.

    ``https://dashscope.aliyuncs.com/api/v1`` →
    ``wss://dashscope.aliyuncs.com/api-ws/v1/realtime``
    """
    parsed = urlparse(base_http_api_url.strip() or "https://dashscope.aliyuncs.com/api/v1")
    scheme = "ws" if parsed.scheme == "http" else "wss"
    path = parsed.path.rstrip("/")
    if path.endswith("/api/v1"):
        ws_path = f"{path[: -len('/api/v1')]}/api-ws/v1/realtime"
    elif path.endswith("/api-ws/v1"):
        ws_path = f"{path}/realtime"
    elif path.endswith("/realtime"):
        ws_path = path
    else:
        ws_path = f"{path}/realtime" if path else "/api-ws/v1/realtime"
    return urlunparse((scheme, parsed.netloc, ws_path, "", "", ""))


class Upstream(Protocol):
    """Minimal duplex JSON channel to the realtime ASR service."""

    async def send(self, payload: dict[str, Any]) -> None: ...

    async def recv(self) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class WebsocketUpstream:
    """Real DashScope connection over ``websockets``."""

    def __init__(self, connection: Any) -> None:
        self._connection = connection

    @classmethod
    async def connect(
        cls,
        *,
        url: str,
        api_key: str,
        model: str,
        open_timeout: float,
        proxy: str | bool | None = True,
    ) -> "WebsocketUpstream":
        import websockets  # imported lazily: only the streaming endpoint needs it

        query = urlencode({"model": model})
        target = f"{url}?{query}" if "?" not in url else f"{url}&{query}"
        headers = {"Authorization": f"Bearer {api_key}", "user-agent": "dashscope-transcription-proxy"}
        connection = await websockets.connect(
            target,
            additional_headers=headers,
            max_size=16 * 1024 * 1024,
            open_timeout=open_timeout,
            # ``True`` = websockets' own env detection; ``None`` = direct; a string = that proxy.
            proxy=proxy,
        )
        logger.info("Realtime upstream connected (model=%s)", model)
        return cls(connection)

    async def send(self, payload: dict[str, Any]) -> None:
        await self._connection.send(json.dumps(payload, ensure_ascii=False))

    async def recv(self) -> dict[str, Any]:
        raw = await self._connection.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        return json.loads(raw)

    async def close(self) -> None:
        await self._connection.close()


class MockUpstream:
    """In-process DashScope stand-in for ``local_mode`` (and offline tests).

    Emits the same event sequence a real session would: ``session.created`` on
    connect, ``session.updated`` after a config change, and for each committed
    buffer a VAD-style burst ending in a final transcript.
    """

    def __init__(self, *, model: str, sample_rate: int) -> None:
        self._model = model
        self._sample_rate = sample_rate
        self._events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._pcm_bytes = 0
        self._events.put_nowait(
            {
                "type": "session.created",
                "event_id": _event_id(),
                "session": {
                    "id": f"sess_{uuid.uuid4().hex[:12]}",
                    "object": "realtime.session",
                    "model": model,
                    "modalities": ["text"],
                    "input_audio_format": "pcm16",
                    "turn_detection": {"type": "server_vad", "threshold": 0.0, "silence_duration_ms": 400},
                },
            }
        )

    @property
    def pcm_bytes(self) -> int:
        """Decoded PCM bytes received so far (asserted by tests)."""
        return self._pcm_bytes

    async def send(self, payload: dict[str, Any]) -> None:
        event_type = payload.get("type")
        if event_type == EV_SESSION_UPDATE:
            session = payload.get("session") or {}
            self._events.put_nowait(
                {"type": "session.updated", "event_id": _event_id(), "session": {"model": self._model, **session}}
            )
            return
        if event_type == EV_APPEND:
            try:
                self._pcm_bytes += len(base64.b64decode(payload.get("audio") or "", validate=True))
            except (binascii.Error, ValueError):
                self._events.put_nowait(
                    {
                        "type": "error",
                        "event_id": _event_id(),
                        "error": {"type": "invalid_request_error", "code": "invalid_audio", "message": "audio is not valid base64"},
                    }
                )
            return
        if event_type == EV_COMMIT:
            item_id = f"item_{uuid.uuid4().hex[:12]}"
            seconds = self._pcm_bytes / (self._sample_rate * 2) or 0.1
            self._pcm_bytes = 0
            for event in (
                {"type": "input_audio_buffer.committed", "event_id": _event_id(), "item_id": item_id},
                {
                    "type": "conversation.item.created",
                    "event_id": _event_id(),
                    "item": {"id": item_id, "object": "realtime.item", "type": "message", "status": "completed", "role": "user"},
                },
                {
                    "type": _UPSTREAM_TEXT,
                    "event_id": _event_id(),
                    "item_id": item_id,
                    "content_index": 0,
                    "language": "zh",
                    "emotion": "neutral",
                    "text": "",
                    "stash": "[mock] ",
                },
                {
                    "type": _UPSTREAM_TEXT,
                    "event_id": _event_id(),
                    "item_id": item_id,
                    "content_index": 0,
                    "language": "zh",
                    "emotion": "neutral",
                    "text": "[mock] ",
                    "stash": "流式识别结果",
                },
                {
                    "type": _COMPLETED,
                    "event_id": _event_id(),
                    "item_id": item_id,
                    "content_index": 0,
                    "language": "zh",
                    "emotion": "neutral",
                    "transcript": f"[mock] 流式识别结果（{seconds:.1f}s 音频）",
                },
            ):
                self._events.put_nowait(event)
            return
        if event_type == EV_SESSION_FINISH:
            self._events.put_nowait({"type": "session.finished", "event_id": _event_id()})

    async def recv(self) -> dict[str, Any]:
        return await self._events.get()

    async def close(self) -> None:  # pragma: no cover - nothing to release
        return None


class RealtimeBridge:
    """Relays one client WebSocket to one DashScope realtime session."""

    def __init__(
        self,
        client: WebSocket,
        *,
        settings: Settings,
        api_key: str,
        model: str,
        input_format: str,
        sample_rate: int,
        language: str,
    ) -> None:
        self._client = client
        self._settings = settings
        self._api_key = api_key
        self._model = model
        self._input_format = input_format
        self._sample_rate = sample_rate
        self._language = language
        # Cumulative partial per item, so DashScope's prefix+draft becomes an OpenAI delta.
        self._prefixes: dict[str, str] = {}
        self._pending: list[dict[str, Any]] = []
        self._upstream_ready = asyncio.Event()
        self._finishing = False
        self._upstream_finished = False
        self._upstream_configured = False
        self._client_gone = False
        # ``_UNSET`` until the client sends an explicit ``turn_detection`` (including null).
        self._vad_override: Any = _UNSET

    # -- lifecycle -----------------------------------------------------------

    async def run(self) -> None:
        await self._client.accept()
        try:
            upstream = await self._connect_upstream()
        except Exception as exc:
            logger.exception("Realtime upstream connect failed")
            await self._send({"type": "error", "event_id": _event_id(), "error": _error_body("upstream_connect_failed", str(exc))})
            await self._close_client(code=1011)
            return

        flush_task = asyncio.create_task(self._flush_pending(upstream))
        upstream_task = asyncio.create_task(self._pump_upstream(upstream), name="realtime-upstream")
        client_task = asyncio.create_task(self._pump_client(upstream), name="realtime-client")
        tasks = (upstream_task, client_task, flush_task)
        try:
            done, _ = await asyncio.wait({upstream_task, client_task}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                error = task.exception()
                if error is not None:
                    logger.debug("Realtime pump %s ended: %r", task.get_name(), error)
        finally:
            # Orderly teardown: closing the client unblocks ``receive()``, and closing
            # the upstream unblocks ``recv()``, so both pumps return on their own.
            # Cancelling a task parked in ASGI I/O would surface as a cancellation to
            # the server (and starlette's TestClient) instead of a clean disconnect.
            await self._close_client(code=1000)
            budget = self._settings.realtime_finish_timeout + 1.0
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=budget,
                )
            except (asyncio.TimeoutError, TimeoutError):
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            await upstream.close()

    async def _connect_upstream(self) -> Upstream:
        if self._settings.local_mode:
            upstream: Upstream = MockUpstream(model=self._model, sample_rate=self._sample_rate)
        else:
            url = self._settings.realtime_ws_url or derive_realtime_ws_url(self._settings.base_http_api_url)
            logger.info("Realtime upstream connecting (model=%s sample_rate=%s)", self._model, self._sample_rate)
            upstream = await WebsocketUpstream.connect(
                url=url,
                api_key=self._api_key,
                model=self._model,
                open_timeout=self._settings.realtime_finish_timeout + 5.0,
                proxy=_resolve_proxy(self._settings),
            )
        # Wait for session.created before talking: DashScope rejects an update that
        # arrives before the session exists.
        first = await upstream.recv()
        await self._emit_session_created(first)
        self._upstream_ready.set()
        return upstream

    async def _ensure_upstream_configured(self, upstream: Upstream) -> None:
        """Send exactly one upstream ``session.update``, before the first audio.

        DashScope accepts a single configuration event per session — a second one is
        rejected with ``1007 session already started or finished or failed`` — so all
        client updates are merged into ``self`` and pushed once, lazily, when audio
        (or a commit) actually needs the session to be configured.
        """
        if self._upstream_configured:
            return
        self._upstream_configured = True
        await upstream.send({"event_id": _event_id(), "type": EV_SESSION_UPDATE, "session": self._session_config()})

    async def _finish_upstream(self, upstream: Upstream) -> None:
        """Best-effort flush so the final transcript is not discarded upstream."""
        try:
            if not self._finishing:
                await upstream.send({"event_id": _event_id(), "type": EV_SESSION_FINISH})
            await asyncio.wait_for(self._drain_until_finished(upstream), timeout=self._settings.realtime_finish_timeout)
        except Exception as exc:  # bounded best-effort flush; never fails the close path
            logger.debug("Realtime finish flush incomplete: %r", exc)

    async def _drain_until_finished(self, upstream: Upstream) -> None:
        while True:
            event = await upstream.recv()
            for outgoing in self._translate(event):
                await self._send(outgoing)
            if event.get("type") == "session.finished":
                self._upstream_finished = True
                return

    # -- client → upstream ---------------------------------------------------

    async def _pump_client(self, upstream: Upstream) -> None:
        try:
            while True:
                message = await self._client.receive()
                if message.get("type") == "websocket.disconnect":
                    logger.debug("Realtime client disconnected (code=%s)", message.get("code"))
                    break
                payload = message.get("bytes")
                if payload is not None:
                    # Bare binary frames are treated as audio (no event envelope).
                    await self._forward_audio(upstream, payload)
                    continue
                text = message.get("text")
                if text is None:
                    continue
                try:
                    event = json.loads(text)
                except json.JSONDecodeError as exc:
                    await self._send(
                        {"type": "error", "event_id": _event_id(), "error": _error_body("invalid_json", f"Invalid JSON frame: {exc}")}
                    )
                    continue
                await self._handle_client_event(upstream, event)
        except Exception as exc:  # client transport vanished mid-frame
            logger.warning("Realtime client pump ended: %s: %s", type(exc).__name__, exc)
        self._client_gone = True
        # DashScope discards the in-progress item when the socket closes without a
        # finish, so flush even on an abrupt disconnect.
        if not self._upstream_finished:
            await self._finish_upstream(upstream)

    async def _handle_client_event(self, upstream: Upstream, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "")
        if event_type == EV_SESSION_UPDATE:
            # Merge, acknowledge locally, and let the single deferred upstream push
            # carry it: forwarding a second update would kill the DashScope session.
            self._apply_client_session(event.get("session") or {})
            await self._send(
                {
                    "type": "session.updated",
                    "event_id": event.get("event_id") or _event_id(),
                    "session": {**self._public_session_fields(), "turn_detection": self._effective_turn_detection()},
                }
            )
            return
        if event_type == EV_APPEND:
            audio = event.get("audio")
            if not isinstance(audio, str) or not audio:
                await self._send(
                    {"type": "error", "event_id": _event_id(), "error": _error_body("invalid_request_error", "input_audio_buffer.append requires base64 'audio'")}
                )
                return
            try:
                raw = base64.b64decode(audio, validate=True)
            except (binascii.Error, ValueError) as exc:
                await self._send({"type": "error", "event_id": _event_id(), "error": _error_body("invalid_audio", f"audio is not valid base64: {exc}")})
                return
            await self._forward_audio(upstream, raw)
            return
        if event_type == EV_COMMIT:
            # Configuration must land before the upstream processes a commit.
            await self._ensure_upstream_configured(upstream)
            await self._enqueue(upstream, {"event_id": _event_id(), "type": EV_COMMIT})
            return
        if event_type in (EV_SESSION_FINISH, EV_SESSION_CLOSE):
            self._finishing = True
            await self._enqueue(upstream, {"event_id": _event_id(), "type": EV_SESSION_FINISH})
            return
        # Forward anything else unchanged so newer DashScope events keep working.
        await self._enqueue(upstream, {"event_id": event.get("event_id") or _event_id(), **event})

    async def _forward_audio(self, upstream: Upstream, raw: bytes) -> None:
        # First audio frame fixes the session configuration.
        await self._ensure_upstream_configured(upstream)
        limit = self._settings.realtime_max_audio_bytes
        if limit > 0 and len(raw) > limit:
            await self._send(
                {"type": "error", "event_id": _event_id(), "error": _error_body("audio_too_large", f"audio chunk exceeds {limit} bytes")}
            )
            return
        pcm = decode_g711(raw, self._input_format)
        await self._enqueue(
            upstream,
            {"event_id": _event_id(), "type": EV_APPEND, "audio": base64.b64encode(pcm).decode()},
        )

    async def _enqueue(self, upstream: Upstream, payload: dict[str, Any]) -> None:
        """Send upstream, or queue while the DashScope session is still starting."""
        if self._upstream_ready.is_set():
            await upstream.send(payload)
            return
        self._pending.append(payload)

    async def _flush_pending(self, upstream: Upstream) -> None:
        await self._upstream_ready.wait()
        while self._pending:
            await upstream.send(self._pending.pop(0))

    # -- upstream → client ---------------------------------------------------

    async def _pump_upstream(self, upstream: Upstream) -> None:
        while True:
            try:
                event = await upstream.recv()
            except Exception as exc:  # upstream closed, timed out, or sent a bad frame
                logger.warning("Realtime upstream ended: %s: %s", type(exc).__name__, exc)
                await self._send(
                    {
                        "type": "error",
                        "event_id": _event_id(),
                        "error": _error_body("upstream_closed", f"{type(exc).__name__}: {exc}"),
                    }
                )
                return
            for outgoing in self._translate(event):
                await self._send(outgoing)
            if event.get("type") == "session.finished":
                self._upstream_finished = True
                await self._close_client(code=1000)
                return

    def _translate(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        event_type = str(event.get("type") or "")
        if event_type == _UPSTREAM_TEXT:
            delta_event = self._to_delta(event)
            return [delta_event] if delta_event else []
        if event_type == _COMPLETED:
            item_id = str(event.get("item_id") or "")
            self._prefixes.pop(item_id, None)
            return [event]
        if event_type in ("session.created", "session.updated"):
            # Keep the client-facing session shape stable regardless of upstream defaults.
            return [{**event, "session": {**(event.get("session") or {}), **self._public_session_fields()}}]
        return [event]

    def _to_delta(self, event: dict[str, Any]) -> dict[str, Any] | None:
        """Turn DashScope's ``text``+``stash`` preview into an incremental ``delta``."""
        item_id = str(event.get("item_id") or "")
        cumulative = f"{event.get('text') or ''}{event.get('stash') or ''}"
        previous = self._prefixes.get(item_id, "")
        if cumulative == previous:
            return None
        if cumulative.startswith(previous):
            delta = cumulative[len(previous) :]
            revised = False
        else:
            # The model revised its draft; OpenAI has no replace event, so send the
            # full preview and flag it. Clients that only append will resync on
            # `completed`; clients that read `text` stay exact.
            delta = cumulative
            revised = True
        self._prefixes[item_id] = cumulative
        return {
            "type": _DOWNSTREAM_DELTA,
            "event_id": event.get("event_id") or _event_id(),
            "item_id": event.get("item_id"),
            "content_index": event.get("content_index", 0),
            "delta": delta,
            # Non-standard extras: cumulative preview and revision flag.
            "text": cumulative,
            "revised": revised,
            **{k: event[k] for k in ("language", "emotion") if k in event},
        }

    async def _emit_session_created(self, upstream_event: dict[str, Any]) -> None:
        session = dict(upstream_event.get("session") or {})
        await self._send(
            {
                "type": "session.created",
                "event_id": _event_id(),
                "session": {**session, **self._public_session_fields()},
            }
        )

    # -- session configuration ----------------------------------------------

    def _apply_client_session(self, session: dict[str, Any]) -> None:
        model = session.get("model")
        if isinstance(model, str) and model.strip():
            self._model = model.strip()
        if session.get("input_audio_format") is not None:
            self._input_format = normalize_input_format(str(session.get("input_audio_format")), self._input_format)
        sample_rate = session.get("sample_rate")
        if isinstance(sample_rate, int) and sample_rate in (8000, 16000):
            self._sample_rate = sample_rate
        transcription = session.get("input_audio_transcription")
        if isinstance(transcription, dict):
            language = transcription.get("language")
            if isinstance(language, str) and language.strip():
                self._language = language.strip()
        # Tolerate a top-level ``language`` too: several OpenAI-compatible clients
        # send the hint flat instead of nested under input_audio_transcription.
        top_level_language = session.get("language")
        if isinstance(top_level_language, str) and top_level_language.strip():
            self._language = top_level_language.strip()
        if "turn_detection" in session:
            # ``null`` selects Manual mode; an object overrides the VAD knobs.
            self._vad_override = session.get("turn_detection")

    def _effective_turn_detection(self) -> dict[str, Any] | None:
        """Resolved VAD config: client override wins, else settings; ``None`` = Manual mode.

        One source of truth for both the upstream ``session.update`` and the
        ``session.updated`` echo sent back to the client.
        """
        override = self._vad_override
        if override is _UNSET:
            if not self._settings.realtime_vad:
                return None
            return {
                "type": "server_vad",
                "threshold": self._settings.realtime_vad_threshold,
                "silence_duration_ms": self._settings.realtime_silence_duration_ms,
            }
        if not isinstance(override, dict):
            return None  # explicit null → the caller commits utterances itself
        return {
            "type": "server_vad",
            "threshold": float(override.get("threshold", self._settings.realtime_vad_threshold)),
            "silence_duration_ms": int(override.get("silence_duration_ms", self._settings.realtime_silence_duration_ms)),
        }

    def _session_config(self) -> dict[str, Any]:
        """Upstream ``session.update`` body (DashScope dialect)."""
        config: dict[str, Any] = {
            "input_audio_format": "pcm",
            "sample_rate": self._sample_rate,
            "turn_detection": self._effective_turn_detection(),
        }
        if self._language:
            config["input_audio_transcription"] = {"language": self._language}
        return config

    def _public_session_fields(self) -> dict[str, Any]:
        """What the client sees: the model and the audio contract it must send."""
        return {
            "model": self._model,
            "modalities": ["text"],
            "input_audio_format": self._input_format,
            "sample_rate": self._sample_rate,
            **({"input_audio_transcription": {"language": self._language}} if self._language else {}),
        }

    # -- transport helpers ---------------------------------------------------

    async def _send(self, payload: dict[str, Any]) -> bool:
        """Deliver one event downstream; ``False`` once the client is gone.

        Sends never raise: a vanished client is a normal end-of-session state, and
        raising here would abort the upstream drain that flushes the final transcript.
        """
        if self._client_gone:
            return False
        try:
            await self._client.send_json(payload)
            return True
        except Exception as exc:
            logger.debug("Realtime client send failed: %r", exc)
            self._client_gone = True
            return False

    async def _close_client(self, *, code: int) -> None:
        try:
            await self._client.close(code=code)
        except Exception:  # already closed or transport gone
            pass
        self._client_gone = True


def _error_body(code: str, message: str) -> dict[str, str]:
    return {"type": "proxy_error", "code": code, "message": message}


def _resolve_proxy(settings: Settings) -> str | bool | None:
    """Upstream proxy choice: explicit URL > env detection > direct."""
    if settings.realtime_proxy_url.strip():
        return settings.realtime_proxy_url.strip()
    return True if settings.realtime_use_env_proxy else None


def _extract_api_key(websocket: WebSocket) -> str:
    """Bearer token from the handshake headers, falling back to a query param.

    Browsers cannot set WebSocket headers, so ``?api_key=`` is accepted too.
    The value is never logged.
    """
    headers = websocket.headers
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    key = headers.get("x-api-key", "")
    if key:
        return key.strip()
    return (websocket.query_params.get("api_key") or "").strip()


@router.websocket("/v1/realtime")
@router.websocket("/realtime")
async def realtime_transcription(websocket: WebSocket) -> None:
    """OpenAI Realtime-compatible streaming transcription endpoint."""
    settings = get_settings()
    if not settings.realtime_enabled:
        await websocket.close(code=1008, reason="Realtime transcription is disabled")
        return

    api_key = _extract_api_key(websocket)
    if not api_key and settings.require_downstream_key and not settings.local_mode:
        await websocket.close(code=1008, reason="Missing Authorization: Bearer <key>")
        return

    params = websocket.query_params
    model = (params.get("model") or settings.realtime_model).strip()
    input_format = normalize_input_format(params.get("input_audio_format"), settings.realtime_input_format)
    try:
        sample_rate = int(params.get("sample_rate") or settings.realtime_sample_rate)
    except ValueError:
        sample_rate = settings.realtime_sample_rate
    if sample_rate not in (8000, 16000):
        sample_rate = 16000
    language = (params.get("language") or settings.realtime_language or "").strip()

    bridge = RealtimeBridge(
        websocket,
        settings=settings,
        api_key=api_key,
        model=model,
        input_format=input_format,
        sample_rate=sample_rate,
        language=language,
    )
    try:
        await bridge.run()
    except WebSocketDisconnect:
        logger.debug("Realtime client disconnected")
    except Exception:  # pragma: no cover - defensive: never leak a stack to the client
        logger.exception("Realtime bridge crashed")
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            pass
