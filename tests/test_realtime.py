"""Streaming bridge tests.

Two layers are covered:

* ``local_mode`` — the in-process mock upstream, which pins the client-facing
  event dialect (delta shaping, commit/finish lifecycle) without any network.
* a fake DashScope WebSocket server — which pins the upstream side: URL and
  model query param, ``Authorization`` passthrough, G.711 → PCM16 conversion,
  and the ``session.finish`` flush.
"""

from __future__ import annotations

import base64
import contextlib
import json
import threading
import time
from concurrent.futures import CancelledError as FuturesCancelledError
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from dashscope_transcription_proxy.app import app
from dashscope_transcription_proxy.realtime import (
    EV_APPEND,
    EV_SESSION_UPDATE,
    derive_realtime_ws_url,
)
from dashscope_transcription_proxy.settings import get_settings

COMPLETED = "conversation.item.input_audio_transcription.completed"
DELTA = "conversation.item.input_audio_transcription.delta"


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def set_env(monkeypatch):
    """Set ``DASHSCOPE_PROXY_*`` vars and drop the cached Settings.

    ``get_settings`` is ``lru_cache``d and the TestClient lifespan reads it on
    startup, so env changes only land after an explicit cache clear.
    """

    def _apply(**values: object) -> None:
        for key, value in values.items():
            monkeypatch.setenv(f"DASHSCOPE_PROXY_{key.upper()}", str(value))
        get_settings.cache_clear()

    yield _apply
    get_settings.cache_clear()


def _recv_until(ws, wanted: str, limit: int = 20, predicate=None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Drain events until ``wanted`` arrives; returns it plus everything seen.

    ``predicate`` narrows the match: the bridge pushes its own default
    ``session.update`` upstream on connect, so a client-initiated update is the
    *second* ``session.updated`` a caller observes.
    """
    seen: list[dict[str, Any]] = []
    for _ in range(limit):
        event = ws.receive_json()
        seen.append(event)
        if event.get("type") == wanted and (predicate is None or predicate(event)):
            return event, seen
    raise AssertionError(f"{wanted} never arrived; got {[e.get('type') for e in seen]}")


def _finish_session(ws) -> None:
    """Close the way a real client does: send ``session.finish``, drain the reply.

    Leaving the ``with`` block without a finish makes starlette's TestClient cancel
    the app task while the bridge is still flushing upstream, which surfaces here as
    a ``CancelledError`` instead of a clean disconnect. Production servers (uvicorn)
    do not cancel the handler, so this is purely a test-harness concern.
    """
    ws.send_json({"type": "session.finish"})
    _recv_until(ws, "session.finished")


# ---------------------------------------------------------------------------
# URL derivation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("https://dashscope.aliyuncs.com/api/v1", "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"),
        (
            "https://ws-abc.cn-beijing.maas.aliyuncs.com/api/v1",
            "wss://ws-abc.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime",
        ),
        ("https://dashscope-intl.aliyuncs.com/api/v1/", "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime"),
        ("http://127.0.0.1:9000/api/v1", "ws://127.0.0.1:9000/api-ws/v1/realtime"),
        ("wss://host/api-ws/v1", "wss://host/api-ws/v1/realtime"),
    ],
)
def test_derive_realtime_ws_url(base, expected):
    assert derive_realtime_ws_url(base) == expected


# ---------------------------------------------------------------------------
# local_mode: client-facing dialect
# ---------------------------------------------------------------------------

def test_streaming_session_lifecycle_local_mode(client, set_env):
    set_env(local_mode="true", realtime_sample_rate="8000", realtime_input_format="g711_ulaw")

    with client.websocket_connect("/v1/realtime") as ws:
        created, _ = _recv_until(ws, "session.created")
        assert created["session"]["model"] == "qwen3-asr-flash-realtime"
        assert created["session"]["input_audio_format"] == "g711_ulaw"
        assert created["session"]["sample_rate"] == 8000

        # 100 bytes of µ-law = 200 bytes of PCM16 after decoding.
        ws.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(bytes(range(100))).decode(),
            }
        )
        ws.send_json({"type": "input_audio_buffer.commit"})

        deltas: list[dict[str, Any]] = []
        completed, seen = _recv_until(ws, COMPLETED)
        deltas = [event for event in seen if event.get("type") == DELTA]

        # DashScope's text+stash preview must surface as incremental OpenAI deltas.
        assert [d["delta"] for d in deltas] == ["[mock] ", "流式识别结果"]
        assert deltas[-1]["text"] == "[mock] 流式识别结果"
        assert all(d["revised"] is False for d in deltas)
        assert completed["transcript"].startswith("[mock]")
        assert completed["language"] == "zh"

        ws.send_json({"type": "session.finish"})
        finished, _ = _recv_until(ws, "session.finished")
        assert finished["type"] == "session.finished"


def test_client_session_update_overrides_defaults(client, set_env):
    set_env(local_mode="true")

    with client.websocket_connect("/v1/realtime") as ws:
        _recv_until(ws, "session.created")
        ws.send_json(
            {
                "type": "session.update",
                "session": {
                    "model": "fun-asr-realtime",
                    "input_audio_format": "g711_alaw",
                    "sample_rate": 16000,
                    "input_audio_transcription": {"language": "en"},
                    "turn_detection": None,  # Manual mode
                },
            }
        )
        updated, _ = _recv_until(
            ws, "session.updated", predicate=lambda e: (e.get("session") or {}).get("model") == "fun-asr-realtime"
        )
        assert updated["session"]["model"] == "fun-asr-realtime"
        assert updated["session"]["input_audio_format"] == "g711_alaw"
        assert updated["session"]["sample_rate"] == 16000
        assert updated["session"]["input_audio_transcription"] == {"language": "en"}
        _finish_session(ws)


def test_query_params_select_model_and_format(client, set_env):
    set_env(local_mode="true")

    with client.websocket_connect("/v1/realtime?model=qwen-audio-3.0-asr-flash-streaming&sample_rate=8000") as ws:
        created, _ = _recv_until(ws, "session.created")
        assert created["session"]["model"] == "qwen-audio-3.0-asr-flash-streaming"
        assert created["session"]["sample_rate"] == 8000
        _finish_session(ws)


def test_invalid_base64_reports_error_event(client, set_env):
    set_env(local_mode="true")

    with client.websocket_connect("/v1/realtime") as ws:
        _recv_until(ws, "session.created")
        ws.send_json({"type": "input_audio_buffer.append", "audio": "!!!not-base64!!!"})
        error, _ = _recv_until(ws, "error")
        assert error["error"]["code"] == "invalid_audio"
        _finish_session(ws)


def test_missing_key_is_rejected(client, set_env):
    set_env(local_mode="false", require_downstream_key="true")

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
    assert excinfo.value.code == 1008


def test_realtime_disabled_closes_connection(client, set_env):
    set_env(realtime_enabled="false")

    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()
    assert excinfo.value.code == 1008


def test_health_reports_realtime(client, set_env):
    set_env(local_mode="true")
    body = client.get("/health").json()
    assert body["realtime"]["enabled"] is True
    assert body["realtime"]["model"] == "qwen3-asr-flash-realtime"
    assert body["realtime"]["upstream"] == "mock"


# ---------------------------------------------------------------------------
# fake DashScope upstream: wire-level behaviour
# ---------------------------------------------------------------------------

class _FakeDashScope:
    """Minimal DashScope realtime server recording what the proxy sends."""

    def __init__(self) -> None:
        self.received: list[dict[str, Any]] = []
        self.path = ""
        self.authorization = ""
        self.pcm_bytes = 0
        self._server = None
        self._thread: threading.Thread | None = None
        self.port = 0

    def start(self) -> None:
        from websockets.sync.server import serve

        self._server = serve(self._handle, "127.0.0.1", 0)
        self.port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _handle(self, connection) -> None:
        self.path = connection.request.path
        self.authorization = connection.request.headers.get("Authorization", "")
        connection.send(json.dumps({"type": "session.created", "event_id": "ev_0", "session": {"id": "sess_x", "model": "upstream"}}))
        for raw in connection:
            event = json.loads(raw)
            self.received.append(event)
            kind = event.get("type")
            if kind == "session.update":
                connection.send(json.dumps({"type": "session.updated", "event_id": "ev_1", "session": event.get("session")}))
            elif kind == "input_audio_buffer.append":
                self.pcm_bytes += len(base64.b64decode(event["audio"]))
            elif kind == "input_audio_buffer.commit":
                connection.send(
                    json.dumps(
                        {
                            "type": "conversation.item.input_audio_transcription.text",
                            "event_id": "ev_2",
                            "item_id": "item_1",
                            "content_index": 0,
                            "text": "今天",
                            "stash": "天气",
                        }
                    )
                )
                connection.send(
                    json.dumps(
                        {
                            "type": COMPLETED,
                            "event_id": "ev_3",
                            "item_id": "item_1",
                            "content_index": 0,
                            "transcript": "今天天气",
                        }
                    )
                )
            elif kind == "session.finish":
                connection.send(json.dumps({"type": "session.finished", "event_id": "ev_4"}))
                return


@pytest.fixture()
def fake_upstream():
    server = _FakeDashScope()
    server.start()
    yield server
    server.stop()


def test_upstream_wiring_and_g711_conversion(client, set_env, fake_upstream):
    set_env(
        local_mode="false",
        realtime_ws_url=f"ws://127.0.0.1:{fake_upstream.port}/api-ws/v1/realtime",
        realtime_sample_rate="8000",
        realtime_input_format="g711_ulaw",
        # The ambient HTTPS_PROXY cannot reach a loopback fake upstream.
        realtime_use_env_proxy="false",
    )

    headers = {"Authorization": "Bearer ***"}
    with client.websocket_connect("/v1/realtime?model=qwen3-asr-flash-realtime", headers=headers) as ws:
        _recv_until(ws, "session.created")
        ws.send_json(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(bytes(160)).decode(),  # 160 µ-law bytes
            }
        )
        ws.send_json({"type": "input_audio_buffer.commit"})
        completed, seen = _recv_until(ws, COMPLETED)
        assert completed["transcript"] == "今天天气"

        deltas = [event for event in seen if event.get("type") == DELTA]
        assert deltas and deltas[0]["delta"] == "今天天气"
        assert deltas[0]["text"] == "今天天气"

        ws.send_json({"type": "session.finish"})
        _recv_until(ws, "session.finished")
    # The model travels as a query param; the downstream key is passed through verbatim.
    assert "model=qwen3-asr-flash-realtime" in fake_upstream.path
    assert fake_upstream.authorization == "Bearer ***"
    # 160 µ-law bytes must arrive upstream as 320 PCM16 bytes at the declared rate.
    assert fake_upstream.pcm_bytes == 320
    session_updates = [event for event in fake_upstream.received if event.get("type") == "session.update"]
    assert session_updates, "proxy must configure the upstream session"
    assert session_updates[0]["session"]["sample_rate"] == 8000
    assert session_updates[0]["session"]["input_audio_format"] == "pcm"
    # Every upstream event carries the mandatory event_id.
    assert all(event.get("event_id") for event in fake_upstream.received)


def test_upstream_gets_exactly_one_session_update(client, set_env, fake_upstream):
    """Regression: DashScope rejects a second ``session.update`` with
    ``1007 session already started or finished or failed``.

    The bridge must coalesce every client update into one upstream push, sent
    lazily right before the first audio frame.
    """
    set_env(
        local_mode="false",
        realtime_ws_url=f"ws://127.0.0.1:{fake_upstream.port}/api-ws/v1/realtime",
        realtime_use_env_proxy="false",
        realtime_input_format="g711_ulaw",
    )

    with client.websocket_connect("/v1/realtime", headers={"Authorization": "Bearer ***"}) as ws:
        _recv_until(ws, "session.created")
        # All three updates arrive before audio: they merge into one upstream push.
        ws.send_json({"type": "session.update", "session": {"sample_rate": 8000}})
        acked, _ = _recv_until(ws, "session.updated", predicate=lambda e: (e.get("session") or {}).get("sample_rate") == 8000)
        assert acked["session"]["turn_detection"]["type"] == "server_vad"
        ws.send_json({"type": "session.update", "session": {"model": "fun-asr-realtime", "turn_detection": None}})
        ws.send_json({"type": "session.update", "session": {"language": "en"}})
        ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x01" * 320).decode()})
        ws.send_json({"type": "input_audio_buffer.commit"})
        _recv_until(ws, COMPLETED)
        # A post-audio update is acknowledged to the client but must never be
        # forwarded: the DashScope session is already started.
        ws.send_json({"type": "session.update", "session": {"language": "ja"}})
        late, _ = _recv_until(
            ws, "session.updated", predicate=lambda e: (e.get("session") or {}).get("input_audio_transcription") == {"language": "ja"}
        )
        assert late["session"]["input_audio_transcription"] == {"language": "ja"}
        ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x01" * 320).decode()})
        ws.send_json({"type": "input_audio_buffer.commit"})
        _recv_until(ws, COMPLETED)
        ws.send_json({"type": "session.finish"})
        _recv_until(ws, "session.finished")

    types = [event.get("type") for event in fake_upstream.received]
    updates = [event for event in fake_upstream.received if event.get("type") == EV_SESSION_UPDATE]
    assert len(updates) == 1, f"expected one upstream session.update, got {types}"
    # The single update must land before any audio, carrying the merged config.
    assert types.index(EV_SESSION_UPDATE) < types.index(EV_APPEND)
    session = updates[0]["session"]
    assert session["sample_rate"] == 8000
    assert session["turn_detection"] is None  # last explicit client choice wins
    assert session["input_audio_transcription"] == {"language": "en"}


def test_client_disconnect_still_flushes_upstream(client, set_env, fake_upstream):
    set_env(
        local_mode="false",
        realtime_ws_url=f"ws://127.0.0.1:{fake_upstream.port}/api-ws/v1/realtime",
        realtime_use_env_proxy="false",
    )

    # Abrupt exit on purpose: no session.finish from the caller. The bridge must
    # still flush upstream. TestClient may surface the app-side teardown as a
    # cancellation, which is a harness artifact and not what this test asserts.
    with contextlib.suppress(WebSocketDisconnect, FuturesCancelledError):
        with client.websocket_connect("/v1/realtime", headers={"Authorization": "Bearer ***"}) as ws:
            _recv_until(ws, "session.created")
            ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x01" * 40).decode()})

    # DashScope drops the in-progress item without session.finish; the bridge must send it.
    saw_finish = False
    for _ in range(100):
        saw_finish = any(event.get("type") == "session.finish" for event in fake_upstream.received)
        if saw_finish:
            break
        time.sleep(0.05)
    assert saw_finish, f"no session.finish upstream; got {[e.get('type') for e in fake_upstream.received]}"
