"""Configuration from environment variables (prefix ``DASHSCOPE_PROXY_``)."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "DASHSCOPE_PROXY_", "case_sensitive": False}

    # --- DashScope ---
    base_http_api_url: str = "https://dashscope.aliyuncs.com/api/v1"
    """DashScope base URL: ``https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api/v1``."""

    # --- Model ---
    model: str = "qwen3-asr-flash"
    """Default DashScope model ID."""

    enable_itn: bool = True
    """Inverse text normalisation (numbers, dates…)."""

    enable_lid: bool = False
    """Auto-detect language (enable_lid in asr_options)."""

    default_language: str = "zh"
    """Fallback when the caller provides no ``language`` hint."""

    # --- Proxy behaviour ---
    results_dir: str = ""
    """Save raw recognition JSON to ``<results_dir>/<id>.json``."""

    local_mode: bool = False
    """Skip DashScope — return mock data."""

    max_upload_mb: float = 10.0
    """Max uploaded audio size in MB (reject larger files). 0 = unlimited."""

    # --- Backend selection ---
    backend: str = "dashscope"
    """``dashscope`` = 阿里云百炼云 API（默认）；``vllm`` = 本地 Qwen3-ASR（qwen-asr-serve）。"""

    require_downstream_key: bool = True
    """Require ``Authorization: Bearer <key>`` from the caller. Set false for a trusted
    LAN deployment of the ``vllm`` backend, where the downstream key is meaningless."""

    # --- Local vLLM backend (backend=vllm) ---
    vllm_base_url: str = "http://127.0.0.1:8001/v1"
    """OpenAI-compatible base URL of the local Qwen3-ASR server."""

    vllm_api_key: str = ""
    """Key for the local server's ``--api-key``. Empty = send no auth header."""

    vllm_model: str = "qwen3-asr"
    """``--served-model-name`` of the local server. The caller's ``model`` is ignored."""

    vllm_timeout: float = 600.0
    """Per-request timeout in seconds. Long audio needs a generous value."""

    vllm_max_tokens: int = 4096
    """``max_tokens`` for the completion. Raise for long audio."""

    vllm_passthrough_context: bool = True
    """Forward the caller's ``prompt`` as a system message (context / hotword hinting)."""

    qwen_output_parser: str = "auto"
    """How to split Qwen3-ASR's raw completion into ``(language, text)``:
    ``auto`` = official ``qwen_asr.parse_asr_output`` if importable, else a conservative
    heuristic, else passthrough; ``official`` = require the official parser;
    ``none`` = passthrough. Pin this once you have observed real output."""

    # --- Realtime (streaming) bridge ---
    realtime_enabled: bool = True
    """Serve the WebSocket streaming endpoint (``/v1/realtime``)."""

    realtime_model: str = "qwen3-asr-flash-realtime"
    """Default realtime model. Callers can override it with ``?model=`` or ``session.update``."""

    realtime_ws_url: str = ""
    """DashScope realtime WebSocket base URL. Empty = derived from ``base_http_api_url``
    (``https://host/api/v1`` → ``wss://host/api-ws/v1/realtime``). Set it explicitly for a
    workspace domain, e.g. ``wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime``."""

    realtime_sample_rate: int = 16000
    """Sample rate (Hz) advertised to DashScope: ``16000`` or ``8000``.
    Use ``8000`` for telephony sources; DashScope upsamples internally."""

    realtime_input_format: str = "pcm16"
    """Audio format the downstream client sends: ``pcm16`` | ``g711_ulaw`` | ``g711_alaw``.
    G.711 is decoded to PCM16 before forwarding. Clients may override per session."""

    realtime_language: str = ""
    """Language hint (``zh``, ``en``, ``yue``, …). Empty = let the model detect."""

    realtime_vad: bool = True
    """``true`` = server VAD segments speech (``turn_detection``); ``false`` = Manual mode,
    where the client commits each utterance with ``input_audio_buffer.commit``."""

    realtime_vad_threshold: float = 0.0
    """VAD trigger threshold, ``[-1, 1]``. DashScope recommends ``0.0``."""

    realtime_silence_duration_ms: int = 400
    """Silence (ms) that ends an utterance in VAD mode, ``[200, 6000]``."""

    realtime_finish_timeout: float = 5.0
    """Seconds to wait for ``session.finished`` after ``session.finish`` before dropping
    the upstream connection. A too-short value loses the final transcript."""

    realtime_max_audio_bytes: int = 15 * 1024 * 1024
    """Reject a single ``input_audio_buffer.append`` payload larger than this (DashScope limit)."""

    realtime_use_env_proxy: bool = True
    """Honor ``HTTPS_PROXY``/``HTTP_PROXY`` for the upstream WebSocket (``websockets`` default).
    Set ``false`` to force a direct connection — needed when the ambient proxy cannot reach
    DashScope, or when the upstream is local/LAN."""

    realtime_proxy_url: str = ""
    """Explicit proxy URL for the upstream WebSocket. Takes precedence over
    ``realtime_use_env_proxy`` when set."""


@lru_cache
def get_settings() -> Settings:
    """Lazily instantiate settings so tests can set env vars before import."""
    return Settings()
