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


@lru_cache
def get_settings() -> Settings:
    """Lazily instantiate settings so tests can set env vars before import."""
    return Settings()
