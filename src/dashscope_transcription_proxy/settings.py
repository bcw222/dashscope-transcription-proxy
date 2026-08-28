"""Configuration from environment variables (prefix ``DASHSCOPE_PROXY_``)."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "DASHSCOPE_PROXY_", "case_sensitive": False}

    # --- DashScope ---
    base_http_api_url: str = ""
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


@lru_cache
def get_settings() -> Settings:
    """Lazily instantiate settings so tests can set env vars before import."""
    return Settings()
