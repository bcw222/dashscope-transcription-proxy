"""Shared test fixtures — force LOCAL_MODE before any settings are read."""

import os

os.environ.setdefault("DASHSCOPE_PROXY_LOCAL_MODE", "true")

import pytest  # noqa: E402

from dashscope_transcription_proxy.settings import get_settings  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure each test reads a fresh Settings instance."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
