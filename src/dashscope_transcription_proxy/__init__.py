"""DashScope Transcription Proxy — OpenAI-compatible STT backed by 阿里云百炼."""

from .app import app
from .client import TranscriptionClient
from .settings import Settings, get_settings

__all__ = ["app", "Settings", "get_settings", "TranscriptionClient"]
