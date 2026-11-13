"""DashScope Transcription Proxy — OpenAI-compatible STT.

Backends: 阿里云百炼 (DashScope, default) or a local Qwen3-ASR server (vLLM).
"""

from .app import app
from .client import TranscriptionClient
from .local_client import VLLMTranscriptionClient, parse_qwen_output
from .settings import Settings, get_settings

__all__ = [
    "app",
    "Settings",
    "get_settings",
    "TranscriptionClient",
    "VLLMTranscriptionClient",
    "parse_qwen_output",
]
