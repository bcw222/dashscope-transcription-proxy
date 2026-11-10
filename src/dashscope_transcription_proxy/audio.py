"""G.711 helpers for the realtime bridge.

Telephony-style sources (OpenClaw's Talk/dictation relay, Twilio media
streams) deliver 8 kHz G.711 µ-law/A-law. DashScope's realtime ASR accepts
PCM, so the bridge decodes G.711 to little-endian PCM16 before forwarding.

The decode tables are the classic ITU-T G.711 reference implementations
(same constants as ``g711.c``), expressed as lookup tables so a stream of
audio stays cheap to convert.
"""

from __future__ import annotations

import array

__all__ = [
    "SUPPORTED_INPUT_FORMATS",
    "decode_g711",
    "g711_alaw_to_pcm16",
    "g711_ulaw_to_pcm16",
    "normalize_input_format",
]

_ULAW_BIAS = 0x84
_ALAW_XOR = 0x55


def _build_ulaw_table() -> tuple[int, ...]:
    """``ulaw2linear`` for all 256 byte values."""
    table: list[int] = []
    for byte in range(256):
        inverted = ~byte & 0xFF
        magnitude = ((inverted & 0x0F) << 3) + _ULAW_BIAS
        magnitude <<= (inverted & 0x70) >> 4
        value = (_ULAW_BIAS - magnitude) if inverted & 0x80 else (magnitude - _ULAW_BIAS)
        table.append(value)
    return tuple(table)


def _build_alaw_table() -> tuple[int, ...]:
    """``alaw2linear`` for all 256 byte values."""
    table: list[int] = []
    for byte in range(256):
        inverted = byte ^ _ALAW_XOR
        magnitude = (inverted & 0x0F) << 4
        segment = (inverted & 0x70) >> 4
        if segment == 0:
            magnitude += 8
        elif segment == 1:
            magnitude += 0x108
        else:
            magnitude += 0x108
            magnitude <<= segment - 1
        table.append(magnitude if inverted & 0x80 else -magnitude)
    return tuple(table)


_ULAW_TABLE = _build_ulaw_table()
_ALAW_TABLE = _build_alaw_table()


def _decode_with_table(payload: bytes, table: tuple[int, ...]) -> bytes:
    samples = array.array("h", bytes(2 * len(payload)))
    for index, byte in enumerate(payload):
        samples[index] = table[byte]
    if array.array("h", b"\x01\x00")[0] != 1:  # pragma: no cover - big-endian host
        samples.byteswap()
    return samples.tobytes()


def g711_ulaw_to_pcm16(payload: bytes) -> bytes:
    """Decode G.711 µ-law bytes into little-endian PCM16 mono samples."""
    return _decode_with_table(payload, _ULAW_TABLE)


def g711_alaw_to_pcm16(payload: bytes) -> bytes:
    """Decode G.711 A-law bytes into little-endian PCM16 mono samples."""
    return _decode_with_table(payload, _ALAW_TABLE)


# Accepted spellings from downstream clients. OpenClaw advertises ``g711_ulaw``;
# OpenAI-compatible clients commonly say ``mulaw``; Twilio docs say ``ulaw``.
_FORMAT_ALIASES = {
    "pcm": "pcm16",
    "pcm16": "pcm16",
    "pcm_s16le": "pcm16",
    "s16le": "pcm16",
    "g711_ulaw": "g711_ulaw",
    "ulaw": "g711_ulaw",
    "mulaw": "g711_ulaw",
    "g711_alaw": "g711_alaw",
    "alaw": "g711_alaw",
}

SUPPORTED_INPUT_FORMATS = ("pcm16", "g711_ulaw", "g711_alaw")


def normalize_input_format(value: str | None, default: str = "pcm16") -> str:
    """Map a client-declared audio format onto a supported canonical name.

    Unknown values fall back to ``default`` rather than raising: a client that
    labels plain PCM as ``audio/pcm`` still transcribes correctly.
    """
    if not value:
        return default
    return _FORMAT_ALIASES.get(value.strip().lower(), default)


def decode_g711(payload: bytes, input_format: str) -> bytes:
    """Return PCM16 bytes for ``payload`` according to ``input_format``."""
    if input_format == "g711_ulaw":
        return g711_ulaw_to_pcm16(payload)
    if input_format == "g711_alaw":
        return g711_alaw_to_pcm16(payload)
    return payload
