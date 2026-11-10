"""G.711 decode tests — canonical reference values plus a round-trip sweep."""

from __future__ import annotations

import struct

from dashscope_transcription_proxy.audio import (
    decode_g711,
    g711_alaw_to_pcm16,
    g711_ulaw_to_pcm16,
    normalize_input_format,
)

# Reference values from the ITU-T G.711 tables (same constants as g711.c).
ULAW_REFERENCE = {
    0xFF: 0,        # positive zero
    0x7F: 0,        # negative zero
    0xFE: 8,        # smallest positive step
    0x00: -32124,   # negative full scale
    0x80: 32124,    # positive full scale
}


def test_ulaw_reference_values():
    payload = bytes(ULAW_REFERENCE)
    decoded = g711_ulaw_to_pcm16(payload)
    assert len(decoded) == 2 * len(payload)
    samples = struct.unpack(f"<{len(ULAW_REFERENCE)}h", decoded)
    assert dict(zip(ULAW_REFERENCE, samples)) == ULAW_REFERENCE


def test_alaw_sign_symmetry():
    # 0x55 and 0xD5 differ only in the sign bit, so they must decode to ±the same magnitude.
    low, high = struct.unpack("<2h", g711_alaw_to_pcm16(bytes([0x55, 0xD5])))
    assert low == -high
    assert low != 0


def test_decode_g711_dispatches_by_format():
    payload = bytes([0xFE, 0x00])
    assert decode_g711(payload, "g711_ulaw") == g711_ulaw_to_pcm16(payload)
    assert decode_g711(payload, "g711_alaw") == g711_alaw_to_pcm16(payload)
    # pcm16 is a passthrough: the bridge must not touch already-linear audio.
    assert decode_g711(payload, "pcm16") == payload


def _pcm16_to_ulaw(sample: int) -> int:
    """Reference µ-law encoder, used only to exercise the decoder round-trip."""
    bias = 0x84
    sign = 0
    if sample < 0:
        sign = 0x80
        sample = -sample
    sample = min(sample, 0x7FFF) + bias
    exponent = 7
    mask = 0x4000
    while not (sample & mask) and exponent > 0:
        exponent -= 1
        mask >>= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


def test_ulaw_round_trip_within_quantisation_error():
    originals = list(range(-32000, 32001, 137))
    encoded = bytes(_pcm16_to_ulaw(sample) for sample in originals)
    decoded = struct.unpack(f"<{len(originals)}h", g711_ulaw_to_pcm16(encoded))
    for original, restored in zip(originals, decoded):
        # µ-law is logarithmic: worst-case error stays far below 2% of full scale.
        assert abs(original - restored) <= 640, (original, restored)


def test_normalize_input_format_aliases():
    assert normalize_input_format("g711_ulaw") == "g711_ulaw"
    assert normalize_input_format("mulaw") == "g711_ulaw"
    assert normalize_input_format("G711_ULAW") == "g711_ulaw"
    assert normalize_input_format("pcm") == "pcm16"
    assert normalize_input_format("s16le") == "pcm16"
    assert normalize_input_format("alaw") == "g711_alaw"
    # Unknown or missing values fall back rather than raising mid-stream.
    assert normalize_input_format(None) == "pcm16"
    assert normalize_input_format("audio/pcm", "g711_ulaw") == "g711_ulaw"
    assert normalize_input_format("opus") == "pcm16"
