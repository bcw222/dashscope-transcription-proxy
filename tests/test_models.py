"""Unit tests for pure mapping / helper functions (no network)."""

from __future__ import annotations

from dashscope_transcription_proxy.models import build_response


def test_build_response_json():
    out = build_response(text="你好", response_format="json")
    assert out == {"text": "你好"}


def test_build_response_text():
    out = build_response(text="你好", response_format="text")
    assert out == "你好"


def test_build_response_verbose_json_empty():
    out = build_response(text="你好", response_format="verbose_json")
    assert out["text"] == "你好"
    assert out["language"] == ""
    assert out["duration"] == 0.0
    assert out["segments"] == []
    assert out["words"] == []


def test_build_response_verbose_json_with_annotations():
    ds = {
        "output": {
            "choices": [
                {
                    "message": {
                        "content": "Hello World",
                        "annotations": [
                            {
                                "type": "audio_info",
                                "language": "en",
                                "words": [
                                    {"text": "Hello", "begin_time": 100, "end_time": 500},
                                    {"text": " World", "begin_time": 500, "end_time": 1000},
                                ],
                            }
                        ],
                    }
                }
            ]
        },
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    out = build_response(text="Hello World", response_format="verbose_json", dashscope_output=ds)
    assert out["language"] == "en"
    assert out["words"] == [
        {"word": "Hello", "start": 0.1, "end": 0.5},
        {"word": " World", "start": 0.5, "end": 1.0},
    ]
    assert out["duration"] == 1.0
    assert len(out["segments"]) == 1
    assert out["segments"][0]["text"] == "Hello World"
    assert out["usage"]["type"] == "tokens"


def test_build_response_verbose_json_duration_usage():
    ds = {"usage": {"audio_seconds": 9}}
    out = build_response(text="x", response_format="verbose_json", dashscope_output=ds)
    assert out["usage"] == {"type": "duration", "seconds": 9}


def test_build_response_verbose_json_zero_duration_falls_back_to_tokens():
    # audio_seconds == 0 should NOT be treated as duration usage
    ds = {
        "usage": {"audio_seconds": 0, "input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }
    out = build_response(text="x", response_format="verbose_json", dashscope_output=ds)
    assert out["usage"]["type"] == "tokens"
    assert out["usage"]["total_tokens"] == 5


def _ds_with_words():
    return {
        "output": {
            "choices": [
                {
                    "message": {
                        "content": "Hello World",
                        "annotations": [
                            {
                                "type": "audio_info",
                                "words": [
                                    {"text": "Hello", "begin_time": 0, "end_time": 500},
                                    {"text": " World", "begin_time": 500, "end_time": 1000},
                                ],
                            }
                        ],
                    }
                }
            ]
        }
    }


def test_build_response_srt():
    out = build_response(text="Hello World", response_format="srt", dashscope_output=_ds_with_words())
    assert isinstance(out, str)
    assert "00:00:00,000 --> 00:00:00,500" in out
    assert "Hello" in out
    assert "1\n" in out


def test_build_response_vtt():
    out = build_response(text="Hello World", response_format="vtt", dashscope_output=_ds_with_words())
    assert isinstance(out, str)
    assert out.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:00.500" in out
