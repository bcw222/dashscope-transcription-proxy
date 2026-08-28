"""Unit tests for DashScope response normalisation."""

from __future__ import annotations

from dashscope_transcription_proxy.client import _normalize_content
from dashscope_transcription_proxy.app import _extract_text


def test_str():
    assert _normalize_content("你好") == "你好"


def test_none():
    assert _normalize_content(None) == ""


def test_list_of_text_dicts():
    assert _normalize_content([{"text": "Hello"}, {"text": " World"}]) == "Hello World"


def test_list_mixed_str_and_dict():
    assert _normalize_content(["你", {"text": "好"}]) == "你好"


def test_list_with_missing_text():
    assert _normalize_content([{"foo": "bar"}, {"text": "ok"}]) == "ok"


# --- _extract_text across DashScope response shapes --------------------------

def test_extract_text_choices_shape():
    ds = {"output": {"choices": [{"message": {"content": "识别文本"}}]}}
    assert _extract_text(ds) == "识别文本"


def test_extract_text_choices_list_content():
    ds = {"output": {"choices": [{"message": {"content": [{"text": "你"}, {"text": "好"}]}}]}}
    assert _extract_text(ds) == "你好"


def test_extract_text_sentence_shape():
    ds = {"output": {"output": {"sentence": {"text": "识别文本"}}}}
    assert _extract_text(ds) == "识别文本"


def test_extract_text_top_level_text():
    ds = {"output": {"text": "识别文本"}}
    assert _extract_text(ds) == "识别文本"


def test_extract_text_missing():
    ds = {"output": {}}
    assert _extract_text(ds) == ""
