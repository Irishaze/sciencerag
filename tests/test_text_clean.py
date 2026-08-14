"""Unit tests for sciencerag.common.text_clean (Unicode cleanup for text
that passed through PDF extraction and/or an LLM before reaching a person)."""

from sciencerag.common.text_clean import clean_text


def test_strips_non_breaking_hyphen():
    # U+2011 NON-BREAKING HYPHEN, the character reported missing a glyph in
    # "发电‑制冷" — many UI fonts render it as a tofu box.
    assert clean_text("发电‑制冷") == "发电-制冷"


def test_strips_zero_width_and_bom_characters():
    assert clean_text("A​B﻿C‌D") == "ABCD"


def test_normalizes_non_breaking_space():
    assert clean_text("10 W") == "10 W"


def test_maps_degree_celsius_compatibility_character():
    assert clean_text("25℃") == "25°C"  # ℃ -> °C, via the explicit map


def test_preserves_fullwidth_chinese_punctuation():
    # Regression: unicodedata.normalize("NFKC", ...) folds the whole CJK
    # "Fullwidth Forms" block to plain ASCII (，（）：！？ -> ,()：!?),
    # silently corrupting Chinese punctuation style. clean_text must NOT
    # do that — verified directly that NFKC does before removing it.
    text = "你好，世界（测试）：一二三！四？"
    assert clean_text(text) == text


def test_strips_control_characters():
    assert clean_text("a\x0bb\x0cc") == "abc"


def test_leaves_ordinary_text_untouched():
    text = "I_opt = alpha * Tc / R, about 200-300 words."
    assert clean_text(text) == text


def test_skips_math_spans():
    # A stray U+2011 inside a $...$ span must survive untouched — cleanup
    # must not risk altering equation content NFKC could otherwise fold.
    text = "见下式 $Q_c ‑ K$ 以及正文中的连字符‑符号。"
    cleaned = clean_text(text)
    assert "$Q_c ‑ K$" in cleaned
    assert "正文中的连字符-符号" in cleaned


def test_empty_string():
    assert clean_text("") == ""
