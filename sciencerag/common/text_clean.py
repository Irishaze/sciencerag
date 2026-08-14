"""Unicode cleanup for text that passed through PDF extraction and/or an
LLM before reaching a person — both stages leak characters that render as
tofu/boxes in normal UI fonts: non-breaking/soft hyphens, non-breaking and
narrow-no-break spaces, zero-width joiners, BOMs, and stray C0 control
characters.

Deliberately an explicit character map, NOT unicodedata.normalize("NFKC",
...): NFKC's compatibility decomposition also folds the CJK "Fullwidth
Forms" block (U+FF00-FF5E) to plain ASCII, e.g. "你好，世界（测试）" ->
"你好,世界(测试)" — verified directly against Python's own unicodedata.
Chinese full-width punctuation is not a rendering defect to "fix"; NFKC
would silently corrupt every Chinese-language answer's punctuation style.

Skips $...$ / $$...$$ spans (see AnswerCard.tsx's KaTeX rendering) so a
stray hyphen/space variant inside equation source is never rewritten mid-
formula.
"""

from __future__ import annotations

import re

# Built from explicit codepoints (chr(0x...)), not literal characters typed
# into this file: a prior version embedded the space variants as literal
# glyphs, which silently collapsed to plain ASCII spaces somewhere in the
# edit pipeline and turned those entries into no-ops with no error raised
# anywhere. chr(0x...) can't be mis-typed into the wrong character that way.
_CHAR_MAP = str.maketrans(
    {
        chr(0x2010): "-",  # hyphen
        chr(0x2011): "-",  # non-breaking hyphen
        chr(0x2012): "-",  # figure dash
        chr(0x2013): "-",  # en dash
        chr(0x2014): chr(0x2014),  # em dash — kept as-is, it's a real glyph
        chr(0x00A0): " ",  # non-breaking space
        chr(0x2007): " ",  # figure space
        chr(0x202F): " ",  # narrow no-break space
        chr(0x200B): "",  # zero-width space
        chr(0x200C): "",  # zero-width non-joiner
        chr(0x200D): "",  # zero-width joiner
        chr(0xFEFF): "",  # BOM
        chr(0x00AD): "",  # soft hyphen
        chr(0x2103): "°C",  # CJK compatibility "degree Celsius"
        chr(0x2109): "°F",  # CJK compatibility "degree Fahrenheit"
    }
)

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_MATH_SPAN_RE = re.compile(r"\${1,2}[^$]*\${1,2}")


def _clean_segment(segment: str) -> str:
    segment = segment.translate(_CHAR_MAP)
    return _CONTROL_CHARS_RE.sub("", segment)


def clean_text(text: str) -> str:
    """Apply _clean_segment to everything except $...$/$$...$$ math spans."""
    if not text:
        return text
    parts = []
    last_end = 0
    for match in _MATH_SPAN_RE.finditer(text):
        parts.append(_clean_segment(text[last_end : match.start()]))
        parts.append(match.group(0))
        last_end = match.end()
    parts.append(_clean_segment(text[last_end:]))
    return "".join(parts)
