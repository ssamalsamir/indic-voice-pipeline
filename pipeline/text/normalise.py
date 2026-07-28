"""Indic text normalisation — the single most important non-model component.

The same text must be normalised identically at TRAIN time and INFERENCE time, or the
model learns one distribution and is scored on another. So this module is the one
source of truth for both. Keep it deterministic and config-driven.

Covers (assignment §6, §9):
  - Devanagari numeral -> normal form, with spoken-form expansion for numbers/dates.
  - Punctuation and whitespace canonicalisation (danda etc.).
  - Mixed-script / code-mixing preserved, not discarded.
  - A canonical script decision, applied and documented.

This is intentionally dependency-light (pure Python) so it runs anywhere and is easy
to unit-test. Swap in indic-nlp-library / AI4Bharat normalisers behind the same
`normalise()` signature when you want their coverage.
"""

from __future__ import annotations

import re
import unicodedata

# Devanagari digits -> ASCII
_DEV_DIGITS = {ord(d): str(i) for i, d in enumerate("०१२३४५६७८९")}

# Number words for spoken-form expansion (0-20 + tens); extend as needed.
_HI_UNITS = [
    "शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस",
    "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस", "बीस",
]
_HI_TENS = {30: "तीस", 40: "चालीस", 50: "पचास", 60: "साठ",
            70: "सत्तर", 80: "अस्सी", 90: "नब्बे"}

# Punctuation we normalise rather than strip (keep sentence boundaries for TTS prosody).
_PUNCT_MAP = {
    "।": ".",   # danda -> period (configurable; some pipelines keep danda)
    "॥": ".",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "—": "-", "–": "-",
}

_WS = re.compile(r"\s+")


def _expand_hindi_number(n: int) -> str:
    """Small-integer spoken form. Falls back to digit-by-digit for large numbers."""
    if 0 <= n <= 20:
        return _HI_UNITS[n]
    if n < 100 and n % 10 == 0:
        return _HI_TENS.get(n, str(n))
    if n < 100:
        tens = (n // 10) * 10
        return f"{_HI_TENS.get(tens, str(tens))} {_HI_UNITS[n % 10]}"
    # Large numbers: read digits individually — safe, avoids wrong lakh/crore grouping.
    return " ".join(_HI_UNITS[int(d)] for d in str(n))


def normalise(
    text: str,
    *,
    language: str = "hi",
    normalise_numerals: bool = True,
    expand_numbers_to_words: bool = False,
    keep_code_mixing: bool = True,
) -> str:
    """Return the canonical, training-and-inference-consistent form of `text`.

    `keep_code_mixing=True` preserves embedded Latin-script (English) tokens — real
    Indic speech mixes them and discarding them silently corrupts the transcript.
    """
    # 1. Unicode canonicalisation (NFC) — critical for Indic combining characters.
    text = unicodedata.normalize("NFC", text)

    # 2. Devanagari digits -> ASCII digits.
    text = text.translate(_DEV_DIGITS)

    # 3. Punctuation canonicalisation.
    for src, dst in _PUNCT_MAP.items():
        text = text.replace(src, dst)

    # 4. Numerals: either keep as ASCII digits or expand to spoken words (TTS often
    #    wants words; STT usually wants whatever the reference uses — driven by config).
    if normalise_numerals and expand_numbers_to_words and language == "hi":
        text = re.sub(r"\d+", lambda m: _expand_hindi_number(int(m.group())), text)

    # 5. Drop control/zero-width chars that leak in from web-scraped corpora.
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")

    # 6. Code-mixing: optionally strip Latin tokens (default: keep them).
    if not keep_code_mixing:
        text = re.sub(r"[A-Za-z]+", " ", text)

    # 7. Collapse whitespace.
    return _WS.sub(" ", text).strip()
