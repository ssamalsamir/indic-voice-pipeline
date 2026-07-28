"""Code-mixing detection + handling.

Real Indic speech freely mixes English ("मैंने meeting cancel कर दी"). The assignment
is explicit: handle it, do not discard it. We tag tokens by script so evaluation can
report code-mixed vs monolingual slices separately, and so TTS G2P can route Latin
tokens to an English fallback instead of mangling them through Devanagari rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_LATIN = re.compile(r"[A-Za-z]")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


@dataclass(frozen=True)
class Token:
    text: str
    script: str  # "latin" | "indic" | "other"


def script_of(token: str) -> str:
    if _DEVANAGARI.search(token):
        return "indic"
    if _LATIN.search(token):
        return "latin"
    return "other"


def tag(text: str) -> list[Token]:
    return [Token(t, script_of(t)) for t in text.split()]


def code_mix_ratio(text: str) -> float:
    """Fraction of alphabetic tokens that are Latin-script. 0 = monolingual Indic."""
    toks = [t for t in tag(text) if t.script in ("latin", "indic")]
    if not toks:
        return 0.0
    latin = sum(1 for t in toks if t.script == "latin")
    return latin / len(toks)


def is_code_mixed(text: str, threshold: float = 0.05) -> bool:
    return code_mix_ratio(text) >= threshold
