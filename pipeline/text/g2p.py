"""Grapheme-to-phoneme for TTS.

Devanagari is largely phonetic, but schwa deletion and code-mixed Latin tokens are the
two things that break naive character-based synthesis. This module exposes a single
`g2p()` that a TTS training/inference path calls; Latin tokens are routed to an English
fallback rather than forced through Indic rules (see codemix.py for why).

For a production pass, back this with `indic-nlp-library` / AI4Bharat G2P behind the
same signature. The stub below keeps the interface stable so the pipeline is testable
end-to-end without the heavy dependency installed.
"""

from __future__ import annotations

from pipeline.text.codemix import script_of


def g2p(text: str, language: str = "hi") -> str:
    """Return a space-separated phoneme (or graphemic-proxy) string.

    Default implementation is a transparent passthrough per token that records which
    tokens need the English fallback, so wiring a real G2P later is a drop-in.
    """
    out: list[str] = []
    for tok in text.split():
        if script_of(tok) == "latin":
            out.append(_english_fallback(tok))
        else:
            out.append(_indic_g2p(tok, language))
    return " ".join(out)


def _indic_g2p(token: str, language: str) -> str:
    # Placeholder: real impl applies schwa-deletion + phoneme mapping.
    return token


def _english_fallback(token: str) -> str:
    # Placeholder: real impl uses g2p_en / CMUdict.
    return token.lower()
