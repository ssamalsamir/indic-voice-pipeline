"""Task metrics. WER/CER are pure-Python (no deps) so eval always runs.

CER matters as much as WER for Indic: WER is unstable for agglutinative scripts where a
single orthographic word carries many morphemes, so a one-character slip nukes a whole
"word". We report both, always (assignment §8).
"""

from __future__ import annotations


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    """Levenshtein over token lists (words or chars)."""
    m, n = len(ref), len(hyp)
    if m == 0:
        return n
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = reference.split(), hypothesis.split()
    return _edit_distance(ref, hyp) / max(len(ref), 1)


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = list(reference.replace(" ", "")), list(hypothesis.replace(" ", ""))
    return _edit_distance(ref, hyp) / max(len(ref), 1)


def corpus_wer(pairs: list[tuple[str, str]]) -> float:
    """Aggregate WER over (ref, hyp) pairs — micro-averaged, the standard for ASR."""
    tot_err = tot_len = 0
    for ref, hyp in pairs:
        r = ref.split()
        tot_err += _edit_distance(r, hyp.split())
        tot_len += len(r)
    return tot_err / max(tot_len, 1)


def corpus_cer(pairs: list[tuple[str, str]]) -> float:
    tot_err = tot_len = 0
    for ref, hyp in pairs:
        r = list(ref.replace(" ", ""))
        tot_err += _edit_distance(r, list(hyp.replace(" ", "")))
        tot_len += len(r)
    return tot_err / max(tot_len, 1)
