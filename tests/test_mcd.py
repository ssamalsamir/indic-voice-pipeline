"""Pin MCD's defining properties.

MCD is a number people compare against published figures, so the failure mode that
matters is a *plausible* wrong value: right units, wrong alignment or wrong
coefficients. These assert the properties that catch that.
"""

from __future__ import annotations

import numpy as np

from pipeline.mcd import _dtw_path, spectral_distance

SR = 16_000


def _tone(freq: float, secs: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(secs * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_identical_audio_scores_zero():
    wav = _tone(220.0)
    assert spectral_distance(wav, wav, SR) < 1e-6


def test_different_timbre_scores_higher_than_similar():
    ref = _tone(220.0)
    near = _tone(233.0)
    far = _tone(1800.0)
    assert (spectral_distance(ref, far, SR)
            > spectral_distance(ref, near, SR))


def test_symmetric():
    a, b = _tone(300.0), _tone(500.0)
    assert abs(spectral_distance(a, b, SR)
               - spectral_distance(b, a, SR)) < 1e-6


def test_tempo_change_is_absorbed_by_alignment():
    """The same tone held for twice as long is the same timbre. Without DTW this would
    score as badly different, which is the single most likely way to get MCD wrong."""
    short, long = _tone(300.0, 0.5), _tone(300.0, 1.0)
    stretched = spectral_distance(short, long, SR)
    unrelated = spectral_distance(short, _tone(1500.0, 0.5), SR)
    assert stretched < unrelated


def test_loudness_alone_barely_moves_it():
    """c0 is energy and is excluded, so halving the gain must not read as a spectral
    error. Guards against silently scoring coefficient 0."""
    wav = _tone(300.0)
    assert spectral_distance(wav, wav * 0.5, SR) < 1.0


def test_dtw_path_is_monotonic_and_spans_both_sequences():
    a = np.random.RandomState(0).randn(20, 4)
    b = np.random.RandomState(1).randn(31, 4)
    path = _dtw_path(a, b)
    assert path[0] == (0, 0) and path[-1] == (19, 30)
    for (i0, j0), (i1, j1) in zip(path, path[1:]):
        assert i1 >= i0 and j1 >= j0, "DTW path ran backwards"
        assert (i1 - i0) <= 1 and (j1 - j0) <= 1, "DTW path skipped a frame"
