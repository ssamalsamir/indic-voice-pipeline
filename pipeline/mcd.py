"""MFCC-based spectral distance between reference and synthesised speech.

How far the synthesised spectral envelope sits from a real recording of the same
sentence. It complements the intelligibility WER in `evaluate` — a voice can be
perfectly intelligible and still sound nothing like the reference, and WER cannot see
that.

**This is NOT the MCD you find in papers, and is deliberately not named `mcd`.** True
mel-cepstral distortion is defined over mel-cepstral coefficients from mgcep/SPTK
analysis, whose magnitudes are ~0.1-1; the `10/ln(10)*sqrt(2)` constant is what maps
*those* into dB, landing published figures around 4-10. MFCCs have magnitudes ~5-20, so
the same formula over MFCCs lands ~20x higher. The value here is a sound RELATIVE signal
between our own runs and a meaningless one against the literature. Computing real MCD
needs an mcep implementation (pysptk), which is a new dependency, not a rename.

Two things make it more than a subtraction:

**Durations differ.** The synthesiser does not speak at the reference speaker's pace,
so frame *i* of one is not frame *i* of the other. The frames must be aligned first,
which is what the DTW below does. Comparing them index-by-index would mostly measure
the rate difference.

**c0 is excluded.** The zeroth cepstral coefficient is frame energy, i.e. how loud the
recording is. Including it means a quieter render scores as spectrally wrong, so the
convention is to score coefficients 1..N-1 only.

The `10/ln(10)*sqrt(2)` scaling is kept so the number is proportional to a true MCD
and moves the same way, but see the caveat above about its absolute magnitude.
"""

from __future__ import annotations

import numpy as np

# 13 coefficients, scoring 1..12. The standard MCD-13 configuration.
_N_MFCC = 13
_SCALE = (10.0 / np.log(10.0)) * np.sqrt(2.0)


def _mfcc(wav: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """`[T, n_mfcc]` cepstra. Torchaudio rather than librosa — already a dependency."""
    import torch  # noqa: PLC0415
    import torchaudio.transforms as T  # noqa: PLC0415, N812

    x = np.asarray(wav, dtype=np.float32)
    # Normalise to unit RMS FIRST. Dropping c0 is supposed to make this gain-invariant,
    # but only under a pure log: torchaudio computes log(mel + 1e-6), and for quiet mel
    # bins that floor dominates, so a level change stops being a clean c0-only offset
    # and leaks into c1..c12. Measured: halving the gain moved the score by 27 before
    # this line existed. Normalising makes the invariance structural instead of assumed.
    rms = float(np.sqrt((x ** 2).mean()))
    if rms > 0:
        x = x / rms

    tf = T.MFCC(sample_rate=sr, n_mfcc=_N_MFCC, log_mels=True,
                melkwargs={"n_fft": n_fft, "hop_length": hop, "n_mels": 80})
    out = tf(torch.from_numpy(x).unsqueeze(0))
    return out.squeeze(0).T.numpy()  # [T, n_mfcc]


def _dtw_path(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
    """Classic DTW over Euclidean frame distances. Returns the aligned index pairs.

    Same shape of dynamic program as the alignment search in `vits_train`, but the
    move set is different: DTW allows a diagonal step (both advance) as well as either
    side advancing alone, because neither sequence is a strict subdivision of the other.
    """
    n, m = len(a), len(b)
    dist = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        prev, cur = cost[i - 1], cost[i]
        for j in range(1, m + 1):
            cur[j] = dist[i - 1, j - 1] + min(prev[j], cur[j - 1], prev[j - 1])

    path, i, j = [], n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = int(np.argmin((cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])))
        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i, j = i - 1, j - 1
    return path[::-1]


def spectral_distance(ref: np.ndarray, syn: np.ndarray, sr: int) -> float:
    """MFCC-based spectral distance between two waveforms of the same sentence.

    Lower is closer. Interpret only against other numbers from this same function —
    it is not on the published MCD scale (see module docstring).
    """
    a, b = _mfcc(ref, sr), _mfcc(syn, sr)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("empty audio — cannot compute MCD")

    path = _dtw_path(a, b)
    # Drop c0 (energy): a quieter render is not a spectrally wrong one.
    diff = np.array([a[i, 1:] - b[j, 1:] for i, j in path])
    return float(_SCALE * np.sqrt((diff ** 2).sum(axis=1)).mean())
