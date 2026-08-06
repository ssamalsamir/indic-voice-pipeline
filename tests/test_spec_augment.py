"""SpecAugment must damage the input a bounded amount, and never the caller's array.

The dangerous failure is silent: masking in place corrupts the cached features so every
epoch sees a more-erased clip, and the model quietly trains on noise. These asserts pin
the copy semantics and the bounds.

Run: .venv/bin/python -m tests.test_spec_augment
"""

import numpy as np

from pipeline.data import _FREQ_MASKS, _FREQ_WIDTH, _TIME_MASKS, _TIME_WIDTH, spec_augment

MELS, FRAMES = 80, 3000


def _ones():
    return np.ones((MELS, FRAMES), dtype=np.float32)


def test_does_not_mutate_the_caller():
    original = _ones()
    spec_augment(original, np.random.default_rng(0))
    assert original.all(), "input was masked in place — every epoch would erase more"


def test_shape_is_preserved():
    out = spec_augment(_ones(), np.random.default_rng(1))
    assert out.shape == (MELS, FRAMES), out.shape


def test_something_actually_gets_masked():
    # Across seeds at least one mask must land, else augmentation is a no-op.
    assert any((spec_augment(_ones(), np.random.default_rng(s)) == 0).any()
               for s in range(10))


def test_masking_stays_within_budget():
    # Worst case: every mask at max width, no overlap. Anything more means the
    # bounds are wrong and we are destroying most of the signal.
    max_rows = _FREQ_MASKS * _FREQ_WIDTH
    max_cols = _TIME_MASKS * _TIME_WIDTH
    worst = 1.0 - ((MELS - max_rows) / MELS) * ((FRAMES - max_cols) / FRAMES)
    for s in range(25):
        frac = float((spec_augment(_ones(), np.random.default_rng(s)) == 0).mean())
        assert frac <= worst + 1e-6, f"seed {s} masked {frac:.3f} > budget {worst:.3f}"


def test_same_seed_same_mask():
    a = spec_augment(_ones(), np.random.default_rng(7))
    b = spec_augment(_ones(), np.random.default_rng(7))
    assert np.array_equal(a, b), "resumed runs must reproduce their augmentation"


def test_different_seeds_differ():
    a = spec_augment(_ones(), np.random.default_rng(1))
    b = spec_augment(_ones(), np.random.default_rng(2))
    assert not np.array_equal(a, b), "every clip sharing one mask defeats the point"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} passed")
