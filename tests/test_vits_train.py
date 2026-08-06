"""Pin the VITS training pieces that fail silently rather than loudly.

MAS is the reason this file exists. A wrong alignment does not raise — it produces a
plausible loss curve and a model that mumbles, and you only find out hours later via
intelligibility. These assertions are the properties the DP must hold whatever the
input: every frame emits exactly one token, tokens are never skipped or reordered,
and the path spans the whole utterance.
"""

from __future__ import annotations

import torch

from pipeline.vits_train import (
    align_prior, discriminator_loss, feature_matching_loss, kl_loss,
    linear_spectrogram, maximum_path, mel_filterbank, rand_slice_segments,
)


def _mask(b: int, t_s: int, t_t: int) -> torch.Tensor:
    return torch.ones(b, t_s, t_t)


def test_every_frame_maps_to_exactly_one_token():
    torch.manual_seed(0)
    path = maximum_path(torch.randn(3, 40, 7), _mask(3, 40, 7))
    assert torch.equal(path.sum(dim=2), torch.ones(3, 40)), "a frame must emit one token"


def test_alignment_is_monotonic_and_skips_nothing():
    torch.manual_seed(1)
    path = maximum_path(torch.randn(2, 50, 9), _mask(2, 50, 9))
    for b in range(2):
        idx = path[b].argmax(dim=1)
        steps = idx[1:] - idx[:-1]
        assert steps.min() >= 0, "alignment ran backwards"
        assert steps.max() <= 1, "alignment skipped a token"
        assert idx[0] == 0 and idx[-1] == 8, "path must span the whole utterance"


def test_every_token_is_spoken():
    torch.manual_seed(2)
    path = maximum_path(torch.randn(1, 30, 6), _mask(1, 30, 6))
    assert (path.sum(dim=1) > 0).all(), "a token was never spoken"


def test_padded_region_gets_no_alignment():
    """Real length 20 frames / 4 tokens inside a 30x6 padded batch."""
    mask = torch.zeros(1, 30, 6)
    mask[0, :20, :4] = 1.0
    path = maximum_path(torch.randn(1, 30, 6), mask)
    assert path[0, 20:, :].sum() == 0, "padding frames were aligned"
    assert path[0, :, 4:].sum() == 0, "padding tokens were aligned"


def test_alignment_prefers_the_likely_path():
    """A diagonal ridge in the likelihood must be recovered exactly."""
    t_s, t_t = 24, 6
    neg = torch.full((1, t_s, t_t), -10.0)
    want = torch.arange(t_s) // (t_s // t_t)
    neg[0, torch.arange(t_s), want] = 10.0
    path = maximum_path(neg, _mask(1, t_s, t_t))
    assert torch.equal(path[0].argmax(dim=1), want)


def test_spectrogram_frames_track_hop_exactly():
    """The segment slicer indexes waveform by spectrogram frame, so frames must be
    length/hop with no off-by-one — that misalignment is invisible in the loss."""
    hop, n_fft = 256, 1024
    wav = torch.randn(2, 64 * hop)
    assert linear_spectrogram(wav, n_fft, hop, n_fft).shape == (2, n_fft // 2 + 1, 64)


def test_expanded_prior_matches_frame_count():
    b, c, t_t, t_s = 2, 8, 5, 30
    m_exp, logs_exp, attn = align_prior(
        torch.randn(b, c, t_s), torch.randn(b, c, t_t), torch.randn(b, c, t_t),
        torch.ones(b, 1, t_t), torch.ones(b, 1, t_s),
    )
    assert m_exp.shape == (b, c, t_s) and logs_exp.shape == (b, c, t_s)
    # durations must account for every frame, or the duration predictor learns a
    # target that does not sum to the utterance length
    assert torch.allclose(attn.sum(1).sum(-1), torch.full((b,), float(t_s)))


def test_slice_is_fixed_length_even_for_short_utterances():
    z = torch.randn(3, 8, 20)
    out, starts = rand_slice_segments(z, torch.tensor([20, 5, 12]), 32)
    assert out.shape == (3, 8, 32)
    assert (starts >= 0).all()


def test_kl_vanishes_in_expectation_when_posterior_equals_prior():
    """VITS estimates the KL from ONE sample of q rather than in closed form, so it is
    unbiased in expectation, not zero pointwise — feed it the mean and it returns
    -0.5, which is the missing E[(z-m_q)^2/s_q^2]/2 term, not a bug. Averaging over
    many samples is the only assertion the estimator actually supports.
    """
    torch.manual_seed(0)
    m, logs, mask = torch.zeros(1, 4, 64), torch.zeros(1, 4, 64), torch.ones(1, 1, 64)
    vals = [kl_loss(torch.randn(1, 4, 64), logs, m, logs, mask) for _ in range(200)]
    assert abs(float(torch.stack(vals).mean())) < 0.05


def test_kl_grows_when_the_prior_drifts_from_the_posterior():
    torch.manual_seed(0)
    z, logs, mask = torch.randn(1, 4, 64), torch.zeros(1, 4, 64), torch.ones(1, 1, 64)
    near = kl_loss(z, logs, torch.zeros(1, 4, 64), logs, mask)
    far = kl_loss(z, logs, torch.full((1, 4, 64), 3.0), logs, mask)
    assert far > near


def test_discriminator_loss_rewards_correct_verdicts():
    perfect = discriminator_loss([torch.ones(2, 5)], [torch.zeros(2, 5)])
    wrong = discriminator_loss([torch.zeros(2, 5)], [torch.ones(2, 5)])
    assert perfect < wrong


def test_feature_matching_is_zero_on_identical_activations():
    feats = [[torch.randn(2, 3, 4), torch.randn(2, 3, 4)]]
    assert feature_matching_loss(feats, feats).abs() < 1e-6


def test_mel_filterbank_orientation():
    """[n_mels, n_freq] — the transpose of what torchaudio hands back. Wrong way round
    still matmuls when n_mels happens to divide, and yields a nonsense loss."""
    assert mel_filterbank(16_000, 1024, 80, "cpu").shape == (80, 513)
