"""VITS fine-tuning: the training half that `transformers` does not ship.

`VitsModel.forward` raises NotImplementedError("Training of VITS is not supported
yet.") and no discriminator class exists in the library, so every piece below is
here because it has to be, not for taste:

* **Monotonic Alignment Search** — VITS has no external aligner. The text-to-frame
  alignment is recovered each step as the highest-likelihood monotonic path through
  the prior, which is a dynamic program, not a neural module.
* **MPD + MSD discriminators** — VITS is adversarial. Train the generator on
  reconstruction alone and the vocoder produces buzzy, metallic speech; the
  adversarial and feature-matching terms are what make it sound like a voice.
* **Segment slicing** — the decoder is trained on short random windows because
  running HiFi-GAN over a whole utterance every step does not fit in memory.

Shapes are the trap in this file. `transformers` returns text-encoder tensors
channels-LAST `[B, T, C]` while the posterior encoder, flow and decoder are all
channels-FIRST `[B, C, T]`. Mixing them does not raise — it broadcasts into a
silently wrong loss — so every transpose below is deliberate.

Reference: Kim et al. 2021, "Conditional Variational Autoencoder with Adversarial
Learning for End-to-End Text-to-Speech" (VITS), and the HiFi-GAN discriminators
it borrows.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

# HiFi-GAN's prime periods. Coprime windows make each sub-discriminator see a
# different periodic structure of the waveform, which is what catches the buzz a
# single discriminator misses.
_PERIODS = (2, 3, 5, 7, 11)
_LRELU = 0.1


# --------------------------------------------------------------- spectrograms


def linear_spectrogram(wav: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    """Magnitude STFT — the posterior encoder's input. `[B, n_fft//2+1, T]`."""
    pad = (n_fft - hop) // 2
    wav = F.pad(wav.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(wav, n_fft, hop_length=hop, win_length=win,
                      window=torch.hann_window(win, device=wav.device),
                      center=False, return_complex=True)
    return spec.abs().clamp_min(1e-9)


def mel_from_linear(spec: torch.Tensor, mel_fb: torch.Tensor) -> torch.Tensor:
    """Log-mel used only for the reconstruction loss. Log domain because L1 in linear
    magnitude is dominated by the loudest few bins and stops constraining formants."""
    return torch.log(torch.matmul(mel_fb, spec).clamp_min(1e-5))


def mel_filterbank(sr: int, n_fft: int, n_mels: int, device) -> torch.Tensor:
    """`[n_mels, n_freq]`, via torchaudio rather than librosa — torchaudio is already
    a training dependency and librosa would be a new one for a single matrix."""
    import torchaudio.functional as AF  # noqa: PLC0415, N812
    fb = AF.melscale_fbanks(n_fft // 2 + 1, 0.0, sr / 2, n_mels, sr,
                            norm="slaney", mel_scale="slaney")
    return fb.T.contiguous().to(device)


# ------------------------------------------------------- monotonic alignment


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Monotonic Alignment Search: the max-likelihood monotonic text->frame path.

    `neg_cent` is `[B, T_spec, T_text]`, `mask` the same shape. Returns a hard 0/1
    alignment of that shape.

    A Viterbi-style DP where the only legal moves are "stay on this text token" or
    "advance exactly one". Monotonicity is not a regulariser here, it is the reason
    the alignment is learnable at all: speech does not reorder its own phonemes, and
    forbidding skips is what stops the model collapsing onto a few loud tokens.

    Runs in numpy on CPU. It is O(T_spec * T_text) of inherently sequential work, so
    a GPU buys nothing, and it sits under `no_grad` anyway — the path is treated as
    an observation, not something to backprop through.
    """
    device, dtype = neg_cent.device, neg_cent.dtype
    value = (neg_cent * mask).detach().cpu().numpy().astype(np.float64)
    mask_np = mask.detach().cpu().numpy().astype(bool)
    b, t_s, t_t = value.shape
    path = np.zeros_like(value)
    # Per-item true lengths; padding must not be allowed to win the argmax.
    s_lens = mask_np.any(axis=2).sum(axis=1)
    t_lens = mask_np.any(axis=1).sum(axis=1)

    for i in range(b):
        n_s, n_t = int(s_lens[i]), int(t_lens[i])
        if n_s == 0 or n_t == 0:
            continue
        v = value[i, :n_s, :n_t]
        cum = np.full((n_s, n_t), -np.inf)
        cum[0, 0] = v[0, 0]
        for s in range(1, n_s):
            # stay on the same text token, or advance one. Nothing else is legal.
            prev_stay = cum[s - 1]
            prev_step = np.concatenate(([-np.inf], cum[s - 1, :-1]))
            cum[s] = np.maximum(prev_stay, prev_step) + v[s]
        # Backtrace from the last frame, which must land on the last text token:
        # every token has to be spoken, so the path is pinned at both ends.
        t = n_t - 1
        for s in range(n_s - 1, -1, -1):
            path[i, s, t] = 1.0
            if s > 0 and (t == 0 or cum[s - 1, t] >= cum[s - 1, t - 1]):
                continue
            if s > 0:
                t -= 1
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def rand_slice_segments(z: torch.Tensor, lengths: torch.Tensor, size: int):
    """Random fixed-length window per item, plus the frame offsets used to cut the
    matching waveform target."""
    b, _, t = z.shape
    max_start = (lengths - size).clamp_min(0)
    starts = (torch.rand(b, device=z.device) * (max_start + 1)).long()
    out = torch.stack([z[i, :, s:s + size] for i, s in enumerate(starts.tolist())])
    if out.shape[-1] < size:  # utterance shorter than one segment
        out = F.pad(out, (0, size - out.shape[-1]))
    return out, starts


# ------------------------------------------------------------- discriminators


def _norm_conv(c_in, c_out, k, s, p, groups=1, two_d=True):
    conv = (nn.Conv2d if two_d else nn.Conv1d)(c_in, c_out, k, s, p, groups=groups)
    return nn.utils.parametrizations.weight_norm(conv)


class _PeriodDiscriminator(nn.Module):
    """Folds the 1-D waveform into 2-D at a given period and looks at it with 2-D
    convs, so periodic artefacts at that period become spatial structure."""

    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period
        chans = [(1, 32), (32, 128), (128, 512), (512, 1024), (1024, 1024)]
        self.convs = nn.ModuleList([
            _norm_conv(i, o, (5, 1), (3, 1) if n < 4 else (1, 1), (2, 0))
            for n, (i, o) in enumerate(chans)
        ])
        self.post = _norm_conv(1024, 1, (3, 1), 1, (1, 0))

    def forward(self, x: torch.Tensor):
        b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - (t % self.period)), "reflect")
        x = x.view(b, c, -1, self.period)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), _LRELU)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1, -1), feats


class _ScaleDiscriminator(nn.Module):
    """Plain 1-D discriminator over the (optionally downsampled) waveform — catches
    the long-range envelope errors the period discriminators are blind to."""

    def __init__(self) -> None:
        super().__init__()
        spec = [(1, 128, 15, 1, 7, 1), (128, 128, 41, 2, 20, 4),
                (128, 256, 41, 2, 20, 16), (256, 512, 41, 4, 20, 16),
                (512, 1024, 41, 4, 20, 16), (1024, 1024, 41, 1, 20, 16),
                (1024, 1024, 5, 1, 2, 1)]
        self.convs = nn.ModuleList([
            _norm_conv(i, o, k, s, p, groups=g, two_d=False) for i, o, k, s, p, g in spec
        ])
        self.post = _norm_conv(1024, 1, 3, 1, 1, two_d=False)

    def forward(self, x: torch.Tensor):
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), _LRELU)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1, -1), feats


class VitsDiscriminator(nn.Module):
    """Multi-period + multi-scale, as HiFi-GAN/VITS use it. Discarded after training —
    it exists to shape the generator, and nothing ships it."""

    def __init__(self) -> None:
        super().__init__()
        self.nets = nn.ModuleList([_ScaleDiscriminator()]
                                  + [_PeriodDiscriminator(p) for p in _PERIODS])

    def forward(self, real: torch.Tensor, fake: torch.Tensor):
        r_out, f_out, r_feat, f_feat = [], [], [], []
        for net in self.nets:
            ro, rf = net(real)
            fo, ff = net(fake)
            r_out.append(ro), f_out.append(fo), r_feat.append(rf), f_feat.append(ff)
        return r_out, f_out, r_feat, f_feat


# -------------------------------------------------------------------- losses


def discriminator_loss(real_out, fake_out) -> torch.Tensor:
    """Least-squares GAN: real -> 1, fake -> 0. LSGAN rather than the log loss because
    it does not saturate, which matters when fine-tuning from an already-good
    generator that the discriminator can separate perfectly on step one."""
    loss = 0.0
    for r, f in zip(real_out, fake_out):
        loss = loss + ((1 - r) ** 2).mean() + (f ** 2).mean()
    return loss


def generator_loss(fake_out) -> torch.Tensor:
    return sum(((1 - f) ** 2).mean() for f in fake_out)


def feature_matching_loss(real_feat, fake_feat) -> torch.Tensor:
    """L1 between discriminator activations on real vs generated audio. This, not the
    adversarial term, carries most of the perceptual signal in HiFi-GAN training."""
    loss = 0.0
    for r_layers, f_layers in zip(real_feat, fake_feat):
        for r, f in zip(r_layers, f_layers):
            loss = loss + (r.detach() - f).abs().mean()
    return loss


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask) -> torch.Tensor:
    """KL(posterior || text-conditioned prior) — the term that ties what the model
    hears to what the text predicts. Without it the posterior carries the audio
    unaided and inference, which has only the prior, produces nothing intelligible."""
    kl = logs_p - logs_q - 0.5 + 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    return torch.sum(kl * z_mask) / torch.sum(z_mask).clamp_min(1.0)


def align_prior(z_p, m_p, logs_p, x_mask, y_mask):
    """Recover the hard alignment, then expand the per-token prior to per-frame.

    `m_p`/`logs_p` are `[B, C, T_text]`, `z_p` is `[B, C, T_spec]`. Returns the
    expanded prior plus the alignment (needed for the duration target).
    """
    with torch.no_grad():
        s_p_sq_r = torch.exp(-2 * logs_p)
        neg = (torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)
               + torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)
               + torch.matmul(z_p.transpose(1, 2), m_p * s_p_sq_r)
               + torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True))
        attn_mask = y_mask.transpose(1, 2) * x_mask  # [B, T_spec, T_text]
        attn = maximum_path(neg, attn_mask)
    m_exp = torch.matmul(attn, m_p.transpose(1, 2)).transpose(1, 2)
    logs_exp = torch.matmul(attn, logs_p.transpose(1, 2)).transpose(1, 2)
    return m_exp, logs_exp, attn
