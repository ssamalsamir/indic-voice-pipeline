"""Dataset + collator for VITS fine-tuning.

Separate from `pipeline.data` (which is Whisper/STT-shaped: log-mel features and
label ids) because VITS wants something different enough that sharing would be a
lie: raw waveform, a LINEAR spectrogram for the posterior encoder, and character
ids from the VITS tokenizer rather than a BPE text target.
"""

from __future__ import annotations

from pathlib import Path

import torch

from pipeline.data import load_audio
from pipeline.utils.io import read_jsonl


def _audio_is_readable(path: str) -> bool:
    """Can this actually be opened? Reads the header only, not the samples."""
    try:
        import soundfile as sf  # noqa: PLC0415
        return sf.info(path).frames > 0
    except Exception:  # noqa: BLE001 - any failure to read means unusable
        return False


class VitsManifestDataset(torch.utils.data.Dataset):
    """One manifest row -> (input_ids, waveform, linear spectrogram).

    Rows whose audio cannot be opened are dropped at construction rather than raising
    mid-epoch: losing one clip of a hundred is not worth killing an hour of unattended
    training, but a silent zero-length tensor reaching the loss would be worse than
    either. The check opens each file's header once, which costs milliseconds against
    a run measured in hours.
    """

    def __init__(self, manifest: Path, tokenizer, target_sr: int,
                 n_fft: int, hop: int, log=None) -> None:
        rows = [r for r in read_jsonl(manifest)
                if r.get("audio_path") and (r.get("text") or "").strip()]
        self.rows = [r for r in rows if _audio_is_readable(r["audio_path"])]
        dropped = len(rows) - len(self.rows)
        if dropped and log is not None:
            # Say how many, always. A silent drop turns "my corpus is 100 clips" into
            # a number nobody can reconcile with the manifest.
            log.warning("dropped %d/%d rows with unreadable audio", dropped, len(rows))
        self.tokenizer = tokenizer
        self.sr, self.n_fft, self.hop = target_sr, n_fft, hop

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        from pipeline.vits_train import linear_spectrogram  # noqa: PLC0415

        row = self.rows[idx]
        wav = torch.from_numpy(load_audio(row["audio_path"], self.sr)).float()
        # Trim to a whole number of hops so spectrogram frames and waveform samples
        # stay in exact correspondence — the segment slicer indexes one by the other,
        # and an off-by-one hop there silently misaligns every training window.
        usable = (wav.shape[-1] // self.hop) * self.hop
        wav = wav[:usable]
        spec = linear_spectrogram(wav.unsqueeze(0), self.n_fft, self.hop,
                                  self.n_fft).squeeze(0)
        ids = self.tokenizer(row["text"], return_tensors="pt")["input_ids"][0]
        return {"input_ids": ids, "waveform": wav, "spectrogram": spec}


class VitsCollator:
    """Pad to the batch max and emit the masks the VITS graph needs.

    Both masks are float, not bool: they are multiplied into activations and summed
    as loss denominators, so a bool would need casting at every use site.
    """

    def __init__(self, hop: int) -> None:
        self.hop = hop

    def __call__(self, batch: list[dict]) -> dict:
        max_t = max(b["input_ids"].shape[0] for b in batch)
        max_s = max(b["spectrogram"].shape[-1] for b in batch)
        n_freq = batch[0]["spectrogram"].shape[0]
        n = len(batch)

        ids = torch.zeros(n, max_t, dtype=torch.long)
        text_mask = torch.zeros(n, max_t)
        spec = torch.zeros(n, n_freq, max_s)
        spec_mask = torch.zeros(n, max_s)
        wav = torch.zeros(n, max_s * self.hop)
        spec_len = torch.zeros(n, dtype=torch.long)

        for i, b in enumerate(batch):
            t, s = b["input_ids"].shape[0], b["spectrogram"].shape[-1]
            ids[i, :t] = b["input_ids"]
            text_mask[i, :t] = 1.0
            spec[i, :, :s] = b["spectrogram"]
            spec_mask[i, :s] = 1.0
            spec_len[i] = s
            w = b["waveform"][:s * self.hop]
            wav[i, :w.shape[-1]] = w

        return {"input_ids": ids, "text_mask": text_mask, "spectrogram": spec,
                "spec_mask": spec_mask, "waveform": wav, "spec_len": spec_len}
