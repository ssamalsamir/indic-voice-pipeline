"""Training data plumbing for the STT track.

Turns a `manifest.jsonl` into Whisper-ready tensors: audio -> log-mel input features,
normalised text -> label token ids. The text is ALREADY normalised by the clean stage,
so we never re-normalise here (that would reintroduce the train/inference skew the whole
normaliser exists to prevent).

Heavy deps (torch, transformers, soundfile) are imported lazily inside the functions so
importing this module never drags them in for the light spine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipeline.utils.io import read_jsonl

# Whisper wants full language names; map the ISO codes our configs use.
WHISPER_LANG = {
    "hi": "hindi", "mr": "marathi", "bn": "bengali", "ta": "tamil",
    "te": "telugu", "kn": "kannada", "ml": "malayalam", "gu": "gujarati",
    "pa": "punjabi", "or": "oriya", "as": "assamese", "ur": "urdu",
}


def load_audio(path: str, target_sr: int = 16_000):
    """Read an audio file to a mono float32 numpy array at target_sr."""
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    array, sr = sf.read(path, dtype="float32", always_2d=False)
    if array.ndim > 1:  # stereo -> mono
        array = array.mean(axis=1)
    if sr != target_sr:
        import torch  # noqa: PLC0415
        import torchaudio.functional as AF  # noqa: PLC0415
        array = AF.resample(torch.from_numpy(array), sr, target_sr).numpy()
    return np.asarray(array, dtype=np.float32)


class WhisperManifestDataset:
    """Lazy map-style dataset over a manifest for Whisper fine-tuning."""

    def __init__(self, manifest: Path, processor: Any, language: str, target_sr: int = 16_000):
        self.rows = [r for r in read_jsonl(manifest) if r.get("audio_path")]
        self.processor = processor
        self.language = language
        self.target_sr = target_sr

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        audio = load_audio(row["audio_path"], self.target_sr)
        features = self.processor.feature_extractor(
            audio, sampling_rate=self.target_sr
        ).input_features[0]
        labels = self.processor.tokenizer(row["text"]).input_ids
        return {"input_features": features, "labels": labels}


@dataclass
class SpeechCollator:
    """Pad audio features and label ids independently; mask pad tokens with -100 so they
    don't contribute to the loss. The standard Whisper seq2seq collator."""

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        import torch  # noqa: PLC0415

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Whisper prepends BOS in the tokenizer; drop it if present (Trainer re-adds it).
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
