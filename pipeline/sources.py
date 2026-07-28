"""Shared dataset source: stream a HF audio dataset, materialise clips to real .wav
files, and yield standard corpus rows. Used by BOTH the ingest stage (train split) and
the evaluate stage (held-out split) so they can't diverge.

Text-key detection reads the first actual row rather than trusting `.features`, which is
None for some streaming datasets — that bug silently produces empty transcripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

_TEXT_KEYS = ("sentence", "text", "transcript", "transcription", "raw_transcription")


def stream_hf_corpus(
    *, hf_id: str, hf_config: str, split: str, target_sr: int,
    audio_dir: Path, language: str, cap: int | None, offset: int = 0,
) -> Iterator[dict]:
    """Yield rows: {id, audio_path, text, duration_s, sr, speaker, domain, source}.

    Non-streaming + HF slice syntax: downloads the split archive ONCE, caches it (so
    reruns are instant), and materialises only `split[offset:offset+cap]`. Slicing lets
    a single cached download serve both a train slice and a disjoint eval slice — honest
    held-out with no second slow download.
    """
    import soundfile as sf  # noqa: PLC0415
    from datasets import Audio, load_dataset  # noqa: PLC0415

    audio_dir.mkdir(parents=True, exist_ok=True)
    split_expr = f"{split}[{offset}:{offset + cap}]" if cap else split
    dset = load_dataset(hf_id, hf_config, split=split_expr, trust_remote_code=True)
    dset = dset.cast_column("audio", Audio(sampling_rate=target_sr))
    text_key = _pick_text_key(dset[0])

    for i, ex in enumerate(dset):
        audio = ex["audio"]
        idx = offset + i
        wav_path = audio_dir / f"{language}_{split}_{idx:07d}.wav"
        sf.write(wav_path, audio["array"], target_sr)
        yield {
            "id": f"{language}_{split}_{idx:07d}",
            "audio_path": str(wav_path),
            "text": ex.get(text_key, "") or "",
            "duration_s": round(len(audio["array"]) / target_sr, 3),
            "sr": target_sr,
            "speaker": ex.get("client_id") or ex.get("speaker_id"),
            "domain": ex.get("domain", "general"),
            "source": f"{hf_id}:{hf_config}:{split_expr}",
        }


def _pick_text_key(example: dict) -> str:
    for k in _TEXT_KEYS:
        if k in example and isinstance(example[k], str):
            return k
    raise KeyError(
        f"no transcript column found in row keys {list(example)}; expected one of {_TEXT_KEYS}"
    )
