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
# Every corpus names its audio column differently: FLEURS "audio", Kathbath
# "audio_filepath". Resolve it from the schema rather than hardcoding, the same way
# _TEXT_KEYS already does for transcripts.
_AUDIO_KEYS = ("audio", "audio_filepath", "audio_path", "path", "file")


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
    audio_key = _pick_key(dset.column_names, _AUDIO_KEYS, "audio")
    dset = dset.cast_column(audio_key, Audio(sampling_rate=target_sr))
    text_key = _pick_text_key(dset[0])

    for i, ex in enumerate(dset):
        audio = ex[audio_key]
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


def _pick_key(columns, candidates, what: str) -> str:
    """First candidate present in `columns`. Fails loudly listing what WAS there, so a
    new corpus with an unknown column name is a one-line fix rather than a mystery."""
    for k in candidates:
        if k in columns:
            return k
    raise KeyError(
        f"no {what} column found in {list(columns)}; expected one of {candidates}"
    )


def _pick_text_key(example: dict) -> str:
    for k in _TEXT_KEYS:
        if k in example and isinstance(example[k], str):
            return k
    raise KeyError(
        f"no transcript column found in row keys {list(example)}; expected one of {_TEXT_KEYS}"
    )
