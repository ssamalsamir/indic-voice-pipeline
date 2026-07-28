"""Registries: map short config keys to concrete base models and datasets.

Adding a base model or dataset = one entry here, not a rewrite of a stage. Keeps
configs terse ("base_model: whisper-small") and keeps licence/citation metadata in
one auditable place.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BaseModelSpec:
    key: str
    track: str  # "stt" | "tts"
    hf_id: str
    arch: str  # "whisper" | "wav2vec2" | "vits" | "parler" | ...
    # sensible LoRA targets per architecture (None → let PEFT auto-pick)
    lora_targets: tuple[str, ...] | None = None
    notes: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    hf_id: str | None
    languages: tuple[str, ...]
    licence: str
    has_transcripts: bool = True
    notes: str = ""


STT_MODELS = {
    "whisper-small": BaseModelSpec(
        "whisper-small", "stt", "openai/whisper-small", "whisper",
        lora_targets=("q_proj", "v_proj"),
        notes="MPS-feasible default; upgrade to medium/IndicWhisper on stronger HW.",
    ),
    "whisper-medium": BaseModelSpec(
        "whisper-medium", "stt", "openai/whisper-medium", "whisper",
        lora_targets=("q_proj", "v_proj"),
        notes="769M; ungated. Clear step up from small on Hindi. ~2.5x slower/step on MPS.",
    ),
    "indicwhisper": BaseModelSpec(
        "indicwhisper", "stt", "parthiv11/indic_whisper_multilingual", "whisper",
        lora_targets=("q_proj", "v_proj"),
        notes="whisper-medium fine-tuned on Indic; strongest Hindi start — but GATED "
              "(401 without an HF token). Needs HF_TOKEN to use.",
    ),
    "indicwav2vec": BaseModelSpec(
        "indicwav2vec", "stt", "ai4bharat/indicwav2vec-hindi", "wav2vec2",
        notes="CTC; pairs well with a KenLM decoder.",
    ),
}

TTS_MODELS = {
    "vits-hi": BaseModelSpec(
        "vits-hi", "tts", "facebook/mms-tts-hin", "vits",
        notes="Small, low-latency, MPS-friendly default. Good RTF story.",
    ),
    "indic-parler": BaseModelSpec(
        "indic-parler", "tts", "ai4bharat/indic-parler-tts", "parler",
        lora_targets=("q_proj", "v_proj"),
        notes="Higher quality / prompt-controllable; heavier — LoRA on MPS.",
    ),
}

DATASETS = {
    "indicvoices": DatasetSpec(
        "indicvoices", "ai4bharat/IndicVoices", ("hi", "mr", "bn", "ta"),
        "CC-BY-4.0 (verify per split)", notes="Read + conversational; code-mixed.",
    ),
    "kathbath": DatasetSpec(
        "kathbath", "ai4bharat/Kathbath", ("hi", "mr", "bn", "ta"),
        "CC-BY-4.0", notes="Read speech, many speakers — clean STT start.",
    ),
    "common_voice": DatasetSpec(
        "common_voice", "mozilla-foundation/common_voice_17_0", ("hi",),
        "CC0-1.0", notes="Crowd-sourced; variable quality — filter hard.",
    ),
    "fleurs": DatasetSpec(
        "fleurs", "google/fleurs", ("hi",),
        "CC-BY-4.0", notes="Clean, small; good held-out eval set.",
    ),
}


def get_base_model(key: str) -> BaseModelSpec:
    for table in (STT_MODELS, TTS_MODELS):
        if key in table:
            return table[key]
    raise KeyError(f"unknown base_model '{key}'. Known: "
                   f"{sorted({*STT_MODELS, *TTS_MODELS})}")


def get_dataset(key: str) -> DatasetSpec:
    if key not in DATASETS:
        raise KeyError(f"unknown dataset '{key}'. Known: {sorted(DATASETS)}")
    return DATASETS[key]
