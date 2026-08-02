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
    "whisper-large-v3": BaseModelSpec(
        "whisper-large-v3", "stt", "openai/whisper-large-v3", "whisper",
        # Attention (all four) + both MLP projections. HF Whisper names the attention
        # output `out_proj`, NOT `o_proj` as Llama-style models do — the wrong name
        # adapts nothing. Widening the target set multiplies adapter capacity for
        # almost no extra memory or step time, which beats raising `r` when the
        # corpus is only a few thousand utterances.
        lora_targets=("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"),
        notes="1.55B; ungated. Too big for a 16GB M4 — needs a CUDA GPU with fp16 "
              "(A100/A10G). The accuracy headroom past whisper-medium lives here.",
    ),
    "whisper-hindi-large-v2": BaseModelSpec(
        "whisper-hindi-large-v2", "stt", "vasista22/whisper-hindi-large-v2", "whisper",
        lora_targets=("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"),
        notes="large-v2 already fine-tuned on Hindi (Apache-2.0, ungated). Adapting a "
              "Hindi-aware model beats teaching Hindi to vanilla large-v3 from a few "
              "thousand utterances. Same size class as large-v3, so L4 batch settings "
              "carry over — but 80 mel bins vs large-v3's 128, which is fine because "
              "the feature extractor is loaded from this checkpoint, not hardcoded.",
    ),
    "indicwhisper": BaseModelSpec(
        "indicwhisper", "stt", "parthiv11/indic_whisper_multilingual", "whisper",
        lora_targets=("q_proj", "v_proj"),
        notes="BROKEN REPO — do not use. Previously annotated 'gated, needs HF_TOKEN'; "
              "that was a guess from a 401. HF returns the same 401 for repos that do "
              "not exist, and this id is absent from public model search while the "
              "same author's other repos appear, so a token will NOT fix it. Use "
              "whisper-hindi-large-v2 above instead.",
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
