"""Config schema — one YAML file fully describes one pipeline run.

A new language or requirement should mean a new config here, never a code change.
The schema is validated at load time (fail fast, clear errors) so a bad run never
reaches training. Both STT and TTS runs use the same top-level shape; track-specific
knobs live under `train.stt` / `train.tts`.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class Track(str, Enum):
    STT = "stt"
    TTS = "tts"


# ---------------------------------------------------------------- sub-configs


class RunCfg(BaseModel):
    name: str = Field(..., description="Unique run id; names the output dir.")
    track: Track
    language: str = Field(..., description="ISO 639 code, e.g. 'hi', 'mr'.")
    seed: int = 1337
    output_dir: Path = Path("runs")


class DataCfg(BaseModel):
    dataset: str = Field(..., description="Registry key, e.g. 'indicvoices'.")
    hf_path: str | None = Field(None, description="HF hub id if pulled from the hub.")
    hf_config: str | None = Field(
        None, description="HF dataset config/subset name, e.g. 'hi_in' for FLEURS. "
        "Defaults to run.language when unset."
    )
    local_dir: Path | None = Field(None, description="Path for a client drop instead.")
    licence: str = Field(..., description="Recorded verbatim in the model card.")
    consent: str | None = Field(
        None, description="Required for any voice cloning; else the run is refused."
    )
    train_split: str = "train"
    eval_split: str = "test"
    max_train_utts: int | None = Field(
        None, description="Cap for fast Mac iteration; None = all."
    )
    max_eval_utts: int | None = Field(
        50,
        description="Held-out clips to score. 50 keeps eval bearable on MPS but the "
                    "resulting WER is a ~50-sample estimate with a wide interval; on a "
                    "CUDA box set None to score the whole split and get a real number.",
    )


class IngestCfg(BaseModel):
    target_sr: int = 16_000
    mono: bool = True
    audio_format: Literal["wav", "flac"] = "wav"


class CleanCfg(BaseModel):
    denoise: bool = False
    trim_silence: bool = True  # energy/VAD-based
    min_duration_s: float = 0.5
    max_duration_s: float = 30.0
    # text normalisation — the Indic make-or-break
    canonical_script: str = Field(
        ..., description="e.g. 'Devanagari'; the one canonical form for this language."
    )
    normalise_numerals: bool = True
    expand_abbreviations: bool = True
    keep_code_mixing: bool = True  # do NOT discard English tokens


class AlignCfg(BaseModel):
    aligner: Literal["whisperx", "ctc", "mfa", "none"] = "whisperx"
    min_score: float = Field(0.5, description="Drop segments below this align confidence.")


class LoRACfg(BaseModel):
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] | None = None  # None → sensible per-arch default


class TrainCfg(BaseModel):
    base_model: str = Field(..., description="Registry key for the base to fine-tune.")
    device: Literal["mps", "cuda", "cpu"] = "mps"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"  # MPS is happiest at fp32
    lora: LoRACfg = LoRACfg()
    epochs: float = 3.0
    lr: float = 1e-4
    batch_size: int = 8
    grad_accum: int = 2
    warmup_ratio: float = 0.05
    eval_every_steps: int = 200
    # TTS-only
    voice_id: str | None = None
    speaker_ref_audio: Path | None = None

    @model_validator(mode="after")
    def _never_from_scratch(self) -> "TrainCfg":
        if not self.base_model:
            raise ValueError("base_model is required — always fine-tune, never scratch.")
        return self


class EvalCfg(BaseModel):
    metrics: list[str] = Field(
        ..., description="e.g. ['wer','cer'] for STT; ['asr_wer','mcd'] for TTS."
    )
    slices: list[str] = Field(
        default_factory=list, description="Report per-slice, e.g. ['domain','noise']."
    )
    # mentor-agreed bars; a run is flagged pass/fail against these
    thresholds: dict[str, float] = Field(default_factory=dict)
    asr_judge_run: str | None = Field(
        None,
        description="TTS only: run name whose STT checkpoint scores intelligibility "
                    "(synthesise -> transcribe -> WER). None uses the registry base "
                    "model untuned. Naming the judge in the config keeps the number "
                    "reproducible — an intelligibility score is meaningless unless you "
                    "know which ears measured it.",
    )
    asr_judge_hf_id: str | None = Field(
        None, description="TTS only: base model for the judge. Defaults to the Hindi "
                          "STT base in the registry.",
    )


class PackageCfg(BaseModel):
    quantise: bool = True
    serve_format: Literal["ctranslate2", "onnx", "torch", "gguf"] = "torch"
    rtf_target: float = Field(1.0, description="RTF must be <= this to pass the gate.")


# ---------------------------------------------------------------- top level


class PipelineConfig(BaseModel):
    run: RunCfg
    data: DataCfg
    ingest: IngestCfg = IngestCfg()
    clean: CleanCfg
    align: AlignCfg = AlignCfg()
    train: TrainCfg
    eval: EvalCfg
    package: PackageCfg = PackageCfg()

    @model_validator(mode="after")
    def _consent_required_for_cloning(self) -> "PipelineConfig":
        if self.run.track == Track.TTS and self.train.voice_id and not self.data.consent:
            raise ValueError(
                "TTS voice cloning requires an explicit consent record in data.consent "
                "— this is a governance line, not a formality."
            )
        return self

    @property
    def run_dir(self) -> Path:
        return self.run.output_dir / self.run.name

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)
