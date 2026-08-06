"""Turn a training checkpoint into a servable artifact.

Kept out of the package stage because export is architecture-specific and noisy,
while the stage itself should stay about orchestration: export, time it, gate it,
write the card.

Why CTranslate2 rather than "ship the PyTorch checkpoint": a LoRA checkpoint is not
deployable on its own. It needs the base weights loaded alongside it, it runs the
HF generation loop, and on this target (Apple Silicon, no CUDA) that measured
RTF 4.47 against a 1.0 bar. Merging the adapter and converting to int8 removes the
adapter indirection, the Python decode loop, and 3/4 of the weight bytes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# faster-whisper reads these from the model directory, not from the ct2 graph, so the
# converter has to copy them across or loading fails with a bare FileNotFoundError.
_RUNTIME_FILES = (
    "tokenizer.json", "preprocessor_config.json", "tokenizer_config.json",
    "vocab.json", "merges.txt", "normalizer.json", "special_tokens_map.json",
    "added_tokens.json", "generation_config.json",
)


def export_whisper_ct2(checkpoint: Path, base_hf_id: str, artifact: Path,
                       quantise: bool, log) -> dict:
    """Merge the LoRA adapter into the base weights and convert to CTranslate2.

    Returns a manifest describing what was produced, which the model card renders
    so a reader can tell what actually ships versus what was trained.
    """
    import ctranslate2.converters  # noqa: PLC0415
    from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: PLC0415

    merged = artifact.parent / "_merged_hf"
    quantization = "int8" if quantise else "float32"

    if merged.exists():
        shutil.rmtree(merged)
    log.info("merging adapter into %s", base_hf_id)
    model = WhisperForConditionalGeneration.from_pretrained(base_hf_id)
    if (checkpoint / "adapter_config.json").exists():
        from peft import PeftModel  # noqa: PLC0415
        # merge_and_unload folds W + BA back into W, so the result is an ordinary
        # Whisper the converter understands. Without this the converter sees PEFT
        # wrapper modules it has no mapping for and silently exports the BASE model —
        # a passing RTF on a model that never learned anything.
        model = PeftModel.from_pretrained(model, checkpoint).merge_and_unload()
    model.save_pretrained(merged)

    # Prefer the checkpoint's own processor: a fine-tune can carry a different feature
    # extractor (large-v2 is 80 mel bins, large-v3 is 128) and the wrong one produces
    # fluent-sounding garbage rather than an error.
    src = checkpoint if (checkpoint / "preprocessor_config.json").exists() else base_hf_id
    WhisperProcessor.from_pretrained(src).save_pretrained(merged)

    copy = [f for f in _RUNTIME_FILES if (merged / f).exists()]
    log.info("converting to ctranslate2 (quantization=%s)", quantization)
    if artifact.exists():
        shutil.rmtree(artifact)
    ctranslate2.converters.TransformersConverter(
        str(merged), copy_files=copy, load_as_float16=False,
    ).convert(str(artifact), quantization=quantization, force=True)
    shutil.rmtree(merged, ignore_errors=True)

    size_mb = round(sum(p.stat().st_size for p in artifact.rglob("*") if p.is_file())
                    / 1e6, 1)
    log.info("exported %s MB to %s", size_mb, artifact)
    return {
        "format": "ctranslate2", "quantization": quantization,
        "adapter_merged": (checkpoint / "adapter_config.json").exists(),
        "base_hf_id": base_hf_id, "size_mb": size_mb, "runtime": "faster-whisper (CPU)",
    }


def load_ct2_whisper(artifact: Path, language: str):
    """Serving-side loader. Returns an object with the same `.transcribe(path, sr,
    num_beams)` shape as WhisperTranscriber so eval, RTF and serving stay one call."""
    from faster_whisper import WhisperModel  # noqa: PLC0415

    # Measured on this M4 (4 performance + 6 efficiency cores): 8 threads beat the
    # library default by ~12% and beat 10 threads by ~25% — past the performance cores
    # the scheduler starts handing work to efficiency cores and the batch waits on the
    # slowest of them. Not a universal constant; re-measure on different silicon.
    return _Ct2Transcriber(WhisperModel(str(artifact), device="cpu",
                                        compute_type="int8", cpu_threads=8), language)


class _Ct2Transcriber:
    """Adapter so the exported artifact is a drop-in for WhisperTranscriber.

    CTranslate2 has no Metal backend, so this is CPU-only on Apple Silicon. That is
    not a downgrade in practice: int8 on eight performance cores beats fp32 on MPS
    for this model, and it is what an on-prem box would actually run.
    """

    def __init__(self, model, language: str) -> None:
        self._model = model
        # faster-whisper wants the ISO code ("hi"), NOT the long name ("hindi") that
        # HF's generate() takes. Our configs already carry the ISO code, so pass it
        # straight through rather than round-tripping it through WHISPER_LANG.
        self._lang = language

    def transcribe(self, audio_path: str, target_sr: int = 16_000,
                   num_beams: int = 5) -> str:
        segments, _ = self._model.transcribe(
            str(audio_path), language=self._lang, task="transcribe",
            beam_size=num_beams,
            # The VAD filter drops audio it judges non-speech. On clean read speech it
            # occasionally eats a quiet first word, which is a WER regression the model
            # is not responsible for, so measure without it.
            vad_filter=False,
        )
        return " ".join(s.text for s in segments).strip()
