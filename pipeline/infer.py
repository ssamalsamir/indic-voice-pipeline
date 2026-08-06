"""Reusable STT inference — used by both the evaluate stage and on-prem serving.

Loads a base Whisper model plus the LoRA adapter produced by training, and transcribes
audio. One class so eval and serving can't drift apart. Lazy heavy imports.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.data import WHISPER_LANG, load_audio


class WhisperTranscriber:
    def __init__(self, checkpoint: Path, base_hf_id: str, language: str,
                 device: str = "mps"):
        import torch  # noqa: PLC0415
        from peft import PeftModel  # noqa: PLC0415
        from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: PLC0415

        self.device = device
        self.language = language
        self.processor = WhisperProcessor.from_pretrained(
            checkpoint if (checkpoint / "preprocessor_config.json").exists() else base_hf_id,
            language=WHISPER_LANG.get(language, language),
            task="transcribe",
        )
        base = WhisperForConditionalGeneration.from_pretrained(base_hf_id)
        # adapter dir contains adapter_config.json when LoRA was used
        if (checkpoint / "adapter_config.json").exists():
            self.model = PeftModel.from_pretrained(base, checkpoint)
        else:
            self.model = base
        self.model.to(device).eval()
        # from_pretrained honours the checkpoint's stored dtype, so large-v3 loads as
        # fp16 while the feature extractor always emits fp32. Feeding one to the other
        # raises "Input type (float) and bias type (c10::Half) should be the same".
        # Read the dtype off the model rather than the config so this holds for any
        # base/adapter combination.
        self._dtype = next(self.model.parameters()).dtype
        self._torch = torch
        # Steer decoding with language=/task= rather than forced_decoder_ids: the latter
        # was REMOVED from generate() in transformers 5.x and raises "model_kwargs are
        # not used by the model", while language/task work in both 4.x and 5.x. Same
        # effect, one less version to be pinned to.
        self._lang_name = WHISPER_LANG.get(language, language)
        self._repair_generation_config()

    def _repair_generation_config(self) -> None:
        """Community Whisper fine-tunes often ship a pre-2023 generation config with no
        lang_to_id/task_to_id, and transformers 5 then refuses `language=` outright
        ("generation config is outdated"). Rebuild the maps from THIS model's own
        tokenizer rather than downloading a canonical config, so the ids are guaranteed
        to match the checkpoint's vocabulary (large-v2 and large-v3 differ here).
        """
        from transformers.models.whisper.tokenization_whisper import (  # noqa: PLC0415
            LANGUAGES,
        )

        gc = getattr(self.model, "generation_config", None)
        if gc is None or getattr(gc, "lang_to_id", None):
            return
        tok = self.processor.tokenizer
        unk = tok.unk_token_id

        def ids(tokens):
            out = {t: tok.convert_tokens_to_ids(t) for t in tokens}
            return {t: i for t, i in out.items() if i is not None and i != unk}

        # Key formats differ and are NOT interchangeable: lang_to_id is keyed by the
        # bracketed token ("<|hi|>"), task_to_id by the bare name ("transcribe") —
        # see generation_whisper.py, task_to_id[generation_config.task].
        gc.lang_to_id = ids(f"<|{c}|>" for c in LANGUAGES)
        gc.task_to_id = {t: i for t, i in
                         ((t, tok.convert_tokens_to_ids(f"<|{t}|>"))
                          for t in ("transcribe", "translate"))
                         if i is not None and i != unk}
        # Same rename: no_timestamps_token_id is required alongside the maps.
        if getattr(gc, "no_timestamps_token_id", None) is None:
            gc.no_timestamps_token_id = tok.convert_tokens_to_ids("<|notimestamps|>")

    def transcribe(self, audio_path: str, target_sr: int = 16_000,
                   num_beams: int = 5) -> str:
        """Transcribe one clip. `num_beams=1` is greedy; 5 is the Whisper default and
        typically buys 1-2 absolute WER points for a few seconds more per clip, which
        is a good trade on a GPU and a bad one on MPS."""
        audio = load_audio(audio_path, target_sr)
        inputs = self.processor.feature_extractor(
            audio, sampling_rate=target_sr, return_tensors="pt"
        ).input_features.to(self.device, dtype=self._dtype)
        with self._torch.no_grad():
            ids = self.model.generate(inputs, language=self._lang_name,
                                      task="transcribe",
                                      max_new_tokens=225, num_beams=num_beams)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
