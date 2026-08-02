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
        self._forced = self.processor.get_decoder_prompt_ids(
            language=WHISPER_LANG.get(language, language), task="transcribe"
        )

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
            ids = self.model.generate(inputs, forced_decoder_ids=self._forced,
                                      max_new_tokens=225, num_beams=num_beams)
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
