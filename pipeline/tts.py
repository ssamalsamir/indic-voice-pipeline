"""Reusable TTS synthesis — used by the evaluate stage and on-prem serving.

Mirrors pipeline/infer.py on the STT side: one class so evaluation and serving cannot
drift apart. Heavy imports stay lazy so the spine imports without torch.
"""

from __future__ import annotations

from pathlib import Path


class VitsSynthesiser:
    """facebook/mms-tts-* family. Small, low-latency, MPS-friendly.

    VITS is non-autoregressive, so one forward pass yields the whole waveform: no
    generate() loop, no beam search, and none of the memory growth that makes Whisper
    eval on MPS delicate.
    """

    def __init__(self, checkpoint: Path | None, base_hf_id: str, device: str = "mps",
                 seed: int = 1337):
        import torch  # noqa: PLC0415
        from transformers import VitsModel, VitsTokenizer  # noqa: PLC0415

        # A checkpoint arrives in one of TWO shapes and both must be honoured. VITS is
        # fine-tuned in FULL (save_pretrained -> config.json + model.safetensors),
        # while the STT side uses LoRA (adapter_config.json). Checking only for the
        # adapter file silently loads the BASE voice for a full fine-tune and reports
        # it as a result — which is exactly what happened here, and the run looked
        # entirely normal because a slightly different score is indistinguishable from
        # this model's own sampling noise.
        full = bool(checkpoint and (checkpoint / "config.json").exists())
        adapter = bool(checkpoint and (checkpoint / "adapter_config.json").exists())

        self.tokenizer = VitsTokenizer.from_pretrained(
            checkpoint if full else base_hf_id)
        model = VitsModel.from_pretrained(checkpoint if full else base_hf_id)
        if adapter:
            from peft import PeftModel  # noqa: PLC0415
            model = PeftModel.from_pretrained(model, checkpoint)
        self.is_finetuned = full or adapter
        self.model = model.to(device).eval()
        self.device = device
        self._torch = torch
        self._seed = seed
        self.sampling_rate = int(model.config.sampling_rate)

    def synthesise(self, text: str):
        """Return a float32 mono waveform at self.sampling_rate.

        Seeded per utterance. VITS's duration predictor is STOCHASTIC — it samples
        from a flow at inference — so the same text yields a different waveform every
        call, and the intelligibility score moves by ~0.02 WER between identical runs.
        Without a fixed seed, any fine-tuning effect smaller than that is unmeasurable
        and you cannot tell a real regression from a resample.
        """
        self._torch.manual_seed(self._seed)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            wav = self.model(**inputs).waveform
        out = wav.squeeze().float().cpu().numpy()
        if self.device == "mps":
            self._torch.mps.empty_cache()
        return out

    def to_wav(self, text: str, path: Path):
        import soundfile as sf  # noqa: PLC0415

        audio = self.synthesise(text)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio, self.sampling_rate)
        return path, len(audio) / self.sampling_rate
