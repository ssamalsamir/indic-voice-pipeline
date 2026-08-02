"""Stage 4 — Train / fine-tune.

Config-driven LoRA fine-tune of a registered base model on `manifest.jsonl`. Dispatches
on track (stt|tts) and architecture. Writes a checkpoint/adapter under the run dir plus
`train_meta.json` (base model, steps, loss curve pointer) for the model card.

Apple-Silicon reality: device defaults to `mps`, precision to fp32 (MPS is unstable in
fp16 for these ops), LoRA on by default so a Whisper-medium / Parler fine-tune actually
fits. Heavy imports are deferred so the rest of the pipeline imports without torch.
"""

from __future__ import annotations

from pathlib import Path

from pipeline.registry import get_base_model
from pipeline.stages.base import Stage
from pipeline.utils.io import dump_json

# Checkpoint AND evaluate on this cadence — load_best_model_at_end requires both to
# fire on the same steps, so one constant drives both.
_CKPT_STEPS = 150
# Enough held-back rows for eval_loss to be a usable signal, few enough that we aren't
# throwing away training data we can't spare.
_HOLDOUT_FRACTION = 0.05


class TrainStage(Stage):
    name = "train"
    output = "checkpoint"  # a directory

    def run(self) -> Path:
        manifest = self.cfg.run_dir / "manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError("run `align` first — manifest.jsonl missing")

        spec = get_base_model(self.cfg.train.base_model)
        if spec.track != self.cfg.run.track.value:
            raise ValueError(
                f"base_model '{spec.key}' is a {spec.track} model but run.track="
                f"{self.cfg.run.track.value}"
            )
        self.log.info("fine-tuning %s (%s) on %s", spec.key, spec.arch, self.cfg.train.device)

        if self.cfg.run.track.value == "stt":
            meta = self._train_stt(spec, manifest)
        else:
            meta = self._train_tts(spec, manifest)

        self.out_path.mkdir(parents=True, exist_ok=True)
        dump_json(self.out_path / "train_meta.json", meta)
        return self.out_path

    # -- track implementations -------------------------------------------------

    def _train_stt(self, spec, manifest: Path) -> dict:
        """Whisper LoRA fine-tune via HF Seq2SeqTrainer on MPS.

        Text is already normalised by the clean stage; we never re-normalise here.
        """
        self._require("transformers", "peft", "torch", "torchaudio", "soundfile")
        import torch  # noqa: PLC0415
        from peft import LoraConfig, get_peft_model  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback,
            WhisperForConditionalGeneration, WhisperProcessor,
        )

        class MPSCacheCallback(TrainerCallback):
            """MPS's caching allocator holds freed blocks, so fp32 Whisper memory grows
            step over step until the 16GB M4 thrashes into swap (steps go 130s->250s->...).
            Emptying the cache each step keeps the footprint flat and steps fast."""

            def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        from pipeline.data import WHISPER_LANG, SpeechCollator, WhisperManifestDataset

        tc = self.cfg.train
        lang = WHISPER_LANG.get(self.cfg.run.language, self.cfg.run.language)
        processor = WhisperProcessor.from_pretrained(
            spec.hf_id, language=lang, task="transcribe"
        )

        model = WhisperForConditionalGeneration.from_pretrained(spec.hf_id)
        # let the model learn the language from data; don't force decoder ids in training
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

        if tc.lora.enabled:
            model = get_peft_model(model, LoraConfig(
                r=tc.lora.r, lora_alpha=tc.lora.alpha, lora_dropout=tc.lora.dropout,
                target_modules=list(spec.lora_targets or ["q_proj", "v_proj"]),
                bias="none",
            ))
            model.print_trainable_parameters()
        model.to(tc.device)

        dataset = WhisperManifestDataset(manifest, processor, self.cfg.run.language,
                                         self.cfg.ingest.target_sr)
        if len(dataset) == 0:
            raise ValueError("no usable (audio,text) rows in manifest — check ingest paths")
        collator = SpeechCollator(processor, model.config.decoder_start_token_id
                                  if not tc.lora.enabled
                                  else model.base_model.config.decoder_start_token_id)

        # Hold a slice back purely as an early-stopping signal. With a few thousand
        # utterances and several epochs the FINAL step is often past the overfitting
        # point, so without this we ship whichever weights step N happened to land on.
        # Not the eval set — that's a different split entirely, scored in stage 5.
        holdout = max(1, round(len(dataset) * _HOLDOUT_FRACTION))
        train_ds, dev_ds = torch.utils.data.random_split(
            dataset, [len(dataset) - holdout, holdout],
            generator=torch.Generator().manual_seed(self.cfg.run.seed),
        )
        self.log.info("train=%d dev=%d (dev is for checkpoint selection only)",
                      len(train_ds), len(dev_ds))

        args = Seq2SeqTrainingArguments(
            output_dir=str(self.out_path),
            per_device_train_batch_size=tc.batch_size,
            gradient_accumulation_steps=tc.grad_accum,
            learning_rate=tc.lr,
            warmup_ratio=tc.warmup_ratio,
            num_train_epochs=tc.epochs,
            # MPS is unstable in fp16 and measurably worse in bf16 (see the config
            # comment), so on MPS this stays fp32 whatever the config says. On CUDA
            # both are safe and fp16 is the point: it halves memory, which is what
            # makes whisper-large-v3 fit on a single A10G/A100.
            fp16=(tc.precision == "fp16" and tc.device == "cuda"),
            bf16=(tc.precision == "bf16" and tc.device == "cuda"),
            # 16GB M4 + whisper-medium fp32 is swap-bound, not compute-bound: stored
            # activations page to disk and steps blow out to minutes. Recomputing them
            # costs ~30% more compute and buys back most of that memory.
            # use_reentrant=False is required for grads to reach LoRA adapters.
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            logging_steps=25,
            save_strategy="steps",              # insurance: periodic checkpoints so a long
            save_steps=_CKPT_STEPS,             # MPS run can't lose everything if interrupted
            # load_best_model_at_end requires eval and save to land on the same steps.
            eval_strategy="steps",
            eval_steps=_CKPT_STEPS,
            per_device_eval_batch_size=tc.batch_size,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,                 # best + newest; HF protects the best one
            remove_unused_columns=False,        # our dataset yields custom keys
            label_names=["labels"],
            report_to=[],
            seed=self.cfg.run.seed,
        )
        trainer = Seq2SeqTrainer(
            model=model, args=args, train_dataset=train_ds, eval_dataset=dev_ds,
            data_collator=collator,
            processing_class=processor.feature_extractor,
            callbacks=[MPSCacheCallback()],
        )
        # Those periodic checkpoints are only insurance if we actually use them: a
        # multi-hour MPS run that dies at step 337 should restart at 300, not at 0.
        resume = any(self.out_path.glob("checkpoint-*"))
        if resume:
            self.log.info("resuming from newest checkpoint in %s", self.out_path)
        result = trainer.train(resume_from_checkpoint=resume or None)

        model.save_pretrained(self.out_path)       # LoRA adapter (or full weights)
        processor.save_pretrained(self.out_path)
        return {
            "track": "stt", "base_model": spec.key, "base_hf_id": spec.hf_id,
            "arch": spec.arch, "device": tc.device, "lora": tc.lora.enabled,
            "epochs": tc.epochs, "n_train": len(dataset),
            "train_loss": round(float(result.training_loss), 4),
            "steps": int(result.global_step),
        }

    def _train_tts(self, spec, manifest: Path) -> dict:
        """VITS/FastPitch (small) or Parler-TTS LoRA. Consent already enforced by the
        config validator before we reach here."""
        self._require("torch")
        raise NotImplementedError(
            "TTS training body — implement per architecture (Weeks 5-6). "
            "voice_id / speaker_ref come from cfg.train."
        )

    # -- helpers ---------------------------------------------------------------

    def _require(self, *modules: str) -> None:
        import importlib  # noqa: PLC0415
        missing = [m for m in modules if not _installed(importlib, m)]
        if missing:
            raise ImportError(
                f"training needs {missing} — install: pip install {' '.join(missing)}"
            )


def _installed(importlib, mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except ImportError:
        return False
