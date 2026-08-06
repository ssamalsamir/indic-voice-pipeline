"""Stage 4 — Train / fine-tune.

Config-driven LoRA fine-tune of a registered base model on `manifest.jsonl`. Dispatches
on track (stt|tts) and architecture. Writes a checkpoint/adapter under the run dir plus
`train_meta.json` (base model, steps, loss curve pointer) for the model card.

Apple-Silicon reality: device defaults to `mps`, precision to fp32 (MPS is unstable in
fp16 for these ops), LoRA on by default so a Whisper-medium / Parler fine-tune actually
fits. Heavy imports are deferred so the rest of the pipeline imports without torch.
"""

from __future__ import annotations

import math
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

# --- VITS training constants (Kim et al. 2021, HiFi-GAN for the vocoder terms).
# The loss weights are the paper's, not tuned here: they balance terms whose natural
# magnitudes differ by two orders, and re-deriving them on 100 utterances would be
# fitting noise.
_W_MEL, _W_KL, _W_FM = 45.0, 1.0, 2.0
_N_MELS = 80
# 32 latent frames = 8192 samples at hop 256. The decoder trains on windows because
# running HiFi-GAN over a whole utterance every step does not fit in memory.
_SEGMENT_FRAMES = 32
_GRAD_CLIP = 5.0
# Discriminator + both optimiser states. Deleted on clean completion: it is training
# scaffolding, several times the generator's size, and nothing serves it.
_VITS_RESUME = "vits_resume.pt"


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
        # Split by INDEX rather than random_split, because train and dev must come from
        # two different dataset objects: SpecAugment is on for train and off for dev.
        # random_split returns Subsets of one shared object, so a single augment flag
        # would mask the dev split too and make eval_loss a moving target.
        perm = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(self.cfg.run.seed)
        ).tolist()
        aug = WhisperManifestDataset(manifest, processor, self.cfg.run.language,
                                     self.cfg.ingest.target_sr, augment=True,
                                     seed=self.cfg.run.seed)
        dev_ds = torch.utils.data.Subset(dataset, perm[:holdout])
        train_ds = torch.utils.data.Subset(aug, perm[holdout:])
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
        """VITS adversarial fine-tune. Consent is enforced by the config validator
        before we reach here.

        Not a Trainer subclass: VITS needs two optimisers stepped in a fixed order
        (discriminator on the detached waveform, then generator), which is exactly the
        thing HF Trainer's single-optimiser loop cannot express. A hand-written loop is
        shorter than the callbacks it would take to bend Trainer into this shape.
        """
        if spec.arch != "vits":
            raise NotImplementedError(
                f"TTS training is implemented for VITS; '{spec.key}' is {spec.arch}. "
                f"Parler is a LoRA fine-tune of a different shape — add it here."
            )
        self._require("torch", "torchaudio", "soundfile")
        import torch  # noqa: PLC0415
        from transformers import AutoTokenizer, VitsModel  # noqa: PLC0415

        from pipeline.tts_data import VitsCollator, VitsManifestDataset  # noqa: PLC0415
        from pipeline.vits_train import (  # noqa: PLC0415
            VitsDiscriminator, align_prior, discriminator_loss, feature_matching_loss,
            generator_loss, kl_loss, linear_spectrogram, mel_filterbank,
            mel_from_linear, rand_slice_segments,
        )

        tc = self.cfg.train
        dev = tc.device
        model = VitsModel.from_pretrained(spec.hf_id).to(dev).train()
        tok = AutoTokenizer.from_pretrained(spec.hf_id)
        cfg_m = model.config
        # hop is the product of the vocoder's upsample factors: it is how many audio
        # samples one latent frame becomes, so it must be derived, never assumed.
        hop = int(torch.tensor(cfg_m.upsample_rates).prod())
        n_fft = (cfg_m.spectrogram_bins - 1) * 2

        ds = VitsManifestDataset(manifest, tok, self.cfg.ingest.target_sr,
                                 n_fft=n_fft, hop=hop, log=self.log)
        if len(ds) == 0:
            raise ValueError("no usable (audio,text) rows in manifest")
        loader = torch.utils.data.DataLoader(
            ds, batch_size=tc.batch_size, shuffle=True, collate_fn=VitsCollator(hop),
            generator=torch.Generator().manual_seed(self.cfg.run.seed),
        )

        # Adapting the PRIOR side only (text encoder, duration predictor, flow) and
        # freezing the waveform side (posterior encoder, HiFi-GAN decoder).
        #
        # This is the recipe after a measured failure, not a guess. Training everything
        # adversarially from a RANDOMLY INITIALISED discriminator destroyed the voice:
        # output amplitude fell ~12x (rms 0.152 -> 0.013), utterance durations halved,
        # and intelligibility went from 0.16 to 1.04 WER. A fresh discriminator emits
        # meaningless gradients for its first few hundred steps, and a pretrained
        # vocoder has much further to fall than to climb. Real VITS runs survive this
        # because they train both nets together for ~1M steps; 390 steps on 100 clips
        # only gets the destructive half.
        #
        # Freezing the decoder also removes the adversarial and mel terms entirely
        # (nothing generates a waveform), leaving KL + duration: teach the text side to
        # predict what the audio side already encodes, and leave the vocoder alone.
        prior_only = bool(getattr(tc, "freeze_vocoder", False))
        if prior_only:
            for module in (model.decoder, model.posterior_encoder):
                for p in module.parameters():
                    p.requires_grad_(False)
            self.log.info("freeze_vocoder=true — adapting prior side only "
                          "(no discriminator, no adversarial or mel loss)")

        disc = None if prior_only else VitsDiscriminator().to(dev).train()
        opt_g = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=tc.lr, betas=(0.8, 0.99))
        opt_d = (None if prior_only
                 else torch.optim.AdamW(disc.parameters(), lr=tc.lr, betas=(0.8, 0.99)))
        mel_fb = mel_filterbank(self.cfg.ingest.target_sr, n_fft, _N_MELS, dev)
        seg_frames = _SEGMENT_FRAMES
        losses: list[dict] = []
        step, start_epoch = self._resume_vits(model, disc, opt_g, opt_d, losses)
        total_epochs = math.ceil(tc.epochs)  # float epochs: HF takes them, this loop can't

        for epoch in range(start_epoch, total_epochs):
            for batch in loader:
                batch = {k: v.to(dev) for k, v in batch.items()}
                wav, spec_lin = batch["waveform"], batch["spectrogram"]
                x_mask = batch["text_mask"].unsqueeze(1)     # [B,1,T_text]
                y_mask = batch["spec_mask"].unsqueeze(1)     # [B,1,T_spec]

                enc = model.text_encoder(batch["input_ids"],
                                         padding_mask=batch["text_mask"].unsqueeze(-1),
                                         attention_mask=batch["text_mask"].long())
                # transformers returns text-encoder tensors channels-LAST; everything
                # downstream is channels-FIRST. This transpose is load-bearing.
                m_p = enc.prior_means.transpose(1, 2) * x_mask
                logs_p = enc.prior_log_variances.transpose(1, 2) * x_mask

                z, m_q, logs_q = model.posterior_encoder(spec_lin, y_mask)
                z_p = model.flow(z, y_mask)
                m_exp, logs_exp, attn = align_prior(z_p, m_p, logs_p, x_mask, y_mask)

                l_kl = kl_loss(z_p, logs_q, m_exp, logs_exp, y_mask) * _W_KL
                l_dur = self._duration_loss(model, enc, batch, x_mask, attn)
                loss_d = torch.zeros(())
                l_mel = torch.zeros(())

                if prior_only:
                    loss_g = l_kl + l_dur
                else:
                    z_slice, starts = rand_slice_segments(z, batch["spec_len"],
                                                          seg_frames)
                    fake = model.decoder(z_slice)
                    real = torch.stack([wav[i, None, s * hop:(s + seg_frames) * hop]
                                        for i, s in enumerate(starts.tolist())])

                    # --- discriminator: never let generator grads flow through here
                    opt_d.zero_grad(set_to_none=True)
                    r_out, f_out, _, _ = disc(real, fake.detach())
                    loss_d = discriminator_loss(r_out, f_out)
                    loss_d.backward()
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), _GRAD_CLIP)
                    opt_d.step()

                    _, f_out, r_feat, f_feat = disc(real, fake)
                    mel_real = mel_from_linear(
                        linear_spectrogram(real.squeeze(1), n_fft, hop, n_fft), mel_fb)
                    mel_fake = mel_from_linear(
                        linear_spectrogram(fake.squeeze(1), n_fft, hop, n_fft), mel_fb)
                    l_mel = (mel_fake - mel_real).abs().mean() * _W_MEL
                    loss_g = (l_mel + l_kl + feature_matching_loss(r_feat, f_feat)
                              * _W_FM + generator_loss(f_out) + l_dur)

                # --- generator
                opt_g.zero_grad(set_to_none=True)
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP)
                opt_g.step()

                if dev == "mps":
                    torch.mps.empty_cache()
                if step % 25 == 0:
                    row = {"step": step, "epoch": epoch, "d": round(float(loss_d), 4),
                           "g": round(float(loss_g), 4), "mel": round(float(l_mel), 4),
                           "kl": round(float(l_kl), 4), "dur": round(float(l_dur), 4)}
                    losses.append(row)
                    self.log.info("step %(step)d  D=%(d).4f G=%(g).4f "
                                  "mel=%(mel).4f kl=%(kl).4f dur=%(dur).4f", row)
                step += 1

            # Checkpoint every epoch, not only at the end. The STT path already does
            # this and the TTS path has the same risk profile: a multi-hour adversarial
            # loop on a laptop that may sleep, or a hosted runner that may be evicted.
            # This project has already lost a full run to a Modal billing-limit kill,
            # so a loop whose only save is after the last step is a loop that throws
            # away everything on any interruption.
            self._save_vits(model, disc, opt_g, opt_d, step, epoch + 1, losses)

        model.save_pretrained(self.out_path)
        tok.save_pretrained(self.out_path)
        dump_json(self.out_path / "loss_curve.json", {"rows": losses})
        # The discriminator exists only to shape the generator; nothing serves it, and
        # it is ~4x the generator on disk.
        (self.out_path / _VITS_RESUME).unlink(missing_ok=True)
        return {
            "track": "tts", "base_model": spec.key, "base_hf_id": spec.hf_id,
            "arch": spec.arch, "device": dev, "lora": False,
            "epochs": tc.epochs, "n_train": len(ds), "steps": step,
            "voice_id": tc.voice_id,
            "final_losses": losses[-1] if losses else None,
            "recipe": "prior-only (vocoder frozen)" if prior_only else "full adversarial",
            "note": "discriminator is training scaffolding and is not shipped",
        }

    def _save_vits(self, model, disc, opt_g, opt_d, step, epoch, losses) -> None:
        """Snapshot everything needed to continue: both networks AND both optimiser
        states. Adam's moments are not incidental — resuming with fresh moments after
        a GAN has reached equilibrium destabilises it, so dropping them would make
        "resume" quietly mean "restart with a warm generator"."""
        import torch  # noqa: PLC0415

        model.save_pretrained(self.out_path)
        torch.save({"disc": disc.state_dict() if disc is not None else None,
                    "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict() if opt_d is not None else None,
                    "step": step, "epoch": epoch, "losses": losses},
                   self.out_path / _VITS_RESUME)
        self.log.info("checkpointed at epoch %d (step %d)", epoch, step)

    def _resume_vits(self, model, disc, opt_g, opt_d, losses) -> tuple[int, int]:
        """Restore a previous run if one is sitting in the run dir. Returns
        (step, start_epoch); (0, 0) when starting fresh."""
        import torch  # noqa: PLC0415
        from transformers import VitsModel  # noqa: PLC0415

        state_path = self.out_path / _VITS_RESUME
        if not state_path.exists():
            return 0, 0
        state = torch.load(state_path, map_location=self.cfg.train.device,
                           weights_only=False)
        # Generator weights live in the HF checkpoint that sits beside this file.
        model.load_state_dict(
            VitsModel.from_pretrained(self.out_path).state_dict())
        if disc is not None and state.get("disc") is not None:
            disc.load_state_dict(state["disc"])
        opt_g.load_state_dict(state["opt_g"])
        if opt_d is not None and state.get("opt_d") is not None:
            opt_d.load_state_dict(state["opt_d"])
        losses.extend(state.get("losses", []))
        self.log.info("resuming VITS from epoch %d (step %d)",
                      state["epoch"], state["step"])
        return int(state["step"]), int(state["epoch"])

    @staticmethod
    def _duration_loss(model, enc, batch, x_mask, attn):
        """Teach the duration predictor the alignment MAS just found.

        Durations are what inference has instead of the posterior: at synthesis time
        there is no audio, so the predictor's output is the only thing deciding how
        long each token is spoken. Train it badly and the voice stays intelligible in
        teacher-forced training and rushes or drawls at inference.

        The two predictor variants have DIFFERENT contracts, not just different
        internals. The stochastic one is a flow: hand it the durations and it returns
        a negative log-likelihood you minimise directly. The deterministic one is a
        regressor with no `durations` argument at all — it returns a log-duration
        prediction, so the loss has to be formed here. Calling the stochastic API on a
        deterministic checkpoint raises TypeError, which is why this branches on the
        config rather than on what today's single registered model happens to be.
        """
        import torch  # noqa: PLC0415
        durations = attn.sum(1).unsqueeze(1)          # [B,1,T_text] frames per token
        hidden = enc.last_hidden_state.transpose(1, 2).detach()

        if model.config.use_stochastic_duration_prediction:
            nll = model.duration_predictor(hidden, x_mask, durations=durations)
            return torch.sum(nll.float()) / torch.sum(x_mask).clamp_min(1.0)

        # Deterministic: MSE in LOG duration, because duration error is proportional,
        # not absolute — being two frames out on a long vowel is not the same mistake
        # as being two frames out on a stop consonant.
        log_pred = model.duration_predictor(hidden, x_mask)
        log_target = torch.log(durations.clamp_min(1e-6)) * x_mask
        return (torch.sum(((log_pred - log_target) * x_mask) ** 2)
                / torch.sum(x_mask).clamp_min(1.0))

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
