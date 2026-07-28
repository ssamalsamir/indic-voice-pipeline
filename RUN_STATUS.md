# Live run status (2026-07-28)

## Goal
Actually train a Hindi STT model through the pipeline on this Mac (MPS).

## Latest — whisper-MEDIUM run COMPLETE ✅ (train_run11.log, 3h24m for 748 steps)
First successful medium run. FLEURS `train`, 1496 utts, 4 epochs, 748 steps, fp32 + LoRA.
- **train_loss:** 4.47 → **1.08** (vs run5's 3.51 — passed run5's *final* loss by step ~125)
- **Eval (39 held-out):** WER **0.2955**, CER **0.1322** → still **FAIL** vs bar (wer≤0.10)
- All three stages green: train → evaluate → package. RTF gate PASS (still a stub).

### What made medium trainable on 16GB (runs 6–11 were all the same job)
whisper-medium fp32 is **swap-bound, not compute-bound**. `gradient_checkpointing=True`
+ `gradient_checkpointing_kwargs={"use_reentrant": False}` took steps from an erratic
30–70s to a flat **~29s** and swap from 7.7GB → 3.0GB. That single flag is the run.

- **bf16 on MPS is WORSE — do not retry.** 0 steps in 10 min, 8.3GB swap, process wedged
  in uninterruptible page-in. Autocast adds cast copies instead of saving memory.
- **Cold-start steps lie.** Steps 1–3 of any run take 5–8 min (dataset load, first MPS
  alloc) before settling. Run7 was killed at step 337 on that misread; its checkpoint-300
  was recovered, and `train.py` now auto-resumes from the newest `checkpoint-*`.

### Ceiling
medium + 1.5k utts lands ~0.30. **0.10 is not reachable on this machine** — that needs
whisper-large-v3 or far more data than FLEURS Hindi has. Either relax the threshold or
treat 0.2955 as the hardware ceiling.

## Previous — scale-up run COMPLETE ✅ (train_run5.log, ~86 min incl. download)
FLEURS `train` split, 1500 utts (1496 usable), 4 epochs, 748 steps. No memory thrash.
- **train_loss:** 4.98 → **3.51** (still descending — more steps/data would keep helping)
- **Eval (39 held-out):** WER **0.3876**, CER **0.1636** → **FAIL** vs bar (wer≤0.10)
- Adapter: `runs/hi_stt_fleurs/checkpoint/adapter_model.safetensors`; RTF gate PASS (stub)

### Progression
| run | data | epochs | WER | CER |
|-----|------|--------|-----|-----|
| run4 | whisper-small, validation, 200 utts | 3 | 0.5126 | 0.227 |
| run5 | whisper-small, train, 1500 utts | 4 | 0.3876 | 0.1636 |
| run11 | **whisper-medium**, train, 1496 utts | 4 | **0.2955** | **0.1322** |

More data ⇒ clear improvement (−24% WER), but whisper-**small** on 1.5k Hindi utts tops
out ~0.39. Hitting 0.10 realistically needs whisper-medium/large or much more data.

## First full end-to-end run (train_run4.log)
- The run that proved the pipeline completes: 200 utts, 3 epochs, WER 0.5126.

## Dataset
- **FLEURS Hindi** (`google/fleurs`, config `hi_in`), CC-BY-4.0, ungated. `datasets` 2.21.0.
- train = `validation[0:200]`, eval = disjoint `validation[200:250]` (one cached download).

## Two blockers fixed this run
1. **`Seq2SeqTrainer(tokenizer=...)` crash** — transformers 5.x removed the arg.
   Now uses `processing_class=processor.feature_extractor`.
2. **MPS memory thrash → ~650s/step** — fp32 whisper-small at batch 4 exhausted the
   16GB M4 into 7GB swap; per-step time climbed 130→250s+. Fixed by:
   - config: `batch_size 4→1`, `grad_accum 2→8` (effective batch unchanged at 8)
   - `MPSCacheCallback` in `train.py` → `torch.mps.empty_cache()` each step
   Steps now flat at **~6s** (75 steps ≈ 8 min), swap stable.

## Why WER=0.51 (honest, not a bug)
- Decoding correctly forces Hindi+transcribe (verified in `infer.py`) — not a lang bug.
- whisper-**small** is a weak Hindi baseline (~0.4–0.6 WER zone) and 200 utts × 3 epochs
  barely moves it (loss still ~5). Genuine undertraining on a small base.

## Next lever to actually pass the bar (config-only, no code)
- More data: switch `train_split` to FLEURS `train` (~2k+ utts), raise `max_train_utts`.
- More epochs / steps (loss was still falling at step 75).
- Bigger base if needed: whisper-medium (fits with LoRA, slower per step).
- Then re-run `python -m pipeline.run all --config configs/hi_stt_fleurs.yaml`.

## Repo
- Not git-initialised yet. `git init` + first commit pending owner OK.

## Re-run
```
python -m pipeline.run all   --config configs/hi_stt_fleurs.yaml         # full pipeline
python -m pipeline.run <stage> --config configs/hi_stt_fleurs.yaml --force
```
