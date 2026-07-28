# Indic Voice Pipeline — Completion Plan

Reusable, config-driven TTS **and** STT pipeline for Indic languages.
Target platform: **Apple Silicon (MPS/MLX)**. First language: **Hindi**.

Graded on two things in balance (per the assignment):
1. **The machine** — reproducible, modular, config-driven, genuinely reusable.
2. **The models** — one STT + one TTS model it produced that are *demoable*, not just "a run that completed".

---

## Non-negotiable design rules

- **A new language/requirement = new config + data. Never a code change.** (Acceptance #9.)
- **One config file drives one run.** Every stage reads the same config, takes a typed
  input, emits a typed output. Stages run independently or end-to-end.
- **Every run auto-emits a metrics report + a model card.** No manual bookkeeping.
- **Build the shared spine first (ingest → clean → align), generically**, assuming both
  tracks will bend it. Take STT all the way through as a *dry run*, then TTS reuses the spine.
- **Data governance is a hard line:** dataset licence recorded per run; explicit consent
  required for any voice cloning. Both land in the model card automatically.
- **Agree the quality bar with the mentor BEFORE training** (WER/CER/MOS/RTF thresholds).

---

## Architecture

```
Ingest → Clean/Normalise → Align/Segment → Train/Finetune → Evaluate → Package/Serve
                                                             ↑ auto: metrics report + model card
```

| Stage | Input | Output | GPU? |
|-------|-------|--------|------|
| 1 ingest   | dataset key / client drop | catalogued corpus (`corpus.jsonl`) | no |
| 2 clean    | corpus | clean `(text,audio)` pairs | no |
| 3 align    | clean pairs | training manifest (`manifest.jsonl`) | light |
| 4 train    | manifest + base model | LoRA adapter / checkpoint | **yes (MPS)** |
| 5 evaluate | checkpoint + held-out set | `metrics.json` + slices | light |
| 6 package  | checkpoint | on-prem artifact + `model_card.md` + RTF | no |

Stages 1–3 + text-normalisation + eval harness + model card are **CPU/Mac-native and
buildable now with zero GPU**. That is deliberately most of the surface area — the
assignment says most of the quality comes from data, not model code.

---

## Mac / Apple-Silicon compute strategy

Fine-tuning full Whisper-medium / Parler-TTS on MPS is impractical, so:

- **Train in PyTorch on the `mps` device, LoRA/PEFT everywhere.** Adapters are tiny,
  swappable, and cheap to iterate.
- **STT base default `openai/whisper-small` (LoRA)**, upgradeable to IndicWhisper /
  whisper-medium via config on stronger hardware. Eval/inference via `mlx-whisper` (fast).
- **TTS base default `VITS` / `FastPitch+HiFi-GAN` (small, low-latency, MPS-friendly)**,
  with **Indic Parler-TTS (LoRA)** as the higher-quality config option.
- Keep data subsets small first (fast full-pipeline loops), scale once the spine is proven.
- `RTF`/latency measured on this Mac — that IS the "target hardware" deployment gate.

---

## 8-week execution map

| Wk | Deliverable | Acceptance |
|----|-------------|-----------|
| 0 | **Agree metric bar with mentor** (WER/CER/MOS/RTF). Register datasets + licences. | §8 |
| 1–2 | Shared spine `ingest→clean→align` on Hindi; lock config schema; golden-path test. | #6 |
| 3–4 | STT track end-to-end (LoRA Whisper) → WER **&** CER + per-slice; demoable checkpoint. | #7 #8 |
| 5–6 | TTS track, one Hindi voice → intelligibility (ASR-WER) + small MOS panel + MCD; demo. | #7 #8 |
| 7 | Eval harness + auto model card polished; on-prem packaging; measure RTF for both. | #7 #10 |
| 8 | **Reuse proof:** second language (Marathi) via config + data only. Write-up. | #9 #11 |

## Definition of done (mirrors assignment §11)

- [ ] #6 Config-driven pipeline, each stage runs independently and end-to-end.
- [ ] #7 One STT + one TTS model, each with auto metrics report + model card.
- [ ] #8 Both clear the mentor-agreed bar — intelligible / usably accurate, demoable.
- [ ] #9 Second-language run via config + data only, no code changes.
- [ ] #10 On-prem packaging for both + measured latency/RTF.
- [ ] #11 Write-up: design, data decisions, results vs bar, next improvements.
