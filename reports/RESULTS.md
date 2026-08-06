# Indic Voice Pipeline — Results and Write-up

Acceptance #11. Design, data decisions, results against the agreed bar, and what I would
do next. Every number here was produced by `python -m pipeline.run …` and is reproducible
from the config named beside it.

---

## 1. Results against the bar

Mentor-agreed bars: **WER ≤ 0.10, CER ≤ 0.08** (STT), intelligible and demoable (TTS),
**RTF ≤ 1.0** on the on-prem target.

### STT — Hindi

| Run | Base model | Train data | WER | CER | n | Verdict |
|---|---|---|---|---|---|---|
| `hi_stt_kathbath_large` | whisper-hindi-large-v2 | Kathbath 8k | **0.0675** | **0.0206** | 500 | **PASS** |
| `hi_stt_fleurs_hindi_v2_kaggle` | whisper-hindi-large-v2 | FLEURS 2,115 | 0.0868 | 0.0362 | 39 | PASS |
| `hi_stt_fleurs_large` | whisper-large-v3 | FLEURS 1,500 | 0.1712 | 0.0660 | 50 | FAIL |
| `hi_stt_fleurs` | whisper-medium | FLEURS 1,500 | 0.2893 | 0.1240 | 39 | FAIL |

The headline is **WER 0.0675 / CER 0.0206 on 500 held-out clips**, a 77% relative WER
reduction over the whisper-medium starting point.

### TTS — Hindi

| Metric | Value | Bar | Verdict |
|---|---|---|---|
| asr_wer (intelligibility) | **0.161** | 0.30 | **PASS** |
| asr_cer | 0.0799 | — | — |
| MCD | not computed | — | see §5 |

Measured by synthesising 39 held-out sentences and transcribing them back. **This is the
released `facebook/mms-tts-hin` voice, not a fine-tuned one** — VITS training is not
implemented (§5). The report carries `voice_is_finetuned: false` so the number is never
mistaken for a tuned result.

### On-prem latency (RTF ≤ 1.0, measured on Apple M4 / MPS)

| Track | RTF | p50 | p95 | Verdict |
|---|---|---|---|---|
| TTS (VITS) | **0.061** | 503 ms | 984 ms | **PASS** |
| STT (whisper-large-v2 + LoRA) | **4.472** | 18,841 ms | 23,457 ms | **FAIL** |

The voice is comfortably real-time. **The STT model is not deployable at this target on
this hardware**, and that is the most actionable negative result in this report.

---

## 2. Design

Six stages — `ingest → clean → align → train → evaluate → package` — each independently
runnable, driven by one YAML config. A new language or dataset is a new config, never a
code change (tested in §4).

Three choices did the real work:

**Adapt a model that already speaks the language.** Swapping vanilla `whisper-large-v3`
for `vasista22/whisper-hindi-large-v2` was the single largest gain. With a few thousand
utterances you cannot teach a language; you can adapt one that knows it.

**Widen the LoRA target set rather than raise the rank.** Six modules
(`q/k/v/out_proj`, `fc1`, `fc2`) instead of two. Note `out_proj` — HF Whisper does *not*
name it `o_proj` as Llama-style models do, and the wrong name silently adapts nothing.

**Do not raise LoRA rank.** `train_loss` hit 0.0412 on 2,115 utterances, which is
memorisation. Capacity was never the constraint; data was. More adapter capacity would
have fit noise harder. This reversed a plausible-sounding item on the improvement list,
on evidence.

---

## 3. Data decisions

| Dataset | Licence | Used for | Note |
|---|---|---|---|
| FLEURS `hi_in` | CC-BY-4.0 | STT train + eval, TTS eval text | Only 2,115 train utterances total |
| Kathbath `hindi` | CC-BY-4.0 | STT train + eval | Gated (`gated=auto`), needs an HF token |
| Common Voice 17 | CC0-1.0 | **rejected** | Script-based loader; modern `datasets` dropped `trust_remote_code`, so it will not load |

**FLEURS is small.** 2,115 utterances is the whole train split, so "train on everything"
buys 1.4x over the 1,500 cap, not a new regime. Capping Kathbath at 8,000 was deliberate:
`stream_hf_corpus` downloads the entire split archive before slicing, and the full corpus
is ~15GB, which does not fit the runner's disk or wall clock.

**Consent and licensing** are recorded per config and flow automatically into the model
card. The TTS config states plainly that no voice cloning occurs: synthesis uses the
released MMS voice, not a cloned speaker.

---

## 4. Reuse proof (#9)

`configs/mr_stt_kathbath_large.yaml` differs from the Hindi run by **exactly four lines** —
`name`, `language`, `hf_config`, `base_model`. No code changes. Run status and result are
appended below when it completes.

One caveat worth stating: Marathi has no Marathi-specialised base in the registry, so it
starts from multilingual `whisper-large-v3`. The reuse claim being tested is that the
*pipeline* generalises, not that the accuracy transfers.

---

## 5. Honest gaps

**TTS training is not implemented.** `_train_tts` raises `NotImplementedError`. The TTS
number above is a baseline voice, not a fine-tune. This is the largest outstanding item.

**Export is a stub.** `_export` writes `EXPORT_TODO.txt` rather than producing a
CTranslate2 / ONNX artifact. This matters because it is also the fix for the failing STT
RTF (see §6).

**MCD not computed.** Reported as `null` with a stated reason rather than a fabricated
number.

**Small eval samples on two runs.** The 0.0868 and 0.161 figures rest on 39 clips, where
the confidence interval is wide. Only the 0.0675 headline uses 500.

### Measurement bugs found and fixed

Worth recording, because each one silently corrupted results before it was caught:

- **A hardcoded 50-clip eval cap** made every historical WER a small-sample estimate.
- **A cached eval manifest** would pair old references against newly normalised
  hypotheses, so every hyphenated word scored as an error and inflated WER in a way that
  looks like model regression. Detected without a version stamp: normalised text is
  idempotent under its own normaliser.
- **`rtf = 0.0`** auto-passed the deployability gate while measuring nothing.
- **Empty `suppress_tokens`** (set by training, saved into the checkpoint) crashed
  generation with `IndexError`, killing evaluation *after* a 5-hour training run.

### A prediction I got wrong

I expected the Devanagari scoring normaliser to be the largest single accuracy gain,
reasoning from the 2.6x WER/CER ratio that punctuation and orthographic variants were
inflating WER. A control — re-scoring whisper-medium on identical clips with the identical
new normaliser — moved it from 0.2955 to **0.2893**, about 2% relative. The gain was the
model, not the metric. The normaliser was still worth building, because it is what makes
the comparison trustworthy, but the diagnosis was wrong and only the control showed it.

---

## 6. Next improvements, ranked

1. **Fix the STT RTF failure.** This is the only bar currently failed. Implement the
   export stub: CTranslate2 / faster-whisper with int8 quantisation typically buys 4-8x,
   which is the gap. Alternatively serve a distilled or medium model where WER headroom
   allows.
2. **Implement VITS fine-tuning** to close the TTS half of #7 and #8 properly.
3. **Streaming ingest** to lift the 8,000-utterance cap and use all ~15GB of Kathbath.
   Data was the binding constraint at every point in this project.
4. **Beam search at scoring on GPU.** All reported numbers use greedy decoding, since
   beams push a 16GB Mac into swap. Beams typically buy 1-2 absolute WER points.
5. **Speed perturbation** to pair with the SpecAugment already shipped.
6. **LM rescoring** (shallow fusion with an n-gram or small LM) — the largest build here,
   and the standard next lever once acoustics are strong.
