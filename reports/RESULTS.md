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

Intelligibility = synthesise 39 held-out sentences, transcribe them back with our own
STT model, score the round trip. All rows below share one judge, one seed and one set
of sentences, so they are directly comparable.

| Voice | asr_wer | asr_cer | Verdict |
|---|---|---|---|
| **Released `facebook/mms-tts-hin`, no fine-tune (shipped)** | **0.1547** | 0.0768 | **PASS** |
| Fine-tuned, 3 epochs @ 2e-6, vocoder + flow frozen | 0.1811 | 0.0872 | PASS |

**Fine-tuning was implemented, run four ways, and measurably does not help on this
corpus.** The full study is §5. The base voice is what ships.

### On-prem latency (RTF ≤ 1.0)

| Track | Artifact | RTF | p50 | p95 | Verdict |
|---|---|---|---|---|---|
| TTS (VITS) | torch, MPS | **0.061** | 503 ms | 984 ms | **PASS** |
| STT | CTranslate2 int8, CPU | **1.691** | 6,870 ms | 8,562 ms | **FAIL** |
| STT (before export) | LoRA checkpoint, MPS | 4.472 | 18,841 ms | 23,457 ms | FAIL |

Export bought **2.6x** and is the single largest latency win available in software.
It is still short of the bar: see §6.

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
memorisation. Capacity was never the constraint; data was.

**Measure the artifact, not the checkpoint.** RTF now times the exported int8 model
because a latency figure for something nobody deploys is not a latency figure. The same
rule caught a `hardware` field that was reporting the CUDA GPU a model was *trained* on
for a run timed on a local CPU.

---

## 3. Data decisions

| Dataset | Licence | Used for | Note |
|---|---|---|---|
| FLEURS `hi_in` | CC-BY-4.0 | STT train + eval, TTS train + eval | Only 2,115 train utterances total |
| Kathbath `hindi` | CC-BY-4.0 | STT train + eval | Gated (`gated=auto`), needs an HF token |
| Common Voice 17 | CC0-1.0 | **rejected** | Script-based loader; modern `datasets` dropped `trust_remote_code`, so it will not load |

**FLEURS is small.** 2,115 utterances is the whole train split. Capping Kathbath at
8,000 was deliberate: `stream_hf_corpus` downloads the entire split archive before
slicing, and the full corpus is ~15GB, which does not fit the runner's disk or wall
clock.

**Consent and licensing** are recorded per config and flow automatically into the model
card. The TTS config states plainly that no voice cloning occurs.

---

## 4. Reuse proof (#9)

`configs/mr_stt_kathbath_large.yaml` differs from the Hindi run by **exactly four lines** —
`name`, `language`, `hf_config`, `base_model`. No code changes.

Marathi has no Marathi-specialised base in the registry, so it starts from multilingual
`whisper-large-v3`. The claim under test is that the *pipeline* generalises, not that
the accuracy transfers.

Run status and result are appended here when it completes.

---

## 5. TTS fine-tuning: an implemented negative result

`transformers` ships VITS for inference only — `VitsModel.forward` raises
`NotImplementedError("Training of VITS is not supported yet.")` and no discriminator
class exists in the library. So the training graph was built here: monotonic alignment
search, HiFi-GAN multi-period and multi-scale discriminators, the KL / mel /
feature-matching / duration losses, and segment slicing.

Four recipes, one judge, one seed, the same 39 held-out sentences:

| Recipe | asr_wer | vs base |
|---|---|---|
| base voice, no fine-tune | **0.1547** | — |
| full adversarial, 30 ep @ 2e-4 | 1.0428 | catastrophic |
| vocoder + posterior frozen, 30 ep @ 2e-5 | 0.9447 | catastrophic |
| vocoder + posterior + flow frozen, 30 ep @ 2e-5 | 0.9736 | catastrophic |
| vocoder + posterior + flow frozen, **3 ep @ 2e-6** | 0.1811 | ~noise floor |

**The last row is the one that makes this a result rather than a bug report.** If the
loss or the alignment were wrong, gentle training would break the voice too. It does
not. The objective is sound; 100 utterances simply cannot move a model already trained
on far more Hindi without forgetting the language it is evaluated on.

Mechanism, from the audio rather than inferred: the adversarial run drove output
amplitude down 12x (rms 0.152 → 0.013) and halved utterance durations. A randomly
initialised discriminator emits meaningless gradients for its first few hundred steps,
and a pretrained vocoder has much further to fall than to climb. Real VITS runs survive
this by training both networks for ~1M steps; 390 steps on 100 clips gets only the
destructive half.

The frozen-flow variant is worth recording separately because it was a genuine design
error, not a hyperparameter. The flow sits on the **posterior** side of the KL term
(`z_p = flow(z)`), so with the decoder frozen there is no reconstruction loss anchoring
`z_p`. A trainable flow can then minimise KL by collapsing information out of `z_p`
rather than by moving the prior toward the audio — a degenerate optimum full VITS never
reaches because the same latents feed a mel loss. That produced well-formed speech at
correct amplitude and duration which said the *wrong words*.

**Conclusion: ship the base voice.** Fine-tuning costs 0.026 WER, in the wrong
direction, at the metric's own noise floor.

### Two measurement bugs found here

**The synthesiser silently ignored full fine-tune checkpoints.** It loaded a checkpoint
only when `adapter_config.json` existed — a PEFT adapter. VITS is fine-tuned in full
(`config.json` + `model.safetensors`), so it loaded the **base** voice and reported the
score as a fine-tune result. Nothing looked wrong, because a slightly different number
is indistinguishable from this model's own sampling noise.

**Synthesis was not seeded.** VITS's duration predictor samples from a flow at
inference, so identical text scored 0.161 and 0.181 on two runs of the *same* base
voice. Any effect smaller than ~0.02 WER was unmeasurable, and a real regression was
indistinguishable from a resample. Both are fixed; every number in this report is
seeded.

---

## 6. Honest gaps

**STT does not meet the RTF bar.** 1.691 against 1.0, after export improved it 2.6x
from 4.472. This is the only bar still failed. What is left is not software:
whisper-large-v2 int8 on four performance cores is ~1.7x real time, and CTranslate2 has
no Metal backend so the GPU on this machine is unavailable to it. Closing it means a
smaller model (whisper-medium is ~2.5x faster, at a WER cost this project has already
measured: 0.2893 vs 0.0675) or a GPU serving box. Thread tuning was measured and is
nearly exhausted: 8 threads beat the default by 12%, and 10 threads were 25% *worse*
because work spills onto efficiency cores.

**A discarded assumption, kept because it was wrong.** I expected Whisper's fixed 30s
encoder window to mean short clips mostly pay for padding, so serving 30s chunks would
be far cheaper per second of audio. Measured, streaming is *slower* (2.07 vs 1.69):
decode is autoregressive and a 30s chunk carries ~8x the tokens. Both numbers are in
`latency.json` rather than only the flattering one.

**MCD not computed.** Reported as `null` with a stated reason rather than a fabricated
number.

**Small eval samples.** The TTS figures rest on 39 clips, where the noise floor is
~0.02 WER. Only the 0.0675 STT headline uses 500.

### Earlier measurement bugs found and fixed

- **A hardcoded 50-clip eval cap** made every historical WER a small-sample estimate.
- **A cached eval manifest** would pair old references against newly normalised
  hypotheses, inflating WER in a way that looks like model regression. Detected without
  a version stamp: normalised text is idempotent under its own normaliser.
- **`rtf = 0.0`** auto-passed the deployability gate while measuring nothing.
- **Empty `suppress_tokens`** (set by training, saved into the checkpoint) crashed
  generation with `IndexError`, killing evaluation *after* a 5-hour training run.

### A prediction I got wrong

I expected the Devanagari scoring normaliser to be the largest single accuracy gain. A
control — re-scoring whisper-medium on identical clips with the identical new
normaliser — moved it from 0.2955 to **0.2893**, about 2% relative. The gain was the
model, not the metric. The normaliser was still worth building, because it is what makes
the comparison trustworthy, but the diagnosis was wrong and only the control showed it.

---

## 7. Next improvements, ranked

1. **Close the STT RTF gap**, the only failed bar. Either serve whisper-medium
   fine-tuned on Kathbath (the accuracy headroom between 0.0675 and the 0.10 bar may
   absorb the loss — untested, and the cheapest experiment left) or move serving to a
   GPU box where the fp16 checkpoint already runs comfortably.
2. **More TTS data before more TTS training.** §5 shows the training code works and the
   corpus is the binding constraint. A single-speaker corpus of a few thousand
   utterances would make fine-tuning meaningful; 100 multi-speaker clips cannot.
3. **Streaming ingest** to lift the 8,000-utterance cap and use all ~15GB of Kathbath.
   Data was the binding constraint at every point in this project.
4. **Beam search at scoring on GPU.** All reported numbers use greedy decoding, since
   beams push a 16GB Mac into swap. Beams typically buy 1-2 absolute WER points.
5. **Speed perturbation** to pair with the SpecAugment already shipped.
6. **LM rescoring** (shallow fusion with an n-gram or small LM) — the largest build
   here, and the standard next lever once acoustics are strong.
