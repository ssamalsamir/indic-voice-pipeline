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

### STT — Marathi (second language, config-only)

| Run | Base model | Train data | WER | CER | n | Verdict |
|---|---|---|---|---|---|---|
| `mr_stt_kathbath_large` | whisper-large-v3 | Kathbath 8k | 0.2014 | **0.0422** | 500 | FAIL (WER) |

CER clears its bar; WER does not. Full analysis in §4 — the short version is that
Marathi has no language-specialised base to start from, and that swap is the largest
single lever in this project.

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
| TTS (VITS) | torch, MPS | **0.062** | 492 ms | 898 ms | **PASS** |
| STT, isolated ~4s clips | CTranslate2 int8, CPU | **1.555** | 6,576 ms | 7,259 ms | **FAIL** |
| STT, 30s stream chunks | CTranslate2 int8, CPU | **0.623** | — | — | **PASS** |
| STT (before export) | LoRA checkpoint, MPS | 4.472 | 18,841 ms | 23,457 ms | FAIL |

Export bought **2.9x** on the per-clip figure and is the single largest latency win
available in software.

**The two STT rows differ by 2.5x and both are real.** Whisper's encoder always runs a
fixed 30s window, so a 4s clip costs nearly what a 30s clip costs: isolated short clips
spend most of their time encoding padding. Fed 30s chunks — how a streaming service
actually runs — the same model reaches 0.623 and clears the bar.

The gate deliberately uses the worse per-clip number. It is the plain definition, it
matches how the eval set is scored, and a gate should never be the flattering choice of
two. So the honest verdict is: **fails as a single-shot API on short utterances, passes
as a streaming service**, and §6 says what would close the gap outright.

One caveat learned the hard way: RTF must be measured on an idle machine. An earlier run
of this exact code reported the stream figure as 2.07 — *slower* than per-clip, which
inverted the conclusion — because a training job was saturating the CPU. Both numbers
above were reproduced twice on an idle box (1.549/0.634 and 1.555/0.623).

### Does the quantised artifact keep its accuracy?

Worth asking, because the headline WER was measured on the fp32 LoRA checkpoint while
what ships is a merged int8 model. A quantised artifact whose accuracy was never checked
is the same class of gap as a latency figure for a model nobody deploys.

| Artifact | WER | CER | n | Verdict |
|---|---|---|---|---|
| fp32 LoRA checkpoint (headline) | 0.0675 | 0.0206 | 500 | PASS |
| **CTranslate2 int8 (shipped)** | **0.0683** | **0.0208** | 500 | **PASS** |

Same 500 clips, same normaliser, same greedy decoding — only the precision differs.
**Quantisation costs 0.0008 WER**, which is nothing against a 0.10 bar. The shipped
artifact passes on its own measurement rather than by inference from the checkpoint's.

A note on stopping early. An interim run over the first 200 clips read 0.0751, and the
tempting read was "int8 costs ~0.008 WER". The right read was that a 200-clip prefix is
not the 500-clip set, so most of that gap was probably subset composition. It was: at
full n the gap is 10x smaller. A partial-set number is not a small-sample version of the
full-set number, it is a different measurement.

Independent corroboration from the TTS track, where int8 and fp32 judged the *same*
synthesised audio: 0.1547 vs 0.161. Quantisation cost the judge nothing measurable
there either.

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

`configs/mr_stt_kathbath_large.yaml` differs from the Hindi run by **exactly four
semantic lines** — `name`, `language`, `hf_config`, `base_model`. No code changes.

**The run completed.** 2 epochs over 8,000 Kathbath Marathi utterances, 9.4h on a Kaggle
GPU, scored on 500 held-out clips:

| Run | Base model | WER | CER | n | Verdict |
|---|---|---|---|---|---|
| `hi_stt_kathbath_large` | whisper-hindi-large-v2 | 0.0675 | 0.0206 | 500 | PASS |
| `mr_stt_kathbath_large` | whisper-large-v3 | **0.2014** | **0.0422** | 500 | **FAIL** (WER) |

`train_loss` 0.1631, `eval_loss` falling monotonically to 0.1108 — the fine-tune worked;
it converged.

**What this does and does not prove.** The pipeline claim is proven: a language it had
never processed ran ingest → clean → align → train → evaluate on a config diff alone,
and produced a scored, gated result with no code edit. That is acceptance #9.

The accuracy claim was never in scope, and the gap is exactly the one predicted before
the run: there is no Marathi-specialised base in the registry, so Marathi starts from
multilingual `whisper-large-v3` while Hindi starts from a model already fine-tuned on
Hindi. §2 identifies that swap as the single largest gain in the entire project, and
Marathi is the control that confirms it from the other direction — 0.2014 against
whisper-large-v3's 0.1712 on Hindi FLEURS is the same weight class, and 3x the Hindi
result on identical data, identical hyperparameters and identical code.

**CER tells the more useful story: 0.0422 passes the 0.08 bar comfortably.** A 4.8x
WER/CER ratio means the model has the Marathi phonology largely right and is losing
whole words — the signature of missing lexical and orthographic priors, not bad
acoustics. That is what a language-specialised base supplies.

The honest read: **the machine generalises; the model does not, and would not be
expected to.** Closing it means finding or building a Marathi-adapted base, which is a
data and model-sourcing problem, not a pipeline one.

One caveat on reproduction: the completed run predates the CTranslate2 export block
landing in this config, so its artifact is the raw checkpoint. Re-running `package`
produces the int8 artifact; no RTF was measured for Marathi.

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

**STT misses the RTF bar on short single-shot requests.** 1.555 against 1.0, after
export improved it 2.9x from 4.472. Streaming clears the bar at 0.623, so this is a
request-shape problem rather than a model-too-slow problem. Closing it outright means a
smaller model (whisper-medium is ~2.5x faster, at a WER cost this project has already
measured: 0.2893 vs 0.0675) or a GPU serving box — CTranslate2 has no Metal backend, so
the GPU on this machine is unavailable to it. Thread tuning is nearly exhausted: 8
threads beat the default by 12%, and 10 threads were 25% *worse* because work spills
onto efficiency cores.

**A measurement I got wrong, and how.** I predicted the fixed 30s encoder window would
make streaming much cheaper per second of audio, then measured streaming as *slower*
(2.07 vs 1.69) and wrote the prediction off as falsified. It was the measurement that
was wrong: a VITS training job was saturating the CPU during that run. Re-measured
twice on an idle machine, streaming is 2.5x cheaper exactly as predicted. The lesson is
narrow and worth keeping — a timing metric taken while anything else is running is not
a timing metric, and contention can move a number in whichever direction happens to
flip your conclusion.

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

1. **Close the STT RTF gap for short requests**, the only failed bar. Cheapest first:
   batch concurrent short requests into one 30s encoder window, which is the same
   mechanism that already gets streaming to 0.623 and needs no retraining. Failing
   that, fine-tune whisper-medium on Kathbath (the headroom between 0.0675 and the 0.10
   bar may absorb the accuracy loss — untested, and the cheapest experiment left) or
   move serving to a GPU box.
2. **A Marathi-specialised base model.** §4 shows the pipeline generalises and the base
   model does not. Hindi gained more from swapping to a Hindi-aware base than from any
   other change; Marathi is currently paying that cost in reverse. Either find an
   existing Marathi or Indic-multilingual fine-tune, or build one — this is the highest
   accuracy lever for any new language, ahead of hyperparameters.
3. **More TTS data before more TTS training.** §5 shows the training code works and the
   corpus is the binding constraint. A single-speaker corpus of a few thousand
   utterances would make fine-tuning meaningful; 100 multi-speaker clips cannot.
4. **Streaming ingest** to lift the 8,000-utterance cap and use all ~15GB of Kathbath.
   Data was the binding constraint at every point in this project.
5. **Beam search at scoring on GPU.** All reported numbers use greedy decoding, since
   beams push a 16GB Mac into swap. Beams typically buy 1-2 absolute WER points.
6. **Speed perturbation** to pair with the SpecAugment already shipped.
7. **LM rescoring** (shallow fusion with an n-gram or small LM) — the largest build
   here, and the standard next lever once acoustics are strong.
