# Indic Voice Pipeline

A reusable, **config-driven** pipeline that fine-tunes existing base models into
**STT** (speech→text) and **TTS** (text→speech) models for Indic languages.
A new language or requirement is a **new config + data — never a code change**.

Built for **Apple Silicon (MPS/MLX)**, first proven on **Hindi**. See [`PLAN.md`](PLAN.md)
for the 8-week roadmap and acceptance-criteria mapping.

## The spine

```
ingest → clean → align → train → evaluate → package
                                  └ auto: metrics.json + model_card.md + latency (RTF)
```

Six modular stages, one config file, each stage independently runnable.

## Quickstart

```bash
pip install -r requirements.txt          # spine deps only (pydantic, pyyaml, pytest)
pytest -q                                # GPU-free core is fully tested

# run one stage, or the whole thing, for a config
python -m pipeline.run ingest   --config configs/hi_stt_kathbath.yaml
python -m pipeline.run all      --config configs/hi_stt_kathbath.yaml
python -m pipeline.run evaluate --config configs/hi_stt_kathbath.yaml --force
```

The STT track is wired (Whisper LoRA on MPS). To run it:

```bash
pip install -r requirements.txt -r requirements-train.txt   # adds torch/transformers/peft
python -m pipeline.run all --config configs/hi_stt_kathbath.yaml
# -> runs/hi_stt_kathbath/{checkpoint, metrics.json, model_card.md, latency.json}
```

The spine (ingest→clean→align), text normalisation, metrics, and model-card generation
run with zero GPU. The TTS `train`/`evaluate` bodies land in Weeks 5–6.

## Layout

```
configs/     one YAML = one run (hi_stt, hi_tts, mr_stt reuse-proof)
pipeline/
  config.py      validated schema — the contract a config must satisfy
  registry.py    base-model + dataset keys (add here, not in stages)
  run.py         CLI: run a stage or 'all'
  stages/        ingest · clean · align · train · evaluate · package
  text/          normalise · g2p · codemix  (the Indic make-or-break)
  metrics.py     WER / CER (pure Python)
  modelcard.py   auto model card per run
model_cards/  emitted cards gallery
reports/      metrics + plots per run
serving/      on-prem wrapper + RTF harness
tests/        GPU-free core tests
```

## Design commitments (from the assignment)

- **Config or data, never code**, to add a language — proven by `mr_stt_kathbath.yaml`,
  which differs from the Hindi config only in language/data/name.
- **Fine-tune, never from scratch** — enforced in `config.py`.
- **Governance is a hard line** — dataset licence recorded per run; TTS voice cloning is
  refused without an explicit `data.consent` record.
- **Honest evaluation** — WER **and** CER, per-slice breakdowns, PASS/FAIL against a
  mentor-agreed bar set *before* training.
- **Deployability is a gate** — RTF measured on target hardware; over budget = FAIL in
  the model card, regardless of accuracy.
```
