"""Auto model-card generator — emitted for every run (assignment #7).

A model card is not decoration: it's the audit record. It pins the base model, the exact
data + licence + consent, the metrics vs the agreed bar, and the on-prem latency. If it
can't be generated automatically from the run artifacts, the run isn't reproducible.
"""

from __future__ import annotations

from pipeline.config import PipelineConfig
from pipeline.registry import get_base_model, get_dataset


def _method_lines(cfg: PipelineConfig, meta: dict) -> str:
    """Describe what training ACTUALLY did, from train_meta.json.

    The config records intent, and the two diverge: the TTS track carries
    `lora.enabled: true` in its config but VITS is fine-tuned in full, so a card
    rendered from the config alone claimed a LoRA adapter that does not exist. A
    model card that misreports the method is worse than one that omits it, because
    it is the artifact someone reads instead of the code.
    """
    if not meta:
        return (f"- Method: per config (run not yet trained)\n"
                f"- LoRA requested: enabled={cfg.train.lora.enabled}, "
                f"r={cfg.train.lora.r}, alpha={cfg.train.lora.alpha}")

    if meta.get("lora"):
        method = (f"LoRA adapter (r={cfg.train.lora.r}, alpha={cfg.train.lora.alpha}, "
                  f"targets={list(get_base_model(cfg.train.base_model).lora_targets or [])})")
    else:
        method = f"full fine-tune — {meta.get('recipe', 'all weights trainable')}"

    lines = [f"- Method: {method}",
             f"- Trained: {meta.get('steps', '?')} steps over "
             f"{meta.get('epochs', '?')} epochs on {meta.get('n_train', '?')} utterances",
             f"- Device: {meta.get('device', cfg.train.device)}  ·  "
             f"Precision: {cfg.train.precision}"]
    if meta.get("train_loss") is not None:
        lines.append(f"- Final train loss: {meta['train_loss']}")
    if meta.get("final_losses"):
        lines.append(f"- Final losses: {meta['final_losses']}")
    if meta.get("note"):
        lines.append(f"- Note: {meta['note']}")
    return "\n".join(lines)


def render_model_card(cfg: PipelineConfig, metrics: dict, latency: dict,
                      meta: dict | None = None) -> str:
    base = get_base_model(cfg.train.base_model)
    ds = get_dataset(cfg.data.dataset)
    consent = cfg.data.consent or "n/a (no voice cloning)"

    metric_lines = "\n".join(f"| {k} | {v} |" for k, v in metrics.items()) or "| — | — |"

    return f"""# Model Card — {cfg.run.name}

**Track:** {cfg.run.track.value.upper()}  ·  **Language:** {cfg.run.language}  ·  **Seed:** {cfg.run.seed}

## Base model (fine-tuned, not trained from scratch)
- **{base.key}** — `{base.hf_id}` ({base.arch})
{_method_lines(cfg, meta or {})}

## Data & governance
- **Dataset:** {ds.key} (`{ds.hf_id}`)
- **Licence:** {cfg.data.licence}
- **Voice consent:** {consent}
- Train utterances: {cfg.data.max_train_utts or "all"}  ·  Canonical script: {cfg.clean.canonical_script}
- Code-mixing: {"preserved" if cfg.clean.keep_code_mixing else "stripped"}

## Results
| metric | value |
|--------|-------|
{metric_lines}

**Agreed bar:** {cfg.eval.thresholds or "not set — agree with mentor before training"}

## On-prem latency ({latency.get("hardware", "?")})
- RTF: {latency.get("rtf")}  (gate ≤ {cfg.package.rtf_target} → **{latency.get("gate")}**)
- p50 / p95: {latency.get("p50_ms")} ms / {latency.get("p95_ms")} ms
- Serve format: {cfg.package.serve_format}  ·  Quantised: {cfg.package.quantise}

## Reproduce
```
python -m pipeline.run all --config configs/{cfg.run.name}.yaml
```

_Generated automatically by the pipeline. Do not hand-edit; change the config and re-run._
"""
