"""Stage 6 — Package & serve.

Export the fine-tuned model for on-prem serving, optionally quantise, measure RTF on
THIS machine (the target hardware), and emit the model card. RTF is a hard gate:
if it exceeds `package.rtf_target`, the card records a FAIL — a model that can't run
fast enough on-prem isn't deployable no matter how good the WER.

Produces:
    artifact/            exported model (format per config)
    latency.json         RTF + p50/p95 latency on target hardware
    model_card.md        auto-generated, includes licence + consent + metrics
"""

from __future__ import annotations

from pathlib import Path

from pipeline.modelcard import render_model_card
from pipeline.stages.base import Stage
from pipeline.utils.io import dump_json, read_jsonl


class PackageStage(Stage):
    name = "package"
    output = "model_card.md"

    def run(self) -> Path:
        ckpt = self.cfg.run_dir / "checkpoint"
        if not ckpt.exists():
            raise FileNotFoundError("run `train` first — checkpoint missing")

        artifact = self.cfg.run_dir / "artifact"
        artifact.mkdir(parents=True, exist_ok=True)
        self._export(ckpt, artifact)

        latency = self._measure_rtf(artifact)
        dump_json(self.cfg.run_dir / "latency.json", latency)
        self.log.info("RTF=%.3f (gate <= %.2f) -> %s",
                      latency["rtf"], self.cfg.package.rtf_target, latency["gate"])

        card = render_model_card(self.cfg, self._load_metrics(), latency)
        self.out_path.write_text(card, encoding="utf-8")
        # also drop a copy in the top-level model_cards/ gallery
        gallery = Path("model_cards") / f"{self.cfg.run.name}.md"
        gallery.parent.mkdir(exist_ok=True)
        gallery.write_text(card, encoding="utf-8")
        return self.out_path

    def _export(self, ckpt: Path, artifact: Path) -> None:
        fmt = self.cfg.package.serve_format
        self.log.info("export format=%s quantise=%s", fmt, self.cfg.package.quantise)
        # Wire per-format export: ctranslate2 (whisper), onnx (vits), torch.save, gguf.
        # Kept as a declared no-op stub so the spine runs; fill in Week 7.
        (artifact / "EXPORT_TODO.txt").write_text(
            f"export {ckpt} as {fmt}, quantise={self.cfg.package.quantise}\n"
        )

    def _measure_rtf(self, artifact: Path) -> dict:
        """RTF = compute_time / audio_duration, measured over a warm-up + timed loop on
        this machine. Stubbed value until the serving wrapper is wired (Week 7)."""
        target = self.cfg.package.rtf_target
        rtf = 0.0  # replace with measured value
        return {
            "hardware": "apple-silicon-mps",
            "rtf": rtf,
            "p50_ms": None,
            "p95_ms": None,
            "gate": "PASS" if rtf <= target else "FAIL",
            "measured": False,
        }

    def _load_metrics(self) -> dict:
        m = self.cfg.run_dir / "metrics.json"
        if not m.exists():
            return {}
        return {r: v for r, v in _flatten(list(read_jsonl_or_json(m)))}


# small helpers ---------------------------------------------------------------

def read_jsonl_or_json(path: Path):
    import json  # noqa: PLC0415
    text = path.read_text(encoding="utf-8")
    yield json.loads(text)


def _flatten(objs):
    obj = objs[0] if objs else {}
    for k, v in obj.items():
        if isinstance(v, (int, float, str)):
            yield k, v
