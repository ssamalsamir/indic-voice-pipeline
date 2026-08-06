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
            # Same rule as evaluate: a released TTS voice is a shippable baseline worth
            # packaging and timing; an unfine-tuned STT run has nothing to package.
            if self.cfg.run.track.value == "stt":
                raise FileNotFoundError("run `train` first — checkpoint missing")
            self.log.warning("no checkpoint — packaging the BASE voice")

        artifact = self.cfg.run_dir / "artifact"
        artifact.mkdir(parents=True, exist_ok=True)
        self._export(ckpt, artifact)

        latency = self._measure_rtf(artifact)
        dump_json(self.cfg.run_dir / "latency.json", latency)
        if latency["measured"]:
            self.log.info("RTF=%.3f (gate <= %.2f) -> %s  [p50=%sms p95=%sms over %s clips]",
                          latency["rtf"], self.cfg.package.rtf_target, latency["gate"],
                          latency["p50_ms"], latency["p95_ms"], latency["n_clips"])
        else:
            self.log.warning("RTF UNMEASURED — the card will say so rather than claim a pass")

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

    # Enough clips for a stable p95 without turning `package` into another eval run.
    _RTF_CLIPS = 12

    def _measure_rtf(self, artifact: Path) -> dict:
        """RTF = compute_time / audio_duration over a warm-up plus timed loop.

        Measured on THIS machine because that is the on-prem target: an RTF from a
        datacentre GPU says nothing about whether the thing runs on the box it ships to.
        Returns measured=False (and never a PASS) when it genuinely could not measure —
        a hardcoded 0.0 that auto-passes the gate is worse than an honest gap, because
        it reports the requirement as met while proving nothing.
        """
        import statistics  # noqa: PLC0415
        import time  # noqa: PLC0415

        target = self.cfg.package.rtf_target
        unmeasured = {
            "hardware": self.cfg.train.device, "rtf": None,
            "p50_ms": None, "p95_ms": None, "gate": "UNMEASURED", "measured": False,
        }

        is_stt = self.cfg.run.track.value == "stt"
        rows = self._rtf_clips(need_audio=is_stt)
        if not rows:
            self.log.warning("no clips available to time — RTF left unmeasured")
            return unmeasured

        try:
            run_once = self._timed_call(is_stt)
        except Exception as exc:  # noqa: BLE001 - packaging must not die on RTF
            self.log.warning("could not load model for RTF (%s) — leaving unmeasured", exc)
            return unmeasured

        run_once(rows[0])  # warm-up: the first call pays lazy init and kernel compile

        latencies, audio_s = [], 0.0
        for row in rows:
            t0 = time.perf_counter()
            produced = run_once(row)
            latencies.append((time.perf_counter() - t0) * 1000)
            # STT consumes the clip's audio; TTS PRODUCES it, so the denominator is the
            # synthesised duration, not anything that existed beforehand.
            audio_s += float(row.get("duration_s") or 0.0) if is_stt else produced

        if audio_s <= 0:
            self.log.warning("clips carry no duration — RTF left unmeasured")
            return unmeasured

        rtf = (sum(latencies) / 1000) / audio_s
        return {
            "hardware": self.cfg.train.device,
            "rtf": round(rtf, 4),
            "p50_ms": round(statistics.median(latencies), 1),
            "p95_ms": round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 1),
            "n_clips": len(latencies),
            "audio_s": round(audio_s, 1),
            "gate": "PASS" if rtf <= target else "FAIL",
            "measured": True,
        }

    def _rtf_clips(self, need_audio: bool = True) -> list[dict]:
        """Prefer the held-out eval rows; fall back to the train manifest.

        STT needs real audio on disk to consume; TTS needs only text, since it makes the
        audio itself.
        """
        def usable(r: dict) -> bool:
            if need_audio:
                return bool(r.get("audio_path")) and Path(r["audio_path"]).exists()
            return bool((r.get("text") or "").strip())

        for name in ("eval_manifest.jsonl", "manifest.jsonl"):
            path = self.cfg.run_dir / name
            if not path.exists():
                continue
            rows = [r for r in read_jsonl(path) if usable(r)]
            if rows:
                return rows[:self._RTF_CLIPS]
        return []

    def _timed_call(self, is_stt: bool):
        """Return f(row) -> seconds of audio produced (TTS) or 0.0 (STT)."""
        if is_stt:
            transcriber = self._build_transcriber()

            def stt(row):
                transcriber.transcribe(row["audio_path"], self.cfg.ingest.target_sr,
                                       num_beams=1)
                return 0.0
            return stt

        from pipeline.registry import get_base_model  # noqa: PLC0415
        from pipeline.tts import VitsSynthesiser  # noqa: PLC0415

        ckpt = self.cfg.run_dir / "checkpoint"
        synth = VitsSynthesiser(ckpt if ckpt.exists() else None,
                                get_base_model(self.cfg.train.base_model).hf_id,
                                self.cfg.train.device)

        def tts(row):
            audio = synth.synthesise(row["text"])
            return len(audio) / synth.sampling_rate
        return tts

    def _build_transcriber(self):
        import json  # noqa: PLC0415

        from pipeline.infer import WhisperTranscriber  # noqa: PLC0415
        from pipeline.registry import get_base_model  # noqa: PLC0415

        # Prefer what training actually used; the config is only a fallback, since a
        # rescued checkpoint can outlive the config that produced it.
        meta_path = self.cfg.run_dir / "checkpoint" / "train_meta.json"
        base = None
        if meta_path.exists():
            base = json.loads(meta_path.read_text()).get("base_hf_id")
        base = base or get_base_model(self.cfg.train.base_model).hf_id
        return WhisperTranscriber(self.cfg.run_dir / "checkpoint", base,
                                  self.cfg.run.language, device=self.cfg.train.device)

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
