"""Stage 5 — Evaluate.

Score the fine-tuned model on a fixed held-out set, automatically, every run. Computes
the configured metrics, breaks them down by the configured slices (domain, noise,
code-mixed vs not), and checks each against the mentor-agreed threshold so the report
says PASS/FAIL, not just a number. Emits `metrics.json`.

STT: WER + CER (+ per-slice).
TTS: intelligibility = synthesise held-out text -> transcribe with an ASR model ->
     WER against the reference (objective clarity proxy); MCD; speaker similarity;
     MOS is collected out-of-band from a small panel and merged in.
"""

from __future__ import annotations

from pathlib import Path

from pipeline import metrics as M
from pipeline.stages.base import Stage
from pipeline.text.normalise import normalise
from pipeline.utils.io import dump_json, read_jsonl, write_jsonl


class EvaluateStage(Stage):
    name = "evaluate"
    output = "metrics.json"

    def run(self) -> Path:
        ckpt = self.cfg.run_dir / "checkpoint"
        if not ckpt.exists():
            raise FileNotFoundError("run `train` first — checkpoint missing")

        if self.cfg.run.track.value == "stt":
            report = self._eval_stt()
        else:
            report = self._eval_tts()

        report["thresholds"] = self.cfg.eval.thresholds
        report["verdict"] = self._verdict(report)
        self.log.info("verdict: %s", report["verdict"])
        return dump_json(self.out_path, report)

    # -- STT -------------------------------------------------------------------

    def _eval_stt(self) -> dict:
        """Transcribe the held-out set, compute corpus WER/CER + per-slice.

        `_transcribe` is the only model-touching call; wire it to mlx-whisper /
        transformers loading the adapter from the checkpoint. Until wired it raises,
        and the metric maths below is already unit-tested via pipeline.metrics.
        """
        pairs, slices = self._collect_stt_pairs()
        report = {
            "track": "stt",
            "n": len(pairs),
            "wer": round(M.corpus_wer(pairs), 4),
            "cer": round(M.corpus_cer(pairs), 4),
            "by_slice": {},
        }
        for slice_name, sub in slices.items():
            report["by_slice"][slice_name] = {
                key: {"wer": round(M.corpus_wer(p), 4),
                      "cer": round(M.corpus_cer(p), 4), "n": len(p)}
                for key, p in sub.items()
            }
        return report

    def _collect_stt_pairs(self):
        """Score on a HELD-OUT split (never the train manifest). References are already
        normalised in the eval manifest; hypotheses are normalised the SAME way here so
        the WER/CER comparison is fair (Whisper emits casing/punct the refs don't have)."""
        manifest = self._ensure_eval_manifest()
        pairs: list[tuple[str, str]] = []
        slices: dict[str, dict[str, list]] = {s: {} for s in self.cfg.eval.slices}
        for row in read_jsonl(manifest):
            ref = row["text"]
            hyp = self._norm(self._transcribe(row["audio_path"]))
            pairs.append((ref, hyp))
            for s in self.cfg.eval.slices:
                key = str(row.get(s, "na")) if s != "noise" else _noise_bucket(row)
                slices[s].setdefault(key, []).append((ref, hyp))
        return pairs, slices

    def _ensure_eval_manifest(self):
        """Stream the held-out split once, normalise references, cache to disk."""
        out = self.cfg.run_dir / "eval_manifest.jsonl"
        if out.exists():
            return out
        from pipeline.sources import stream_hf_corpus  # noqa: PLC0415

        # If eval reuses the train split, take a DISJOINT slice after the train clips so
        # the held-out set never overlaps training (honest WER) with one cached download.
        same = self.cfg.data.eval_split == self.cfg.data.train_split
        offset = (self.cfg.data.max_train_utts or 0) if same else 0
        eval_cap = 50  # keep eval fast on MPS (each clip is a full transcription)
        rows = stream_hf_corpus(
            hf_id=self.cfg.data.hf_path,
            hf_config=self.cfg.data.hf_config or self.cfg.run.language,
            split=self.cfg.data.eval_split,
            target_sr=self.cfg.ingest.target_sr,
            audio_dir=self.cfg.run_dir / "eval_audio",
            language=self.cfg.run.language,
            cap=eval_cap,
            offset=offset,
        )
        return write_jsonl(out, ({**r, "text": self._norm(r["text"])} for r in rows))

    def _norm(self, text: str) -> str:
        cc = self.cfg.clean
        return normalise(
            text, language=self.cfg.run.language,
            normalise_numerals=cc.normalise_numerals,
            expand_numbers_to_words=False,  # STT refs/hyps stay as digits
            keep_code_mixing=cc.keep_code_mixing,
        )

    def _transcribe(self, audio_path: str) -> str:
        return self._transcriber().transcribe(audio_path, self.cfg.ingest.target_sr)

    def _transcriber(self):
        """Lazily build + cache the transcriber (loads the adapter once per eval run)."""
        if getattr(self, "_cached_transcriber", None) is None:
            from pipeline.infer import WhisperTranscriber  # noqa: PLC0415
            from pipeline.registry import get_base_model  # noqa: PLC0415
            spec = get_base_model(self.cfg.train.base_model)
            self._cached_transcriber = WhisperTranscriber(
                self.cfg.run_dir / "checkpoint", spec.hf_id,
                self.cfg.run.language, self.cfg.train.device,
            )
        return self._cached_transcriber

    # -- TTS -------------------------------------------------------------------

    def _eval_tts(self) -> dict:
        raise NotImplementedError(
            "TTS eval: synthesise held-out text, run ASR-WER intelligibility + MCD, "
            "merge MOS panel CSV. Metric maths reuses pipeline.metrics."
        )

    # -- verdict ---------------------------------------------------------------

    def _verdict(self, report: dict) -> str:
        th = self.cfg.eval.thresholds
        if not th:
            return "UNSCORED (no mentor thresholds set)"
        for metric, bar in th.items():
            val = report.get(metric)
            if val is None:
                continue
            # error metrics (wer/cer/mcd) must be <= bar; quality metrics (mos) >=
            ok = val <= bar if metric in ("wer", "cer", "mcd", "asr_wer") else val >= bar
            if not ok:
                return f"FAIL ({metric}={val} vs bar {bar})"
        return "PASS"


def _noise_bucket(row: dict) -> str:
    snr = row.get("snr_db")
    if snr is None:
        return "unknown"
    return "clean" if snr >= 20 else "noisy"
