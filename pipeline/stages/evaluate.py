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
from pipeline.text.normalise import normalise_for_scoring
from pipeline.utils.io import dump_json, read_jsonl, write_jsonl


class EvaluateStage(Stage):
    name = "evaluate"
    output = "metrics.json"

    def run(self) -> Path:
        ckpt = self.cfg.run_dir / "checkpoint"
        if not ckpt.exists():
            # STT has nothing to score without a fine-tune. TTS does: the released voice
            # is a legitimate baseline, and measuring it is how you learn whether
            # fine-tuning is even needed. The report carries voice_is_finetuned=false so
            # the number is never mistaken for a tuned result.
            if self.cfg.run.track.value == "stt":
                raise FileNotFoundError("run `train` first — checkpoint missing")
            self.log.warning("no checkpoint — scoring the BASE voice as a baseline")

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
        if out.exists() and self._refs_match_current_norm(out):
            return out
        from pipeline.sources import stream_hf_corpus  # noqa: PLC0415

        # If eval reuses the train split, take a DISJOINT slice after the train clips so
        # the held-out set never overlaps training (honest WER) with one cached download.
        same = self.cfg.data.eval_split == self.cfg.data.train_split
        offset = (self.cfg.data.max_train_utts or 0) if same else 0
        eval_cap = self.cfg.data.max_eval_utts  # None = score the whole split
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

    def _refs_match_current_norm(self, manifest) -> bool:
        """Is the cached manifest's normalisation the one we score hypotheses with?

        Refs are normalised when this file is WRITTEN, hyps every time we score. Change
        the normaliser and a cached manifest silently pairs old refs against new hyps —
        stripped hyps vs punctuated refs, which inflates WER and looks like the model
        got worse. Normalised text is idempotent under its own normaliser, so refs that
        change when re-normalised prove the cache is stale.
        """
        for row in read_jsonl(manifest):
            if self._norm(row["text"]) != row["text"]:
                self.log.info("eval manifest was built with a different normaliser — "
                              "rebuilding so refs and hyps are scored the same way")
                return False
            return True   # first row is enough; they all took the same path
        return False      # empty file: rebuild

    def _norm(self, text: str) -> str:
        # Scoring pass, not the corpus pass: strips punctuation and folds nukta /
        # anusvara spelling variants. Applied to BOTH sides (refs here, hyps in
        # _collect_stt_pairs), so it can only remove differences, never invent
        # agreement. The corpus itself keeps its punctuation — see normalise.py.
        cc = self.cfg.clean
        return normalise_for_scoring(
            text, language=self.cfg.run.language,
            normalise_numerals=cc.normalise_numerals,
            expand_numbers_to_words=False,  # STT refs/hyps stay as digits
            keep_code_mixing=cc.keep_code_mixing,
        )

    def _transcribe(self, audio_path: str) -> str:
        # Beam search multiplies decoder activation memory by num_beams. On a CUDA box
        # that buys 1-2 WER points for a little time; on a 16GB Mac running a large
        # model it is what pushes the process into swap, where it stalls forever and
        # returns no number at all. A greedy result beats a beam result you never get.
        beams = 1 if self.cfg.train.device == "mps" else 5
        return self._transcriber().transcribe(
            audio_path, self.cfg.ingest.target_sr, num_beams=beams
        )

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
        """Intelligibility: synthesise held-out text, transcribe it, score the round trip.

        A TTS model that sounds fine but garbles words fails the only property that can
        be measured without a listening panel. The ASR judge is named in the config
        rather than inferred, because an intelligibility number means nothing unless you
        know which ears produced it — a weak judge flatters the voice.

        The judge's OWN error rate is the noise floor: our Hindi STT sits at ~0.07 WER
        on real speech, so an asr_wer near that is the ceiling, not a defect.
        """
        from pipeline.registry import get_base_model  # noqa: PLC0415
        from pipeline.tts import VitsSynthesiser  # noqa: PLC0415

        spec = get_base_model(self.cfg.train.base_model)
        ckpt = self.cfg.run_dir / "checkpoint"
        synth = VitsSynthesiser(ckpt if ckpt.exists() else None, spec.hf_id,
                                self.cfg.train.device)
        judge = self._asr_judge()
        out_dir = self.cfg.run_dir / "tts_audio"

        pairs, rows = [], []
        for i, row in enumerate(read_jsonl(self._ensure_eval_manifest())):
            ref = row["text"]
            if not ref.strip():
                continue
            wav, dur = synth.to_wav(ref, out_dir / f"tts_{i:05d}.wav")
            hyp = self._norm(judge.transcribe(str(wav), synth.sampling_rate, num_beams=1))
            pairs.append((self._norm(ref), hyp))
            rows.append({"id": row.get("id", i), "text": ref, "asr": hyp,
                         "wav": str(wav), "duration_s": round(dur, 3)})

        if not pairs:
            raise ValueError("no usable held-out text to synthesise")

        dump_json(self.cfg.run_dir / "tts_intelligibility.json", {"rows": rows})
        report = {
            "track": "tts", "n": len(pairs),
            "asr_wer": round(M.corpus_wer(pairs), 4),
            "asr_cer": round(M.corpus_cer(pairs), 4),
            "voice_is_finetuned": synth.is_finetuned,
            "asr_judge": self.cfg.eval.asr_judge_run or "registry base (untuned)",
            "by_slice": {},
        }
        # MCD needs reference audio AND a spectral lib; say so rather than emit a number
        # that was never computed — a fabricated metric is worse than a declared gap.
        report["mcd"] = None
        report["mcd_note"] = ("not computed: requires reference audio aligned to the "
                              "synthesised text plus librosa; see PLAN.md Wk5-6")
        return report

    def _asr_judge(self):
        """The STT model that scores intelligibility. Config-named, never guessed."""
        from pipeline.infer import WhisperTranscriber  # noqa: PLC0415
        from pipeline.registry import get_base_model  # noqa: PLC0415

        base = self.cfg.eval.asr_judge_hf_id or get_base_model("whisper-hindi-large-v2").hf_id
        run = self.cfg.eval.asr_judge_run
        ckpt = Path("runs") / run / "checkpoint" if run else None
        if ckpt is not None and not ckpt.exists():
            raise FileNotFoundError(
                f"eval.asr_judge_run='{run}' but {ckpt} is missing — train it first or "
                f"clear the field to judge with the untuned base model"
            )
        self.log.info("intelligibility judge: %s (%s)", base, run or "untuned base")
        return WhisperTranscriber(ckpt, base, self.cfg.run.language, self.cfg.train.device)

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
