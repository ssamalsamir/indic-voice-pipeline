"""Stage 3 — Align & segment.

Forced alignment turns long/loosely-labelled audio into utterance-level segments and,
crucially, gives an alignment confidence we can threshold on. Bad alignments are the
top source of silent quality loss in both tracks, so we drop below `align.min_score`
and log the yield. Emits the final `manifest.jsonl` that the train stage consumes.

Aligner is pluggable (whisperx | ctc | mfa | none) via config. `none` passes utterances
through unchanged — correct when the source is already utterance-level (Kathbath, FLEURS).
"""

from __future__ import annotations

from pathlib import Path

from pipeline.stages.base import Stage
from pipeline.utils.io import read_jsonl, write_jsonl


class AlignStage(Stage):
    name = "align"
    output = "manifest.jsonl"

    def run(self) -> Path:
        clean = self.cfg.run_dir / "clean.jsonl"
        if not clean.exists():
            raise FileNotFoundError("run `clean` first — clean.jsonl missing")

        aligner = self.cfg.align.aligner
        rows = list(read_jsonl(clean))

        if aligner == "none":
            self.log.info("aligner=none — passing %d utterances through", len(rows))
            manifest = [self._to_manifest(r, score=1.0) for r in rows]
        else:
            manifest = list(self._align(rows, aligner))

        write_jsonl(self.out_path, manifest)
        self.log.info("manifest: %d segments (from %d utterances)", len(manifest), len(rows))
        return self.out_path

    def _align(self, rows, aligner: str):
        """Real forced alignment. Deferred import; falls back to pass-through with a
        loud warning if the aligner backend isn't installed, so the spine still runs."""
        try:
            backend = _load_aligner(aligner)
        except ImportError as e:  # keep the spine runnable
            self.log.warning("aligner %s unavailable (%s) — passing through", aligner, e)
            for r in rows:
                yield self._to_manifest(r, score=1.0)
            return

        min_score = self.cfg.align.min_score
        for r in rows:
            for seg in backend.align(r["audio_path"], r["text"]):
                if seg["score"] < min_score:
                    continue
                yield self._to_manifest(r, score=seg["score"],
                                        start=seg["start"], end=seg["end"],
                                        text=seg.get("text", r["text"]))

    @staticmethod
    def _to_manifest(row, *, score, start=0.0, end=None, text=None):
        return {
            "id": row["id"],
            "audio_path": row["audio_path"],
            "text": text if text is not None else row["text"],
            "start": start,
            "end": end if end is not None else row.get("duration_s"),
            "align_score": score,
            "speaker": row.get("speaker"),
            "domain": row.get("domain", "general"),
            "is_code_mixed": row.get("is_code_mixed", False),
        }


def _load_aligner(name: str):
    # Wire real backends here (whisperx / ctc-segmentation / MFA). Raises ImportError
    # until installed, which the caller handles gracefully.
    raise ImportError(f"aligner backend '{name}' not wired yet")
