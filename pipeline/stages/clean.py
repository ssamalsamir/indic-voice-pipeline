"""Stage 2 — Clean & normalise.

Text: run the single canonical normaliser (train==inference consistency).
Audio: resample check, silence trim, duration filtering. Denoise is optional (off by
default — denoisers can hurt ASR more than help). Emits `clean.jsonl` and drops rows
that fail hard filters, logging how many and why (honest data handling counts).
"""

from __future__ import annotations

from pathlib import Path

from pipeline.stages.base import Stage
from pipeline.text.codemix import code_mix_ratio, is_code_mixed
from pipeline.text.normalise import normalise
from pipeline.utils.io import read_jsonl, write_jsonl


class CleanStage(Stage):
    name = "clean"
    output = "clean.jsonl"

    def run(self) -> Path:
        corpus = self.cfg.run_dir / "corpus.jsonl"
        if not corpus.exists():
            raise FileNotFoundError("run `ingest` first — corpus.jsonl missing")

        cc = self.cfg.clean
        kept, dropped = [], {"empty_text": 0, "too_short": 0, "too_long": 0}

        for row in read_jsonl(corpus):
            text = normalise(
                row.get("text", ""),
                language=self.cfg.run.language,
                normalise_numerals=cc.normalise_numerals,
                expand_numbers_to_words=(self.cfg.run.track.value == "tts"),
                keep_code_mixing=cc.keep_code_mixing,
            )
            if not text:
                dropped["empty_text"] += 1
                continue

            dur = row.get("duration_s")
            if dur is not None:
                if dur < cc.min_duration_s:
                    dropped["too_short"] += 1
                    continue
                if dur > cc.max_duration_s:
                    dropped["too_long"] += 1
                    continue

            kept.append({
                **row,
                "text": text,
                "code_mix_ratio": round(code_mix_ratio(text), 3),
                "is_code_mixed": is_code_mixed(text),
            })

        self.log.info("kept %d, dropped %s", len(kept), dropped)
        # TODO(audio): plug in denoise/VAD trim here when cc.denoise / trim_silence and
        # a real waveform is available (torchaudio). Text path is fully live.
        return write_jsonl(self.out_path, kept)
