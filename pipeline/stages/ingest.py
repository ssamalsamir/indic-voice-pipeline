"""Stage 1 — Ingest.

Pull audio+text from a registered dataset OR a client drop directory, standardise to a
single sample rate / channel count / encoding, and catalogue every utterance into
`corpus.jsonl`. Output rows are the stable schema the rest of the pipeline consumes.

Row schema:
    {"id", "audio_path", "text", "duration_s", "sr", "speaker", "domain", "source"}
"""

from __future__ import annotations

from pathlib import Path

from pipeline.registry import get_dataset
from pipeline.stages.base import Stage
from pipeline.utils.io import write_jsonl


class IngestStage(Stage):
    name = "ingest"
    output = "corpus.jsonl"

    def run(self) -> Path:
        ds = get_dataset(self.cfg.data.dataset)
        self.log.info("dataset=%s licence=%s", ds.key, ds.licence)

        if self.cfg.data.local_dir:
            rows = list(self._from_local_drop(self.cfg.data.local_dir))
        else:
            rows = list(self._from_hf(ds.hf_id))

        cap = self.cfg.data.max_train_utts
        if cap:
            rows = rows[:cap]
            self.log.info("capped to %d utterances for fast iteration", cap)

        self.log.info("catalogued %d utterances", len(rows))
        return write_jsonl(self.out_path, rows)

    # -- sources ---------------------------------------------------------------

    def _from_hf(self, hf_id: str | None):
        """Stream the train split via the shared HF source, materialising clips to real
        .wav files so downstream training has actual paths."""
        if not hf_id:
            raise ValueError("data.hf_path/registry hf_id missing and no local_dir given")
        from pipeline.sources import stream_hf_corpus  # noqa: PLC0415

        yield from stream_hf_corpus(
            hf_id=hf_id,
            hf_config=self.cfg.data.hf_config or self.cfg.run.language,
            split=self.cfg.data.train_split,
            target_sr=self.cfg.ingest.target_sr,
            audio_dir=self.cfg.run_dir / "audio",
            language=self.cfg.run.language,
            cap=self.cfg.data.max_train_utts,
        )

    def _from_local_drop(self, root: Path):
        """Client drop = a dir of audio files + a sidecar `transcripts.tsv`
        (`filename<TAB>text`). Deliberately dead simple so a client can produce it."""
        root = Path(root)
        tsv = root / "transcripts.tsv"
        if not tsv.exists():
            raise FileNotFoundError(f"expected {tsv} (filename<TAB>text per line)")
        for i, line in enumerate(tsv.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            fname, _, text = line.partition("\t")
            yield {
                "id": f"{self.cfg.run.language}_local_{i:07d}",
                "audio_path": str(root / fname),
                "audio_array": None,
                "text": text,
                "duration_s": None,  # measured in clean stage
                "sr": None,
                "speaker": None,
                "domain": "client_drop",
                "source": str(root),
            }
