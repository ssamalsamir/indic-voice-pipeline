"""JSONL helpers — the pipeline's inter-stage exchange format.

Every stage reads/writes JSONL (one utterance per line). It streams, diffs cleanly in
git review, and is trivial to inspect by hand — which matters when you're debugging why
3% of utterances got filtered.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator


def write_jsonl(path: Path, rows: Iterable[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_json(path: Path, obj: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
