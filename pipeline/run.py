"""Entry point: run one stage, a range, or the whole pipeline for a config.

    python -m pipeline.run all      --config configs/hi_stt_kathbath.yaml
    python -m pipeline.run ingest   --config configs/hi_stt_kathbath.yaml
    python -m pipeline.run evaluate --config configs/hi_stt_kathbath.yaml --force

Because every stage is addressable on its own, you iterate on `clean` without
re-running `train`, and rerun `evaluate` a dozen times cheaply.
"""

from __future__ import annotations

import argparse
import sys

from pipeline.config import PipelineConfig, Track
from pipeline.stages import build_stages
from pipeline.utils.logging import get_logger

log = get_logger("run")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipeline.run")
    parser.add_argument(
        "stage",
        help="stage name (ingest|clean|align|train|evaluate|package) or 'all'.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true", help="ignore cached outputs")
    args = parser.parse_args(argv)

    cfg = PipelineConfig.load(args.config)
    log.info("run=%s track=%s lang=%s", cfg.run.name, cfg.run.track.value, cfg.run.language)

    stages = build_stages(cfg)  # ordered dict name -> Stage
    if args.stage == "all":
        for stage in stages.values():
            stage(force=args.force)
        return 0

    if args.stage not in stages:
        log.error("unknown stage %r; known: %s", args.stage, list(stages))
        return 2
    stages[args.stage](force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
