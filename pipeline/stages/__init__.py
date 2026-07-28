"""Ordered stage registry. `build_stages` returns the spine in execution order; the CLI
addresses any one by name or runs 'all'."""

from __future__ import annotations

from collections import OrderedDict

from pipeline.config import PipelineConfig
from pipeline.stages.align import AlignStage
from pipeline.stages.base import Stage
from pipeline.stages.clean import CleanStage
from pipeline.stages.evaluate import EvaluateStage
from pipeline.stages.ingest import IngestStage
from pipeline.stages.package import PackageStage
from pipeline.stages.train import TrainStage

_ORDER = [IngestStage, CleanStage, AlignStage, TrainStage, EvaluateStage, PackageStage]


def build_stages(cfg: PipelineConfig) -> "OrderedDict[str, Stage]":
    stages: "OrderedDict[str, Stage]" = OrderedDict()
    for cls in _ORDER:
        stages[cls.name] = cls(cfg)
    return stages
