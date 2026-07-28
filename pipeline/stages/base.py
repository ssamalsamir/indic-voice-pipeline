"""Stage contract.

Every stage takes the run config + the run directory, does one job, and writes a
declared artifact. Stages are idempotent-ish: rerunning overwrites its own output and
never touches another stage's. This is what lets any stage be swapped or rerun in
isolation (assignment §5).
"""

from __future__ import annotations

import abc
from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.utils.logging import get_logger


class Stage(abc.ABC):
    #: stable stage name, also used as the CLI subcommand
    name: str
    #: filename this stage produces inside run_dir (relative)
    output: str

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.log = get_logger(self.name)

    @property
    def out_path(self) -> Path:
        return self.cfg.run_dir / self.output

    def already_done(self) -> bool:
        return self.out_path.exists()

    @abc.abstractmethod
    def run(self) -> Path:
        """Do the work, write self.out_path, return it."""

    def __call__(self, force: bool = False) -> Path:
        self.cfg.run_dir.mkdir(parents=True, exist_ok=True)
        if self.already_done() and not force:
            self.log.info("skip (exists): %s", self.out_path)
            return self.out_path
        self.log.info("running %s", self.name)
        path = self.run()
        self.log.info("wrote %s", path)
        return path
