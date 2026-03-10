import json
import logging
from pathlib import Path
from typing import List, Dict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class MetricsCallback(BaseCallback):
    """Logs per-objective reward breakdown to JSON for post-analysis."""

    def __init__(self, log_dir: str, flush_every: int = 5_000, verbose: int = 0):
        super().__init__(verbose)
        self._dir        = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._flush_every = flush_every
        self._buf: List[Dict] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            rc = info.get("reward_components")
            if rc:
                self._buf.append({"t": self.num_timesteps, **rc})

        if self.n_calls % self._flush_every == 0:
            self._flush()
            for key in ("cost", "latency", "carbon", "sla"):
                vals = [e[key] for e in self._buf[-1000:] if key in e]
                if vals:
                    self.logger.record(f"objective/{key}", float(np.mean(vals)))

        return True

    def _on_training_end(self):
        self._flush()
        logger.info("MetricsCallback: saved %d records to %s", len(self._buf), self._dir)

    def _flush(self):
        if not self._buf:
            return
        out = self._dir / "reward_breakdown.jsonl"
        with open(out, "a") as fh:
            for entry in self._buf:
                fh.write(json.dumps(entry) + "\n")
        self._buf.clear()