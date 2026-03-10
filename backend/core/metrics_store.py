import time
from collections import deque
from threading import Lock
from typing import Any, Dict, List

import numpy as np


class MetricsStore:
    """Thread-safe rolling window store for scheduling decisions and metrics."""

    _MAX = 10_000

    def __init__(self):
        self._decisions: deque = deque(maxlen=self._MAX)
        self._lock = Lock()

    async def record_decision(self, d: Dict):
        with self._lock:
            self._decisions.append({**d, "_ts": time.time()})

    def aggregate(self, window_hours: float = 1.0) -> Dict[str, Any]:
        cutoff = time.time() - window_hours * 3600
        with self._lock:
            recent = [d for d in self._decisions if d.get("_ts", 0) >= cutoff]

        n = len(recent)
        if n == 0:
            return {"total_decisions": 0}

        costs    = [d.get("cost_savings_pct", 0.0)   for d in recent]
        carbons  = [d.get("carbon_savings_pct", 0.0) for d in recent]
        latencies = [d.get("latency_ms", 0.0)        for d in recent]

        return {
            "total_decisions":       n,
            "avg_cost_savings_pct":  float(np.mean(costs)),
            "avg_carbon_savings_pct":float(np.mean(carbons)),
            "avg_latency_ms":        float(np.mean(latencies)),
            "p95_latency_ms":        float(np.percentile(latencies, 95)),
            "decisions_per_hour":    n / max(window_hours, 1e-6),
            "cloud_dist":            self._count(recent, "cloud"),
            "purchase_dist":         self._count(recent, "purchase_option"),
        }

    def recent(self, n: int = 50) -> List[Dict]:
        with self._lock:
            return list(self._decisions)[-n:]

    @staticmethod
    def _count(items: List[Dict], key: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for item in items:
            v = item.get(key, "unknown")
            out[v] = out.get(v, 0) + 1
        return out