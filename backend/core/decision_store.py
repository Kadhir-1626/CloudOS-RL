"""
Decision Store — in-memory ring buffer for recent scheduling decisions.
Thread-safe. Holds the last max_size decisions.
"""

import threading
from collections import deque
from typing import Deque, List, Optional


class DecisionStore:

    def __init__(self, max_size: int = 1000):
        self._store:  Deque = deque(maxlen=max_size)
        self._index:  dict  = {}          # decision_id → decision
        self._lock    = threading.Lock()
        self._total   = 0

    def put(self, decision) -> None:
        with self._lock:
            self._store.append(decision)
            self._index[decision.decision_id] = decision
            self._total += 1

    def get(self, decision_id: str) -> Optional[object]:
        with self._lock:
            return self._index.get(decision_id)

    def list(
        self,
        limit:  int = 20,
        cloud:  Optional[str] = None,
        region: Optional[str] = None,
    ) -> List:
        with self._lock:
            items = list(self._store)
        items.reverse()  # newest first
        if cloud:
            items = [d for d in items if d.cloud == cloud]
        if region:
            items = [d for d in items if d.region == region]
        return items[:limit]

    def last(self) -> Optional[object]:
        with self._lock:
            return self._store[-1] if self._store else None

    def total_count(self) -> int:
        with self._lock:
            return self._total