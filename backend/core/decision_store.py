"""
Decision Store — in-memory ring buffer for recent scheduling decisions.

Stores:
- SchedulingDecision
- optional originating workload
- optional state vector
- optional decoded action

Thread-safe.
Backward compatible with existing code that calls put(decision).
"""

import logging
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np


logger = logging.getLogger(__name__)
DEFAULT_STATE_SIZE = 45


class DecisionRecord:
    __slots__ = ("decision", "workload", "state", "decoded")

    def __init__(
        self,
        decision: Any,
        workload: Optional[Dict[str, Any]] = None,
        state: Optional[np.ndarray] = None,
        decoded: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.decision = decision
        self.workload = workload or {}
        self.state = (
            state
            if state is not None
            else np.zeros(DEFAULT_STATE_SIZE, dtype=np.float32)
        )
        self.decoded = decoded or {}


class DecisionStore:
    def __init__(self, max_size: int = 1000):
        self._max_size = max_size
        self._records: Deque[DecisionRecord] = deque(maxlen=max_size)
        self._index: Dict[str, DecisionRecord] = {}
        self._lock = threading.Lock()
        self._total = 0

    def put(
        self,
        decision: Any,
        workload: Optional[Dict[str, Any]] = None,
        state: Optional[np.ndarray] = None,
        decoded: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store a decision record.

        Backward compatible:
            put(decision)

        SHAP-compatible:
            put(decision, workload, state, decoded)
        """
        record = DecisionRecord(
            decision=decision,
            workload=workload,
            state=state,
            decoded=decoded,
        )

        with self._lock:
            # Remove stale index entry before deque evicts oldest record
            if len(self._records) == self._max_size and self._records:
                evicted = self._records[0]
                evicted_id = getattr(evicted.decision, "decision_id", None)
                if evicted_id is not None:
                    self._index.pop(evicted_id, None)

            self._records.append(record)
            self._index[decision.decision_id] = record
            self._total += 1

    def attach_explanation(self, decision_id: str, explanation: Dict[str, Any]) -> bool:
        """
        Attach explanation to a stored decision in-place.

        Returns:
            True  -> decision found and updated
            False -> decision not found or could not be updated

        Entire operation runs under a single lock to avoid races between:
        - reading the record
        - mutating/replacing the decision object
        - updating the index
        """
        with self._lock:
            record = self._index.get(decision_id)

            if record is None:
                logger.warning(
                    "DecisionStore.attach_explanation: %s not found in store "
                    "(may have been evicted). Current indexed=%d",
                    decision_id[:8],
                    len(self._index),
                )
                return False

            # Path 1: direct attribute set
            try:
                record.decision.explanation = explanation
                if getattr(record.decision, "explanation", None) == explanation:
                    return True
            except Exception:
                pass

            # Path 2: __dict__ mutation
            try:
                record.decision.__dict__["explanation"] = explanation
                return True
            except Exception:
                pass

            # Path 3: object.__setattr__ bypass
            try:
                object.__setattr__(record.decision, "explanation", explanation)
                return True
            except Exception:
                pass

            # Path 4: Pydantic v2 model_copy replacement
            try:
                model_copy = getattr(record.decision, "model_copy", None)
                if callable(model_copy):
                    new_decision = model_copy(update={"explanation": explanation})
                    record.decision = new_decision
                    self._index[decision_id] = record
                    return True
            except Exception as exc:
                logger.error(
                    "DecisionStore.attach_explanation: model_copy path failed for %s: %s",
                    decision_id[:8],
                    exc,
                )

            logger.error(
                "DecisionStore.attach_explanation: all update paths failed for %s",
                decision_id[:8],
            )
            return False

    def get_record(self, decision_id: str) -> Optional[DecisionRecord]:
        """
        Return full stored record including state/workload/decoded.
        """
        with self._lock:
            return self._index.get(decision_id)

    def get(self, decision_id: str) -> Optional[Any]:
        """
        Return only the SchedulingDecision object.
        Backward compatible with existing callers.
        """
        record = self.get_record(decision_id)
        return record.decision if record else None

    def list(
        self,
        limit: int = 20,
        cloud: Optional[str] = None,
        region: Optional[str] = None,
    ) -> List[Any]:
        """
        Return recent SchedulingDecision objects, newest first.
        """
        with self._lock:
            items = [record.decision for record in self._records]

        items.reverse()

        if cloud:
            items = [decision for decision in items if decision.cloud == cloud]
        if region:
            items = [decision for decision in items if decision.region == region]

        return items[:limit]

    def last(self) -> Optional[Any]:
        """
        Return most recent SchedulingDecision.
        """
        with self._lock:
            return self._records[-1].decision if self._records else None

    def total_count(self) -> int:
        with self._lock:
            return self._total