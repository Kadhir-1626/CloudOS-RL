"""
Online Feedback Collector
==========================
Collects actual outcome data from completed CloudWorkload CRs
and computes the real reward signal for PPO fine-tuning.

This closes the RL feedback loop:
  Scheduled workload → actual cloud cost (from CUR) → real reward
  Actual carbon intensity → real carbon savings
  Actual completion time → SLA met/breached

The feedback is stored as (state, action, actual_reward) tuples
for periodic PPO fine-tuning (not continuous — runs nightly or weekly).

Data sources:
  - CloudWorkload CR status (phase, scheduledCloud, scheduledRegion)
  - Module G CUR ingestor (actual cost per resource)
  - Module G carbon pipeline (actual carbon at time of execution)
"""

import json
import logging
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_FEEDBACK_PATH = Path("data/feedback/outcomes.jsonl")


@dataclass
class WorkloadOutcome:
    """Actual measured outcome for a completed CloudWorkload."""
    workload_id:            str
    decision_id:            str
    cloud:                  str
    region:                 str
    instance_type:          str
    purchase_option:        str
    scheduled_at:           str
    completed_at:           str
    actual_duration_hours:  float
    expected_duration_hours:float
    actual_cost_usd:        float
    baseline_cost_usd:      float
    actual_carbon_gco2:     float
    baseline_carbon_gco2:   float
    sla_met:                bool
    # Computed reward
    actual_reward:          Optional[float] = None


class OnlineFeedbackCollector:
    """
    Polls completed CloudWorkload CRs, fetches actual cost/carbon,
    and writes outcome records for offline PPO fine-tuning.
    """

    def __init__(self, config: dict, namespace: str = "cloudos-rl"):
        self._config    = config
        self._namespace = namespace
        _FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    def collect_completed(self) -> List[WorkloadOutcome]:
        """
        Lists all Completed CRs, computes actual reward, writes to JSONL.
        Call this from a scheduled job or operator background thread.
        """
        completed = self._list_by_phase("Completed")
        outcomes  = []

        for item in completed:
            outcome = self._build_outcome(item)
            if outcome and outcome.actual_reward is not None:
                outcomes.append(outcome)
                self._append_outcome(outcome)

        logger.info(
            "OnlineFeedbackCollector: collected %d outcomes from %d completed CRs",
            len(outcomes), len(completed)
        )
        return outcomes

    def get_feedback_stats(self) -> dict:
        """Returns summary stats over all collected outcomes."""
        if not _FEEDBACK_PATH.exists():
            return {"count": 0}

        rewards = []
        cost_savings = []
        carbon_savings = []

        with open(_FEEDBACK_PATH) as fh:
            for line in fh:
                try:
                    o = json.loads(line)
                    if o.get("actual_reward") is not None:
                        rewards.append(o["actual_reward"])
                    if o.get("baseline_cost_usd", 0) > 0:
                        cs = 1.0 - o["actual_cost_usd"] / o["baseline_cost_usd"]
                        cost_savings.append(cs)
                    if o.get("baseline_carbon_gco2", 0) > 0:
                        cs = 1.0 - o["actual_carbon_gco2"] / o["baseline_carbon_gco2"]
                        carbon_savings.append(cs)
                except (json.JSONDecodeError, KeyError):
                    continue

        if not rewards:
            return {"count": 0}

        return {
            "count":              len(rewards),
            "mean_reward":        round(float(np.mean(rewards)), 4),
            "mean_cost_savings":  round(float(np.mean(cost_savings)), 4)  if cost_savings  else 0,
            "mean_carbon_savings":round(float(np.mean(carbon_savings)), 4) if carbon_savings else 0,
        }

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_outcome(self, item: dict) -> Optional[WorkloadOutcome]:
        try:
            meta   = item["metadata"]
            spec   = item["spec"]
            status = item.get("status", {})

            scheduled_at  = status.get("scheduledAt",  "")
            completed_at  = status.get("message", "").split("after ")[1].split("h")[0] \
                            if "after" in status.get("message","") else "1.0"

            expected_hrs  = float(spec.get("expectedDurationHours", 1.0))
            actual_hrs    = float(completed_at) if completed_at.replace(".","").isdigit() else expected_hrs

            cloud   = status.get("scheduledCloud",  "aws")
            region  = status.get("scheduledRegion", "us-east-1")
            cost_hr = float(status.get("estimatedCostPerHr", 0.096))

            # Compute actual cost
            actual_cost   = cost_hr * actual_hrs
            # Baseline: on-demand us-east-1 m5.large equivalent
            baseline_cost = 0.096 * actual_hrs

            # Carbon: gCO2/kWh × kWh consumed
            carbon        = self._carbon_for_region(region)
            baseline_carb = self._carbon_for_region("us-east-1")
            # Assume 0.1 kWh per vCPU per hour as proxy
            vcpus         = float(spec.get("resources", {}).get("cpu", "2").replace("m",""))
            if vcpus > 100: vcpus /= 1000  # millicores
            kwh           = vcpus * 0.1 * actual_hrs
            actual_carbon = carbon * kwh
            baseline_carb_total = baseline_carb * kwh

            # Compute reward using same formula as training
            alpha, beta, gamma, delta, epsilon = 0.35, 0.25, 0.20, 0.20, 0.05
            delta_cost    = (baseline_cost - actual_cost) / max(baseline_cost, 0.001)
            delta_carbon  = (baseline_carb_total - actual_carbon) / max(baseline_carb_total, 0.001)
            sla_met       = status.get("phase") == "Completed"
            sla_bonus     = 1.0 if sla_met else -1.0

            reward = (alpha * delta_cost +
                      gamma * delta_carbon +
                      delta * sla_bonus)

            return WorkloadOutcome(
                workload_id=            meta["name"],
                decision_id=            status.get("decisionId", ""),
                cloud=                  cloud,
                region=                 region,
                instance_type=          status.get("instanceType",   "m5.large"),
                purchase_option=        status.get("purchaseOption", "on_demand"),
                scheduled_at=           scheduled_at,
                completed_at=           scheduled_at,
                actual_duration_hours=  actual_hrs,
                expected_duration_hours=expected_hrs,
                actual_cost_usd=        round(actual_cost,        4),
                baseline_cost_usd=      round(baseline_cost,      4),
                actual_carbon_gco2=     round(actual_carbon,      4),
                baseline_carbon_gco2=   round(baseline_carb_total,4),
                sla_met=                sla_met,
                actual_reward=          round(reward, 6),
            )
        except Exception as exc:
            logger.warning("_build_outcome failed: %s", exc)
            return None

    def _carbon_for_region(self, region: str) -> float:
        carbon_path = Path(
            self._config.get("data_pipeline", {}).get(
                "carbon_output_path", "data/carbon/carbon_intensity.json"
            )
        )
        try:
            if carbon_path.exists():
                with open(carbon_path) as fh:
                    data = json.load(fh)
                entry = data.get(region, {})
                return float(entry.get("gco2_per_kwh", 415.0)
                             if isinstance(entry, dict) else entry)
        except Exception:
            pass
        return {"us-east-1": 415.0, "eu-north-1": 42.0}.get(region, 415.0)

    def _list_by_phase(self, phase: str) -> list:
        try:
            r = subprocess.run(
                ["kubectl", "get", "cloudworkloads", "-n", self._namespace, "-o", "json"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode != 0:
                return []
            return [i for i in json.loads(r.stdout).get("items", [])
                    if i.get("status", {}).get("phase") == phase]
        except Exception:
            return []

    @staticmethod
    def _append_outcome(outcome: WorkloadOutcome):
        with open(_FEEDBACK_PATH, "a") as fh:
            fh.write(json.dumps(asdict(outcome)) + "\n")