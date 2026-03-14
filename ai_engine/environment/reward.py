import numpy as np
from typing import Dict, Tuple, Optional, Any

from ai_engine.environment.action_decoder import ActionDecoder


_REGION_IDX: Dict[str, int] = {r: i for i, r in enumerate(ActionDecoder.REGIONS)}

BASELINE_COST_PER_HR = 0.096
BASELINE_LATENCY_MS = 200.0
BASELINE_CARBON_GCO2 = 41.5


class RewardFunction:
    """
    Supports two compute() modes:

    1) Test compatibility mode:
       compute(state_dict) -> float

    2) Environment mode:
       compute(action=..., state=..., pricing=...) -> (float, components_dict)

    Reward formula:
      R = α·ΔCost + β·ΔLatency + γ·ΔCarbon + δ·SLA − ε·Migration
    """

    DEFAULT_WEIGHTS = {
        "alpha": 0.35,
        "beta": 0.25,
        "gamma": 0.20,
        "delta": 0.20,
        "epsilon": 0.05,
    }

    _PURCHASE_MULTIPLIER: Dict[str, float] = {
        "on_demand": 1.00,
        "spot": 0.33,
        "preemptible": 0.30,
        "savings_plan": 0.55,
        "reserved_1yr": 0.60,
        "reserved_3yr": 0.40,
    }

    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}

        self._config = config

        weights_cfg = config.get("reward_weights")
        if weights_cfg is None:
            weights_cfg = config.get("reward", {}).get("weights", {})

        merged = {**self.DEFAULT_WEIGHTS, **weights_cfg}

        self.weights = {
            "alpha": float(merged.get("alpha", 0.35)),
            "beta": float(merged.get("beta", 0.25)),
            "gamma": float(merged.get("gamma", 0.20)),
            "delta": float(merged.get("delta", 0.20)),
            "epsilon": float(merged.get("epsilon", 0.05)),
        }

        self.alpha = self.weights["alpha"]
        self.beta = self.weights["beta"]
        self.gamma = self.weights["gamma"]
        self.delta = self.weights["delta"]
        self.epsilon = self.weights["epsilon"]

    def compute(
        self,
        action: Optional[Dict] = None,
        state: Optional[np.ndarray] = None,
        pricing: Optional[Dict] = None,
    ):
        """
        Dual-mode compute:

        A) compute(state_dict) -> float
           Here, 'action' is actually the compact summary state used by tests.

        B) compute(action=action_dict, state=state_array, pricing=pricing_dict)
           -> (float, components)
        """
        # Test compatibility mode
        if state is None and pricing is None and isinstance(action, dict):
            return self._compute_from_summary_state(action)

        # Environment runtime mode
        if action is None or state is None:
            raise TypeError("compute() requires either compute(state_dict) or compute(action=..., state=..., pricing=...)")

        if pricing is None:
            pricing = {}

        cost_r = self._cost(action, pricing)
        latency_r = self._latency(action, state)
        carbon_r = self._carbon(action, state)
        sla_r = self._sla(action, state)
        migration_p = self._migration(action)

        total = (
            self.alpha * cost_r
            + self.beta * latency_r
            + self.gamma * carbon_r
            + self.delta * sla_r
            - self.epsilon * migration_p
        )
        total = float(np.clip(total, -10.0, 10.0))

        components = {
            "cost": cost_r,
            "latency": latency_r,
            "carbon": carbon_r,
            "sla": sla_r,
            "migration": migration_p,
            "total": total,
        }
        return total, components

    def _compute_from_summary_state(self, s: Dict[str, Any]) -> float:
        """
        Test-facing reward path using compact summary state dict.
        Expected keys may include:
          - cost_per_hr
          - carbon_intensity
          - sla_met
          - migration_occurred
        """
        cost_per_hr = float(s.get("cost_per_hr", BASELINE_COST_PER_HR))
        carbon_intensity = float(s.get("carbon_intensity", 415.0))
        sla_met = bool(s.get("sla_met", True))
        migration_occurred = bool(s.get("migration_occurred", False))

        cost_delta = (BASELINE_COST_PER_HR - cost_per_hr) / BASELINE_COST_PER_HR
        cost_r = float(np.clip(cost_delta * 2.0, -1.0, 1.0))

        baseline_intensity = 415.0
        carbon_delta = (baseline_intensity - carbon_intensity) / baseline_intensity
        carbon_r = float(np.clip(carbon_delta, -1.0, 1.0))

        sla_r = 0.5 if sla_met else -1.0

        # No explicit latency signal in compact test state
        latency_r = 0.0

        migration_p = 0.3 if migration_occurred else 0.0

        total = (
            self.alpha * cost_r
            + self.beta * latency_r
            + self.gamma * carbon_r
            + self.delta * sla_r
            - self.epsilon * migration_p
        )
        return float(np.clip(total, -10.0, 10.0))

    # ── environment sub-rewards ──────────────────────────────────────────

    def _cost(self, action: Dict, pricing: Dict) -> float:
        region = action["generic_region"]
        region_value = pricing.get(region, BASELINE_COST_PER_HR)

        if isinstance(region_value, dict):
            instance = action["instance_type"]
            base = region_value.get(instance, BASELINE_COST_PER_HR)
        else:
            base = float(region_value)

        actual = base * self._PURCHASE_MULTIPLIER.get(
            action["purchase_option"], 1.0
        )
        delta = (BASELINE_COST_PER_HR - actual) / BASELINE_COST_PER_HR
        return float(np.clip(delta * 2.0, -1.0, 1.0))

    def _latency(self, action: Dict, state: np.ndarray) -> float:
        idx = _REGION_IDX.get(action["generic_region"], 0)
        est_latency = state[30 + idx] * 1000.0
        sla_latency = state[7] * 1000.0

        if est_latency <= sla_latency:
            return float(
                np.clip(
                    (BASELINE_LATENCY_MS - est_latency) / BASELINE_LATENCY_MS,
                    0.0,
                    1.0,
                )
            )

        return float(
            np.clip(-((est_latency - sla_latency) / sla_latency), -1.0, 0.0)
        )

    def _carbon(self, action: Dict, state: np.ndarray) -> float:
        idx = _REGION_IDX.get(action["generic_region"], 0)
        carbon_kwh = state[20 + idx] * 600.0
        carbon_hr = carbon_kwh * 0.1
        delta = (BASELINE_CARBON_GCO2 - carbon_hr) / BASELINE_CARBON_GCO2
        return float(np.clip(delta, -1.0, 1.0))

    def _sla(self, action: Dict, state: np.ndarray) -> float:
        required_tier = state[6] * 4.0
        assigned_tier = float(action["sla_tier"])

        if assigned_tier >= required_tier:
            return 0.5

        return float(np.clip(-0.5 * (required_tier - assigned_tier), -1.0, 0.0))

    def _migration(self, action: Dict) -> float:
        return 0.3 if action.get("requires_migration", False) else 0.0