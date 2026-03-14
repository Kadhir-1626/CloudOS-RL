import numpy as np
from typing import Dict, List


REGIONS = [
    "us-east-1", "us-west-2", "eu-west-1", "eu-central-1",
    "ap-southeast-1", "ap-northeast-1", "us-central1",
    "europe-west4", "eastus", "westeurope",
]

_BASE_LATENCY_MS: Dict[str, float] = {
    "us-east-1": 45.0,
    "us-west-2": 72.0,
    "eu-west-1": 110.0,
    "eu-central-1": 115.0,
    "ap-southeast-1": 185.0,
    "ap-northeast-1": 168.0,
    "us-central1": 55.0,
    "europe-west4": 108.0,
    "eastus": 48.0,
    "westeurope": 112.0,
}


class StateBuilder:
    """
    Constructs a 45-dimensional observation vector.

    Layout:
      [0:10]  workload features        (10 dims)
      [10:20] per-region pricing       (10 dims)
      [20:30] per-region carbon gCO2   (10 dims)
      [30:40] per-region latency       (10 dims)
      [40:45] historical reward signal (5 dims)
    """

    def __init__(self, config: Dict):
        self._p_norm = config.get("pricing_normalization", 10.0)
        self._c_norm = config.get("carbon_normalization", 600.0)
        self._l_norm = config.get("latency_normalization", 1000.0)

    @staticmethod
    def _extract_price(pricing: Dict, region: str, default: float = 0.096) -> float:
        """
        Extracts on-demand price from pricing dict.

        Handles both:
          pricing[region] = 0.096
          pricing[region] = {"on_demand_per_vcpu_hr": 0.096}
        """
        value = pricing.get(region, default)

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, dict):
            return float(
                value.get("on_demand_per_vcpu_hr")
                or value.get("on_demand")
                or value.get("m5.large")
                or default
            )

        return float(default)

    def build(
        self,
        workload: Dict,
        pricing: Dict,
        carbon: Dict,
        history: List[Dict],
    ) -> np.ndarray:
        state = np.zeros(45, dtype=np.float32)

        # ── workload (10)
        state[0] = float(workload.get("cpu_request", 1.0)) / 64.0
        state[1] = float(workload.get("memory_request_gb", 2.0)) / 256.0
        state[2] = float(workload.get("gpu_count", 0)) / 8.0
        state[3] = float(workload.get("storage_gb", 100.0)) / 2000.0
        state[4] = float(workload.get("network_bandwidth_gbps", 1.0)) / 100.0
        state[5] = float(workload.get("expected_duration_hours", 1.0)) / 720.0
        state[6] = float(workload.get("priority", 2)) / 4.0
        state[7] = float(workload.get("sla_latency_ms", 200.0)) / 1000.0
        state[8] = 1.0 if workload.get("workload_type") in ("batch", "ml_training") else 0.0
        state[9] = 1.0 if workload.get("is_spot_tolerant", False) else 0.0

        # ── pricing (10)
        for i, region in enumerate(REGIONS):
            price = self._extract_price(pricing, region, 0.096)
            state[10 + i] = price / self._p_norm

        # ── carbon (10)
        for i, region in enumerate(REGIONS):
            value = carbon.get(region, 400.0)
            if isinstance(value, dict):
                value = (
                    value.get("gco2_per_kwh")
                    or value.get("carbon_intensity_gco2_per_kwh")
                    or 400.0
                )
            state[20 + i] = float(value) / self._c_norm

        # ── latency (10)
        for i, region in enumerate(REGIONS):
            state[30 + i] = _BASE_LATENCY_MS.get(region, 200.0) / self._l_norm

        # ── history (5) — last 5 normalised total rewards
        recent = history[-5:]
        for i, entry in enumerate(recent):
            state[40 + i] = float(entry.get("reward", 0.0)) / 10.0

        return state