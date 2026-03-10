import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import shap
import torch
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

FEATURE_NAMES: List[str] = [
    "cpu_req", "mem_gb", "gpu", "storage_gb", "net_bw",
    "duration_hr", "priority", "sla_ms", "is_batch", "spot_ok",
    "price_us_e", "price_us_w", "price_eu_w", "price_eu_c",
    "price_ap_se", "price_ap_ne", "price_gcp_c", "price_gcp_eu",
    "price_az_e", "price_az_we",
    "co2_us_e", "co2_us_w", "co2_eu_w", "co2_eu_c",
    "co2_ap_se", "co2_ap_ne", "co2_gcp_c", "co2_gcp_eu",
    "co2_az_e", "co2_az_we",
    "lat_us_e", "lat_us_w", "lat_eu_w", "lat_eu_c",
    "lat_ap_se", "lat_ap_ne", "lat_gcp_c", "lat_gcp_eu",
    "lat_az_e", "lat_az_we",
    "hist_r0", "hist_r1", "hist_r2", "hist_r3", "hist_r4",
]


class SHAPExplainer:
    """
    Wraps a trained PPO model with SHAP KernelExplainer.
    Uses the critic's value function as the explainability proxy —
    higher value = agent considers the state better, which correlates
    with decision quality.
    """

    def __init__(self, model: PPO, n_background: int = 100):
        self.model        = model
        self.n_background = n_background
        self._explainer: Optional[shap.KernelExplainer] = None

    def initialize(self, background: np.ndarray):
        bg = background[: self.n_background].astype(np.float32)

        def _value_fn(states: np.ndarray) -> np.ndarray:
            t = torch.FloatTensor(states).to(self.model.device)
            with torch.no_grad():
                v = self.model.policy.predict_values(t)
            return v.cpu().numpy().flatten()

        self._explainer = shap.KernelExplainer(_value_fn, bg, link="identity")
        logger.info("SHAPExplainer ready with %d background samples.", len(bg))

    def explain(self, state: np.ndarray, nsamples: int = 100) -> Dict:
        if self._explainer is None:
            raise RuntimeError("Call initialize() first.")

        obs    = state.reshape(1, -1).astype(np.float32)
        sv     = self._explainer.shap_values(obs, nsamples=nsamples, silent=True)
        values = (sv[0] if isinstance(sv, list) else sv).flatten()

        feature_map: Dict[str, float] = {
            n: float(v) for n, v in zip(FEATURE_NAMES, values)
        }
        ranked: List[Tuple[str, float]] = sorted(
            feature_map.items(), key=lambda x: abs(x[1]), reverse=True
        )

        return {
            "shap_values":  feature_map,
            "top_drivers":  ranked[:5],
            "base_value":   float(self._explainer.expected_value),
            "explanation":  self._format(ranked[:5]),
        }

    @staticmethod
    def _format(top: List[Tuple[str, float]]) -> str:
        lines = ["Top decision drivers:"]
        for name, val in top:
            arrow = "▲" if val > 0 else "▼"
            group = "cost" if "price" in name else "carbon" if "co2" in name else "latency" if "lat" in name else "workload"
            lines.append(f"  {arrow} {name:12s} [{group:8s}]  {val:+.4f}")
        return "\n".join(lines)