import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ai_engine.cloud_adapter.pricing_cache import PricingCache
from ai_engine.environment.action_decoder import ActionDecoder
from ai_engine.environment.cloud_env import CloudOSEnv
from ai_engine.environment.state_builder import StateBuilder
from ai_engine.explainability.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)

_PURCHASE_MULT = {
    "on_demand": 1.00, "spot": 0.33, "preemptible": 0.30,
    "savings_plan": 0.55, "reserved_1yr": 0.60, "reserved_3yr": 0.40,
}
_BASELINE_CARBON = {
    "us-east-1": 415.0, "us-west-2": 192.0, "eu-west-1": 316.0,
    "eu-central-1": 338.0, "ap-southeast-1": 453.0, "ap-northeast-1": 506.0,
    "us-central1": 360.0, "europe-west4": 284.0,
    "eastus": 400.0, "westeurope": 290.0,
}
_BASELINE_COST = 0.096  # m5.large on-demand


class SchedulerAgent:
    MODEL_PATH  = "models/best/best_model"
    VECNORM_PATH = "models/vec_normalize.pkl"

    def __init__(self, model: PPO, vec_norm: Optional[VecNormalize], config: Dict):
        self._model        = model
        self._vec_norm     = vec_norm
        self._state_builder = StateBuilder(config)
        self._decoder      = ActionDecoder()
        self._pricing      = PricingCache(config)
        self._shap: Optional[SHAPExplainer] = None

    @classmethod
    def load(cls, config_path: str = "config/settings.yaml") -> "SchedulerAgent":
        with open(config_path) as fh:
            config = yaml.safe_load(fh)

        model    = PPO.load(cls.MODEL_PATH, device="cpu")
        vec_norm = None

        vp = Path(cls.VECNORM_PATH)
        if vp.exists():
            dummy    = DummyVecEnv([lambda: CloudOSEnv(config)])
            vec_norm = VecNormalize.load(str(vp), dummy)
            vec_norm.training   = False
            vec_norm.norm_reward = False

        logger.info("SchedulerAgent loaded.")
        return cls(model, vec_norm, config)

    # ── public API ──────────────────────────────────────────────────────────

    def build_state(self, workload: Dict) -> np.ndarray:
        return self._state_builder.build(
            workload=workload,
            pricing=self._pricing.get_current_pricing(),
            carbon=_BASELINE_CARBON,
            history=[],
        ).astype(np.float32)

    def decide(self, state: np.ndarray) -> Tuple[Dict, Dict]:
        obs = state.reshape(1, -1)
        if self._vec_norm is not None:
            obs = self._vec_norm.normalize_obs(obs)

        action, _ = self._model.predict(obs, deterministic=True)
        decoded   = self._decoder.decode(action[0])

        explanation: Dict = {}
        if self._shap is not None:
            try:
                explanation = self._shap.explain(state)
            except Exception as exc:
                logger.warning("SHAP explain failed: %s", exc)

        return decoded, explanation

    def enable_shap(self, background: np.ndarray):
        self._shap = SHAPExplainer(self._model)
        self._shap.initialize(background)

    # ── financial helpers ───────────────────────────────────────────────────

    def estimate_cost_per_hr(self, action: Dict) -> float:
        pricing  = self._pricing.get_current_pricing()
        base     = pricing.get(action["generic_region"], {}).get(action["instance_type"], _BASELINE_COST)
        return base * _PURCHASE_MULT.get(action["purchase_option"], 1.0) * action["scaling_level"]

    def cost_savings_pct(self, action: Dict) -> float:
        baseline = _BASELINE_COST * action["scaling_level"]
        actual   = self.estimate_cost_per_hr(action)
        return max(0.0, (baseline - actual) / baseline * 100.0)

    def carbon_savings_pct(self, action: Dict) -> float:
        region    = action.get("generic_region", "us-east-1")
        carbon    = _BASELINE_CARBON.get(region, 400.0) * 0.1   # gCO2/hr @ 100W
        baseline  = _BASELINE_CARBON["us-east-1"] * 0.1
        return max(0.0, (baseline - carbon) / baseline * 100.0)