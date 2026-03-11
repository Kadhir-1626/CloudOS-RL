"""
SchedulerAgent
===============
Loads the trained PPO model and runs inference.
Integrates Module A SHAP explainability into every decision.

Decision pipeline per request:
  WorkloadRequest
       ↓
  build_state()         → 45-dim float32 from workload + pricing + carbon
       ↓
  PPO.predict()         → MultiDiscrete action [cloud, region, instance, ...]
       ↓
  ActionDecoder.decode() → CloudDecision dict
       ↓
  SHAPExplainer.explain() → per-feature attributions
       ↓
  ExplanationFormatter.format() → human-readable structured explanation
       ↓
  SchedulingDecision (returned to API + Kafka producer)

SHAP is optional — if the model is not yet trained or background dataset
is missing, the agent returns decisions without explanations.

Compatible with:
  - Module G pricing + carbon data
  - Module D Kafka producer (publish_decision)
  - backend/api/routes/scheduling.py
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_PATH    = Path("models/best/best_model")
_VECNORM_PATH  = Path("models/vec_normalize.pkl")


class SchedulerAgent:
    """
    Inference wrapper for the trained PPO scheduling agent.
    Loads model once, serves many requests, with optional SHAP explanations.
    """

    def __init__(
        self,
        model,
        vec_env,
        config:    Dict,
        explainer  = None,
        formatter  = None,
    ):
        self._model     = model
        self._vec_env   = vec_env
        self._config    = config
        self._explainer = explainer
        self._formatter = formatter

        # Import here to avoid circular at module level
        from ai_engine.environment.action_decoder import ActionDecoder
        from ai_engine.environment.state_builder  import StateBuilder
        from ai_engine.cloud_adapter.pricing_cache import PricingCache

        self._decoder       = ActionDecoder()
        self._state_builder = StateBuilder(config)
        self._pricing_cache = PricingCache(config)

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        config:           Dict,
        model_path:       Optional[str] = None,
        vecnorm_path:     Optional[str] = None,
        with_explainer:   bool = True,
        force_bg_regen:   bool = False,
    ) -> "SchedulerAgent":
        """
        Loads a trained PPO model and optionally initialises SHAP explainer.

        Args:
            config:         Project config dict.
            model_path:     Override for model file path.
            vecnorm_path:   Override for VecNormalize pickle path.
            with_explainer: If True, initialise SHAP explainer.
            force_bg_regen: If True, regenerate SHAP background dataset.

        Returns:
            Ready SchedulerAgent. If model not found, returns None.
        """
        import os
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        mp = Path(model_path or config.get("model_path", str(_MODEL_PATH)))
        vp = Path(vecnorm_path or config.get("vecnorm_path", str(_VECNORM_PATH)))

        # ── Load PPO model ────────────────────────────────────────────────
        model_file = mp if mp.suffix else Path(str(mp) + ".zip")
        if not model_file.exists():
            logger.error(
                "SchedulerAgent: model not found at %s. "
                "Run training first: bash scripts/run_training.sh",
                model_file,
            )
            return None

        try:
            model = PPO.load(str(mp))
            logger.info("SchedulerAgent: loaded PPO model from %s", mp)
        except Exception as exc:
            logger.error("SchedulerAgent: model load failed: %s", exc)
            return None

        # ── Load VecNormalize ────────────────────────────────────────────
        vec_env = None
        if vp.exists():
            try:
                import pickle
                from ai_engine.environment.cloud_env import CloudOSEnv

                dummy = DummyVecEnv([lambda: CloudOSEnv(config)])
                with open(vp, "rb") as fh:
                    vec_env = pickle.load(fh)
                vec_env.set_venv(dummy)
                vec_env.training = False
                vec_env.norm_reward = False
                logger.info("SchedulerAgent: VecNormalize loaded from %s", vp)
            except Exception as exc:
                logger.warning("SchedulerAgent: VecNormalize load failed (%s) — running unnormalised.", exc)
                vec_env = None
        else:
            logger.warning("SchedulerAgent: %s not found — running unnormalised.", vp)

        # ── Initialise SHAP explainer ─────────────────────────────────────
        explainer = None
        formatter = None
        if with_explainer:
            try:
                from ai_engine.explainability.shap_explainer     import SHAPExplainer
                from ai_engine.explainability.explanation_formatter import ExplanationFormatter

                explainer = SHAPExplainer.load(
                    model=model,
                    config=config,
                    nsamples=100,
                    force_regen=force_bg_regen,
                )
                formatter = ExplanationFormatter()
                logger.info("SchedulerAgent: SHAP explainer ready.")
            except Exception as exc:
                logger.warning(
                    "SchedulerAgent: SHAP init failed (%s) — decisions will have no explanations.", exc
                )

        return cls(model, vec_env, config, explainer, formatter)

    # -----------------------------------------------------------------------
    # Public: inference
    # -----------------------------------------------------------------------

    def decide(self, workload: Dict) -> Optional[Dict]:
        """
        Main inference entry point.

        Args:
            workload: WorkloadRequest dict (from API schema).

        Returns:
            Decision dict:
            {
              "cloud", "region", "instance_type", "scaling_level",
              "purchase_option", "sla_tier",
              "estimated_cost_per_hr", "cost_savings_pct", "carbon_savings_pct",
              "latency_ms",
              "explanation": {...}   ← from SHAP
            }
            Returns None if model not loaded.
        """
        if self._model is None:
            logger.error("SchedulerAgent.decide: model not loaded.")
            return None

        t0 = time.perf_counter()

        # Build 45-dim state
        state = self.build_state(workload)

        # Normalise if VecNormalize is available
        obs = state.reshape(1, -1)
        if self._vec_env is not None:
            try:
                obs = self._vec_env.normalize_obs(obs)
            except Exception:
                pass

        # PPO inference
        action, _ = self._model.predict(obs, deterministic=True)
        decoded   = self._decoder.decode(action[0] if action.ndim > 1 else action)

        # Cost + carbon savings estimates
        cost_per_hr      = self.estimate_cost_per_hr(decoded)
        cost_savings_pct = self.cost_savings_pct(decoded)
        carbon_savings   = self.carbon_savings_pct(decoded)

        latency_ms = (time.perf_counter() - t0) * 1000

        # SHAP explanation
        explanation: Dict = {}
        if self._explainer is not None and self._formatter is not None:
            try:
                raw_shap    = self._explainer.explain(state)
                explanation = self._formatter.format(raw_shap, decoded)
            except Exception as exc:
                logger.warning("SchedulerAgent: SHAP explain failed: %s", exc)
                explanation = {"error": str(exc)}

        return {
            **decoded,
            "estimated_cost_per_hr": round(cost_per_hr,      4),
            "cost_savings_pct":      round(cost_savings_pct, 2),
            "carbon_savings_pct":    round(carbon_savings,   2),
            "latency_ms":            round(latency_ms,        2),
            "explanation":           explanation,
        }

    def build_state(self, workload: Dict) -> np.ndarray:
        """Builds a 45-dim state vector from a workload request."""
        pricing = self._pricing_cache.get_current_pricing()
        carbon  = self._load_carbon()

        return self._state_builder.build(
            workload=workload,
            pricing=pricing,
            carbon=carbon,
            history=[],
        )

    # -----------------------------------------------------------------------
    # Public: cost + carbon helpers
    # -----------------------------------------------------------------------

    def estimate_cost_per_hr(self, decoded: Dict) -> float:
        """Returns estimated $/hr for the decoded action."""
        try:
            region   = decoded.get("region",          "us-east-1")
            instance = decoded.get("instance_type",   "m5.large")
            purchase = decoded.get("purchase_option", "on_demand")
            return self._pricing_cache.get_price(region, instance, purchase)
        except Exception:
            return 0.096

    def cost_savings_pct(self, decoded: Dict) -> float:
        """
        Returns estimated cost savings % vs on-demand baseline.
        Formula: (1 - actual/on_demand) * 100
        """
        try:
            region   = decoded.get("region",          "us-east-1")
            instance = decoded.get("instance_type",   "m5.large")
            purchase = decoded.get("purchase_option", "on_demand")
            actual   = self._pricing_cache.get_price(region, instance, purchase)
            baseline = self._pricing_cache.get_price(region, instance, "on_demand")
            if baseline <= 0:
                return 0.0
            return max(0.0, (1.0 - actual / baseline) * 100.0)
        except Exception:
            return 0.0

    def carbon_savings_pct(self, decoded: Dict) -> float:
        """
        Returns estimated carbon savings % vs us-east-1 baseline (415 gCO2/kWh).
        """
        try:
            region  = decoded.get("region", "us-east-1")
            carbon  = self._load_carbon()
            actual  = carbon.get(region, 415.0)
            baseline = carbon.get("us-east-1", 415.0)
            if baseline <= 0:
                return 0.0
            return max(0.0, (1.0 - actual / baseline) * 100.0)
        except Exception:
            return 0.0

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _load_carbon(self) -> Dict[str, float]:
        """Reads latest carbon data from Module G pipeline file."""
        try:
            carbon_path = Path(
                self._config.get("data_pipeline", {}).get(
                    "carbon_output_path", "data/carbon/carbon_intensity.json"
                )
            )
            if carbon_path.exists():
                import json
                with open(carbon_path, encoding="utf-8") as fh:
                    raw = json.load(fh)
                return {
                    r: float(e.get("gco2_per_kwh", 400.0))
                    for r, e in raw.items()
                }
        except Exception:
            pass
        return {
            "us-east-1": 415.0, "us-west-2": 192.0,
            "eu-west-1": 316.0, "eu-north-1": 42.0,
            "eu-west-3": 58.0,  "ca-central-1": 89.0,
        }