"""
Explanation Formatter
======================
Converts raw SHAP output into human-readable text and
structured JSON for the API response and Kafka decision payload.

Used by:
  - ai_engine/inference/scheduler_agent.py  (formats explain() output)
  - backend/api/routes/scheduling.py        (embeds in API response)
  - ai_engine/kafka/producer.py             (serialised into decision message)

Output schema (embedded in SchedulingDecision.explanation):
  {
    "summary":         "Cost optimised via spot instance. Carbon risk from ap-south-1.",
    "top_drivers":     [{"feature": str, "shap_value": float, "direction": str, "label": str}],
    "base_value":      float,
    "top_positive":    [{"feature": str, "shap_value": float, "label": str}],
    "top_negative":    [{"feature": str, "shap_value": float, "label": str}],
    "explanation_ms":  float,
    "confidence":      float,   <- |base_value + sum(top_shap)| / max_possible
  }
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Human-readable labels for state features
_FEATURE_LABELS: Dict[str, str] = {
    # Workload
    "cpu_request_vcpu":          "CPU request (vCPU)",
    "memory_request_gb":         "Memory request (GB)",
    "gpu_count":                 "GPU count",
    "storage_gb":                "Storage (GB)",
    "network_bandwidth_gbps":    "Network bandwidth (Gbps)",
    "expected_duration_hours":   "Expected duration (hrs)",
    "priority":                  "Workload priority",
    "sla_latency_ms":            "SLA latency requirement",
    "workload_type_encoded":     "Workload type",
    "is_spot_tolerant":          "Spot tolerance",
    # Pricing
    "price_cloud_0":             "Pricing: us-east-1",
    "price_cloud_1":             "Pricing: us-west-2",
    "price_cloud_2":             "Pricing: eu-west-1",
    "price_cloud_3":             "Pricing: eu-central-1",
    "price_cloud_4":             "Pricing: ap-southeast-1",
    "price_cloud_5":             "Pricing: ap-northeast-1",
    "price_cloud_6":             "Pricing: us-central1 (GCP)",
    "price_cloud_7":             "Pricing: europe-west4 (GCP)",
    "price_cloud_8":             "Pricing: eastus (Azure)",
    "price_cloud_9":             "Pricing: westeurope (Azure)",
    # Carbon
    "carbon_region_0":           "Carbon: us-east-1 (gCO2/kWh)",
    "carbon_region_1":           "Carbon: us-west-2 (gCO2/kWh)",
    "carbon_region_2":           "Carbon: eu-west-1 (gCO2/kWh)",
    "carbon_region_3":           "Carbon: eu-central-1 (gCO2/kWh)",
    "carbon_region_4":           "Carbon: ap-southeast-1 (gCO2/kWh)",
    "carbon_region_5":           "Carbon: ap-northeast-1 (gCO2/kWh)",
    "carbon_region_6":           "Carbon: us-central1 (GCP)",
    "carbon_region_7":           "Carbon: europe-west4 (GCP)",
    "carbon_region_8":           "Carbon: eastus (Azure)",
    "carbon_region_9":           "Carbon: westeurope (Azure)",
    # Latency
    "latency_region_0":          "Latency: us-east-1 (ms)",
    "latency_region_1":          "Latency: us-west-2 (ms)",
    "latency_region_2":          "Latency: eu-west-1 (ms)",
    "latency_region_3":          "Latency: eu-central-1 (ms)",
    "latency_region_4":          "Latency: ap-southeast-1 (ms)",
    "latency_region_5":          "Latency: ap-northeast-1 (ms)",
    "latency_region_6":          "Latency: us-central1 (GCP)",
    "latency_region_7":          "Latency: europe-west4 (GCP)",
    "latency_region_8":          "Latency: eastus (Azure)",
    "latency_region_9":          "Latency: westeurope (Azure)",
    # History
    "history_avg_reward":        "Historical avg reward",
    "history_avg_cost_savings":  "Historical avg cost savings",
    "history_avg_carbon_savings":"Historical avg carbon savings",
    "history_episode_step":      "Episode step progress",
    "history_sla_breach_rate":   "Historical SLA breach rate",
}


class ExplanationFormatter:
    """
    Converts raw SHAP dict into human-readable structured explanation.
    Stateless — safe to call from multiple threads.
    """

    def format(
        self,
        shap_output:    Dict,
        decision:       Dict,
    ) -> Dict:
        """
        Formats SHAP output into structured explanation for the API response.

        Args:
            shap_output:  Raw dict from SHAPExplainer.explain()
            decision:     Decoded action dict from ActionDecoder.decode()

        Returns:
            Formatted explanation dict ready for API + Kafka embedding.
        """
        if not shap_output or shap_output.get("error"):
            return self._empty_explanation()

        top_drivers  = self._label_drivers(shap_output.get("top_drivers",  []))
        top_positive = self._label_list(shap_output.get("top_positive", []))
        top_negative = self._label_list(shap_output.get("top_negative", []))

        summary    = self._build_summary(top_drivers, decision)
        confidence = self._compute_confidence(shap_output)

        return {
            "summary":        summary,
            "top_drivers":    top_drivers,
            "base_value":     shap_output.get("base_value",     0.0),
            "top_positive":   top_positive,
            "top_negative":   top_negative,
            "explanation_ms": shap_output.get("explanation_ms", 0.0),
            "confidence":     round(confidence, 3),
        }

    def format_text(self, explanation: Dict) -> str:
        """Returns a single-line human-readable summary string."""
        if not explanation or not explanation.get("top_drivers"):
            return "No explanation available."
        return explanation.get("summary", "Explanation generated.")

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _label_drivers(self, drivers: List[Dict]) -> List[Dict]:
        return [
            {
                **d,
                "label": _FEATURE_LABELS.get(d.get("feature", ""), d.get("feature", "")),
            }
            for d in drivers
        ]

    def _label_list(self, items: List[Dict]) -> List[Dict]:
        return [
            {
                **item,
                "label": _FEATURE_LABELS.get(item.get("feature", ""), item.get("feature", "")),
            }
            for item in items
        ]

    @staticmethod
    def _build_summary(drivers: List[Dict], decision: Dict) -> str:
        """Generates a concise 1-sentence explanation summary."""
        if not drivers:
            return "Decision made by RL agent. No dominant feature identified."

        top = drivers[0]
        label     = top.get("label", top.get("feature", "unknown"))
        direction = top.get("direction", "positive")
        purchase  = decision.get("purchase_option", "on_demand")
        cloud     = decision.get("cloud",           "aws")
        region    = decision.get("region",          "us-east-1")

        if "carbon" in top.get("feature", ""):
            return (
                f"Carbon intensity in {region} is the primary driver. "
                f"Scheduled on {cloud}/{region} as {purchase} to reduce emissions."
            )
        elif "price" in top.get("feature", ""):
            return (
                f"Cost optimisation via {purchase} on {cloud}/{region}. "
                f"Pricing differential ({label}) is the top decision factor."
            )
        elif "latency" in top.get("feature", ""):
            return (
                f"Latency constraint is the primary driver. "
                f"Routed to {cloud}/{region} for lowest observed latency."
            )
        elif direction == "positive":
            return (
                f"Scheduled on {cloud}/{region} ({purchase}). "
                f"Top positive driver: {label}."
            )
        else:
            return (
                f"Scheduled on {cloud}/{region} ({purchase}). "
                f"Top risk factor: {label}."
            )

    @staticmethod
    def _compute_confidence(shap_output: Dict) -> float:
        """
        Confidence proxy: fraction of total SHAP mass explained by top-5 drivers.
        Range [0, 1]. Higher = explanation is more concentrated in fewer features.
        """
        shap_vals = shap_output.get("shap_values", {})
        if not shap_vals:
            return 0.0

        all_abs   = [abs(v) for v in shap_vals.values()]
        total_abs = sum(all_abs)
        if total_abs < 1e-9:
            return 0.0

        top5_abs = sorted(all_abs, reverse=True)[:5]
        return min(1.0, sum(top5_abs) / total_abs)

    @staticmethod
    def _empty_explanation() -> Dict:
        return {
            "summary":        "Explanation unavailable.",
            "top_drivers":    [],
            "base_value":     0.0,
            "top_positive":   [],
            "top_negative":   [],
            "explanation_ms": 0.0,
            "confidence":     0.0,
        }