"""
Explanation Formatter
======================
Converts raw SHAP output into human-readable text and
structured JSON for the API response and Kafka decision payload.

Used by:
  - ai_engine/inference/scheduler_agent.py
  - backend/api/routes/scheduling.py
  - ai_engine/kafka/producer.py
"""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

_FEATURE_LABELS: Dict[str, str] = {
    "cpu_request_vcpu": "CPU request (vCPU)",
    "memory_request_gb": "Memory request (GB)",
    "gpu_count": "GPU count",
    "storage_gb": "Storage (GB)",
    "network_bandwidth_gbps": "Network bandwidth (Gbps)",
    "expected_duration_hours": "Expected duration (hrs)",
    "priority": "Workload priority",
    "sla_latency_ms": "SLA latency requirement",
    "workload_type_encoded": "Workload type",
    "is_spot_tolerant": "Spot tolerance",
    "price_cloud_0": "Pricing: us-east-1",
    "price_cloud_1": "Pricing: us-west-2",
    "price_cloud_2": "Pricing: eu-west-1",
    "price_cloud_3": "Pricing: eu-central-1",
    "price_cloud_4": "Pricing: ap-southeast-1",
    "price_cloud_5": "Pricing: ap-northeast-1",
    "price_cloud_6": "Pricing: us-central1 (GCP)",
    "price_cloud_7": "Pricing: europe-west4 (GCP)",
    "price_cloud_8": "Pricing: eastus (Azure)",
    "price_cloud_9": "Pricing: westeurope (Azure)",
    "carbon_region_0": "Carbon: us-east-1 (gCO2/kWh)",
    "carbon_region_1": "Carbon: us-west-2 (gCO2/kWh)",
    "carbon_region_2": "Carbon: eu-west-1 (gCO2/kWh)",
    "carbon_region_3": "Carbon: eu-central-1 (gCO2/kWh)",
    "carbon_region_4": "Carbon: ap-southeast-1 (gCO2/kWh)",
    "carbon_region_5": "Carbon: ap-northeast-1 (gCO2/kWh)",
    "carbon_region_6": "Carbon: us-central1 GCP (gCO2/kWh)",
    "carbon_region_7": "Carbon: europe-west4 GCP (gCO2/kWh)",
    "carbon_region_8": "Carbon: eastus Azure (gCO2/kWh)",
    "carbon_region_9": "Carbon: westeurope Azure (gCO2/kWh)",
    "latency_region_0": "Latency: us-east-1 (ms)",
    "latency_region_1": "Latency: us-west-2 (ms)",
    "latency_region_2": "Latency: eu-west-1 (ms)",
    "latency_region_3": "Latency: eu-central-1 (ms)",
    "latency_region_4": "Latency: ap-southeast-1 (ms)",
    "latency_region_5": "Latency: ap-northeast-1 (ms)",
    "latency_region_6": "Latency: us-central1 GCP (ms)",
    "latency_region_7": "Latency: europe-west4 GCP (ms)",
    "latency_region_8": "Latency: eastus Azure (ms)",
    "latency_region_9": "Latency: westeurope Azure (ms)",
    "history_avg_reward": "Historical avg reward",
    "history_avg_cost_savings": "Historical avg cost savings",
    "history_avg_carbon_savings": "Historical avg carbon savings",
    "history_episode_step": "Episode step progress",
    "history_sla_breach_rate": "Historical SLA breach rate",
}


class ExplanationFormatter:
    """
    Converts raw SHAP dict into human-readable structured explanation.
    Stateless — safe to call from multiple threads.
    """

    def format(self, shap_output: Dict, decision: Dict) -> Dict:
        """
        Formats SHAP output into structured explanation for the API response.

        Args:
            shap_output: Raw dict from SHAPExplainer.explain()
            decision: Decoded action dict from ActionDecoder.decode()

        Returns:
            Formatted explanation dict ready for API + Kafka embedding.
        """
        if not shap_output or shap_output.get("error"):
            return self._empty_explanation()

        top_drivers = self._label_drivers(shap_output.get("top_drivers", []))
        top_positive = self._label_list(shap_output.get("top_positive", []))
        top_negative = self._label_list(shap_output.get("top_negative", []))

        summary = self._build_summary(top_drivers, decision or {})
        confidence = self._compute_confidence(shap_output)

        return {
            "summary": summary,
            "top_drivers": top_drivers,
            "base_value": shap_output.get("base_value", 0.0),
            "top_positive": top_positive,
            "top_negative": top_negative,
            "explanation_ms": shap_output.get("explanation_ms", 0.0),
            "confidence": round(confidence, 3),
        }

    def format_text(self, explanation: Dict) -> str:
        """Returns a single-line human-readable summary string."""
        if not explanation or not explanation.get("top_drivers"):
            return "No explanation available."
        return explanation.get("summary", "Explanation generated.")

    def _label_drivers(self, drivers: List[Dict]) -> List[Dict]:
        return [
            {
                **driver,
                "label": _FEATURE_LABELS.get(
                    driver.get("feature", ""),
                    driver.get("feature", ""),
                ),
            }
            for driver in drivers
        ]

    def _label_list(self, items: List[Dict]) -> List[Dict]:
        return [
            {
                **item,
                "label": _FEATURE_LABELS.get(
                    item.get("feature", ""),
                    item.get("feature", ""),
                ),
            }
            for item in items
        ]

    @staticmethod
    def _build_summary(drivers: List[Dict], decision: Dict) -> str:
        """
        Produces specific summaries for dominant driver categories and directions.
        Falls back to a generic but still informative summary.
        Never emits raw 'unknown' region text in production output.
        """
        cloud = decision.get("cloud", "cloud")
        region = decision.get("region", "")
        purchase = decision.get("purchase_option", "on_demand")

        region_display = region if region and region != "unknown" else "selected region"
        cloud_display = cloud.upper() if cloud and cloud != "unknown" else "Cloud"
        purchase_display = purchase.replace("_", " ")

        if not drivers:
            return (
                f"RL agent placed workload on {cloud_display}/{region_display} "
                f"as {purchase_display}. No single dominant feature identified."
            )

        top = drivers[0]
        feature = top.get("feature", "")
        label = top.get("label", top.get("feature", "a key factor"))
        direction = top.get("direction", "positive")
        value = float(top.get("shap_value", 0.0) or 0.0)
        abs_value = abs(value)

        if "carbon" in feature:
            if direction == "positive":
                return (
                    f"Low carbon intensity in {region_display} is the primary driver "
                    f"(SHAP +{abs_value:.3f}). "
                    f"Placed on {cloud_display}/{region_display} as {purchase_display} "
                    f"to minimise emissions."
                )
            return (
                f"Carbon intensity in {region_display} is elevated (SHAP {value:.3f}). "
                f"{cloud_display}/{region_display} was selected because pricing or latency "
                f"factors outweighed the carbon cost."
            )

        if "price" in feature:
            if direction == "positive":
                return (
                    f"Favourable pricing on {cloud_display}/{region_display} "
                    f"is the top driver (SHAP +{abs_value:.3f}). "
                    f"{purchase_display.title()} option maximises cost reduction."
                )
            return (
                f"Pricing on {cloud_display}/{region_display} is above baseline "
                f"(SHAP {value:.3f}). "
                f"Latency or SLA advantages justify the placement."
            )

        if "latency" in feature:
            if direction == "positive":
                return (
                    f"Low latency to {region_display} is the primary driver "
                    f"(SHAP +{abs_value:.3f}). "
                    f"Routed to {cloud_display}/{region_display} to meet SLA requirements."
                )
            return (
                f"Latency to {region_display} is above preferred threshold "
                f"(SHAP {value:.3f}). "
                f"Cost and carbon savings justify the placement."
            )

        if "spot" in feature:
            return (
                f"Spot tolerance enabled aggressive cost optimisation "
                f"(SHAP +{abs_value:.3f}). "
                f"Placed on {cloud_display}/{region_display} as {purchase_display}."
            )

        if feature in ("cpu_request_vcpu", "memory_request_gb", "gpu_count"):
            action = "favours" if direction == "positive" else "constrains"
            return (
                f"Workload resource profile {action} {cloud_display}/{region_display} "
                f"({label}, SHAP {value:+.3f}). "
                f"Instance selected for resource fit."
            )

        if "history" in feature:
            return (
                f"Historical performance patterns influenced this decision "
                f"({label}, SHAP {value:+.3f}). "
                f"Placed on {cloud_display}/{region_display} as {purchase_display}."
            )

        action = "strengthened" if direction == "positive" else "penalised"
        return (
            f"Placed on {cloud_display}/{region_display} as {purchase_display}. "
            f"Top factor: {label} {action} this choice (SHAP {value:+.3f})."
        )

    @staticmethod
    def _compute_confidence(shap_output: Dict) -> float:
        """
        Confidence proxy based on concentration of SHAP mass in top features.

        Returns a bounded value in approximately [0.35, 0.95] when there is
        meaningful attribution, and 0.0 when SHAP values are absent/degenerate.
        """
        shap_vals = shap_output.get("shap_values", {})
        if not shap_vals:
            return 0.0

        all_vals = list(shap_vals.values())
        feature_count = len(all_vals)
        if feature_count == 0:
            return 0.0

        all_abs = [abs(value) for value in all_vals]
        total_abs = sum(all_abs)
        if total_abs < 1e-9:
            return 0.0

        top_k = min(5, feature_count)
        top_k_abs = sorted(all_abs, reverse=True)[:top_k]
        top_ratio = sum(top_k_abs) / total_abs

        uniform_baseline = top_k / feature_count
        span = 1.0 - uniform_baseline

        if span < 1e-9:
            normalised = 0.5
        else:
            normalised = (top_ratio - uniform_baseline) / span

        confidence = 0.35 + 0.60 * normalised

        base_value = abs(float(shap_output.get("base_value", 0.0) or 0.0))
        if base_value > 1e-9:
            magnitude_ratio = min(1.0, total_abs / base_value)
            confidence = 0.80 * confidence + 0.20 * (0.35 + 0.60 * magnitude_ratio)

        return min(0.95, max(0.35, confidence))

    @staticmethod
    def _empty_explanation() -> Dict:
        return {
            "summary": "Explanation unavailable.",
            "top_drivers": [],
            "base_value": 0.0,
            "top_positive": [],
            "top_negative": [],
            "explanation_ms": 0.0,
            "confidence": 0.0,
        }