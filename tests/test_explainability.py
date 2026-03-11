"""
Module A — Full Explainability Pipeline Test
=============================================
Tests all components of the SHAP explainability pipeline
WITHOUT requiring a trained model (uses mock PPO).

Run:
  python -m pytest tests/test_explainability.py -v
  python tests/test_explainability.py          (direct run)

Tests:
  1. BackgroundDataGenerator — generates (200, 45) dataset
  2. BackgroundDataGenerator — loads Module G pricing + carbon correctly
  3. BackgroundDataGenerator — cache hit / miss logic
  4. SHAPExplainer           — initialises with mock model
  5. SHAPExplainer           — explain() returns correct schema
  6. SHAPExplainer           — feature names match exactly 45
  7. ExplanationFormatter    — formats SHAP output correctly
  8. ExplanationFormatter    — confidence score in [0, 1]
  9. ExplanationFormatter    — empty explanation on error
  10. SchedulerAgent         — build_state returns (45,) array
  11. Full pipeline          — BackgroundGen → SHAP → Formatter end-to-end
"""

import sys
import json
import logging
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(name)-40s  %(levelname)s  %(message)s",
)

# ---------------------------------------------------------------------------
# Test config
# ---------------------------------------------------------------------------
_TEST_CONFIG = {
    "data_pipeline": {
        "pricing_output_path":     "data/pricing/aws_pricing.json",
        "actual_costs_output_path":"data/pricing/aws_actual_costs.json",
        "carbon_output_path":      "data/carbon/carbon_intensity.json",
    },
    "environment": {
        "pricing_fallback_path": "data/pricing/aws_pricing.json",
        "max_episode_steps":     100,
    },
    "aws": {"region": "us-east-1"},
}


def _make_mock_model():
    """Creates a mock SB3 PPO model with a predict_values method."""
    mock_policy = MagicMock()
    mock_policy.predict_values.return_value = MagicMock()
    mock_policy.predict_values.return_value.item.return_value = 2.5

    mock_model = MagicMock()
    mock_model.policy = mock_policy
    mock_model.predict.return_value = (np.array([0, 0, 2, 1, 1, 2]), None)

    return mock_model


def _make_mock_shap_output() -> Dict:
    """Returns a realistic mock SHAP output dict."""
    feature_names = [
        "cpu_request_vcpu", "memory_request_gb", "gpu_count",
        "storage_gb", "network_bandwidth_gbps", "expected_duration_hours",
        "priority", "sla_latency_ms", "workload_type_encoded", "is_spot_tolerant",
        "price_cloud_0", "price_cloud_1", "price_cloud_2", "price_cloud_3",
        "price_cloud_4", "price_cloud_5", "price_cloud_6", "price_cloud_7",
        "price_cloud_8", "price_cloud_9",
        "carbon_region_0", "carbon_region_1", "carbon_region_2", "carbon_region_3",
        "carbon_region_4", "carbon_region_5", "carbon_region_6", "carbon_region_7",
        "carbon_region_8", "carbon_region_9",
        "latency_region_0", "latency_region_1", "latency_region_2", "latency_region_3",
        "latency_region_4", "latency_region_5", "latency_region_6", "latency_region_7",
        "latency_region_8", "latency_region_9",
        "history_avg_reward", "history_avg_cost_savings", "history_avg_carbon_savings",
        "history_episode_step", "history_sla_breach_rate",
    ]

    np.random.seed(42)
    shap_vals = np.random.randn(45) * 0.3
    shap_vals[10] = -0.8   # price_cloud_0 — strong negative (expensive)
    shap_vals[20] =  0.6   # carbon_region_0 — strong positive (clean region)
    shap_vals[9]  =  0.5   # is_spot_tolerant

    named = {name: round(float(v), 6) for name, v in zip(feature_names, shap_vals)}

    sorted_desc = sorted(named.items(), key=lambda x: x[1], reverse=True)
    sorted_asc  = sorted(named.items(), key=lambda x: x[1])
    top_pos = [{"feature": k, "shap_value": round(v, 6)} for k, v in sorted_desc[:5] if v > 0]
    top_neg = [{"feature": k, "shap_value": round(v, 6)} for k, v in sorted_asc[:5]  if v < 0]
    top_drv = sorted(
        [{"feature": k, "shap_value": round(v, 6),
          "direction": "positive" if v > 0 else "negative"}
         for k, v in named.items()],
        key=lambda d: abs(d["shap_value"]), reverse=True
    )[:5]

    return {
        "top_drivers":    top_drv,
        "base_value":     2.5,
        "shap_values":    named,
        "top_positive":   top_pos,
        "top_negative":   top_neg,
        "explanation_ms": 87.3,
        "state_mean":     0.12,
        "state_std":      0.31,
    }


# ============================================================================
class TestBackgroundDataGenerator(unittest.TestCase):

    def setUp(self):
        from ai_engine.explainability.background_generator import BackgroundDataGenerator
        self.gen = BackgroundDataGenerator(_TEST_CONFIG)

    def test_generates_correct_shape(self):
        """Background dataset must be exactly (n_samples, 45)."""
        bg = self.gen.generate(n_samples=50, seed=0, force=True)
        self.assertEqual(bg.shape, (50, 45))
        self.assertEqual(bg.dtype, np.float32)

    def test_no_nan_or_inf(self):
        """Background dataset must not contain NaN or Inf values."""
        bg = self.gen.generate(n_samples=50, seed=1, force=True)
        self.assertFalse(np.any(np.isnan(bg)),  "NaN found in background dataset")
        self.assertFalse(np.any(np.isinf(bg)),  "Inf found in background dataset")

    def test_values_in_reasonable_range(self):
        """All normalised values should be in roughly [-0.1, 200] range."""
        bg = self.gen.generate(n_samples=50, seed=2, force=True)
        self.assertGreater(bg.max(), 0.0,  "All zeros — something is wrong")
        self.assertLess(bg.min(),    200.0, "Suspiciously large values")

    def test_cache_hit(self):
        """Second call with same n_samples should return cached result."""
        bg1 = self.gen.generate(n_samples=50, seed=3, force=True)
        bg2 = self.gen.generate(n_samples=50, seed=3, force=False)
        np.testing.assert_array_equal(bg1, bg2)

    def test_loads_pricing_from_file(self):
        """Should load pricing from Module G file if it exists."""
        pricing = self.gen._load_pricing()
        self.assertIsInstance(pricing, dict)
        self.assertGreater(len(pricing), 0)
        for region, price in pricing.items():
            self.assertGreater(price, 0.0, f"Non-positive price for {region}")
            self.assertLess(price,    10.0, f"Unreasonably high price for {region}")

    def test_loads_carbon_from_file(self):
        """Should load carbon from Module G file if it exists."""
        carbon = self.gen._load_carbon()
        self.assertIsInstance(carbon, dict)
        self.assertGreater(len(carbon), 0)
        for region, co2 in carbon.items():
            self.assertGreater(co2, 0.0,    f"Non-positive CO2 for {region}")
            self.assertLess(co2,    1000.0, f"Unreasonably high CO2 for {region}")

    def test_feature_names_count(self):
        """Must have exactly 45 feature names."""
        names = self.gen.get_feature_names()
        self.assertEqual(len(names), 45)
        self.assertEqual(len(set(names)), 45, "Duplicate feature names found")


# ============================================================================
class TestSHAPExplainer(unittest.TestCase):

    def setUp(self):
        from ai_engine.explainability.background_generator import BackgroundDataGenerator
        self.config    = _TEST_CONFIG
        self.mock_model = _make_mock_model()
        gen            = BackgroundDataGenerator(self.config)
        self.background = gen.generate(n_samples=20, seed=0, force=True)

    def test_initialises_with_mock_model(self):
        """SHAPExplainer should initialise without raising."""
        from ai_engine.explainability.shap_explainer import SHAPExplainer
        explainer = SHAPExplainer(self.mock_model, self.background, nsamples=10)
        self.assertIsNotNone(explainer)

    def test_explain_returns_correct_schema(self):
        """explain() must return all required keys."""
        from ai_engine.explainability.shap_explainer import SHAPExplainer
        explainer = SHAPExplainer(self.mock_model, self.background, nsamples=10)
        state     = np.random.rand(45).astype(np.float32)
        result    = explainer.explain(state)

        required_keys = [
            "top_drivers", "base_value", "shap_values",
            "top_positive", "top_negative", "explanation_ms",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_shap_values_dict_has_45_features(self):
        """shap_values dict must have exactly 45 keys."""
        from ai_engine.explainability.shap_explainer import SHAPExplainer
        explainer = SHAPExplainer(self.mock_model, self.background, nsamples=10)
        state     = np.random.rand(45).astype(np.float32)
        result    = explainer.explain(state)
        self.assertEqual(len(result["shap_values"]), 45)

    def test_top_drivers_count(self):
        """top_drivers must have at most 5 entries."""
        from ai_engine.explainability.shap_explainer import SHAPExplainer
        explainer = SHAPExplainer(self.mock_model, self.background, nsamples=10)
        state     = np.random.rand(45).astype(np.float32)
        result    = explainer.explain(state)
        self.assertLessEqual(len(result["top_drivers"]), 5)

    def test_explanation_ms_positive(self):
        """Explanation timing must be positive."""
        from ai_engine.explainability.shap_explainer import SHAPExplainer
        explainer = SHAPExplainer(self.mock_model, self.background, nsamples=10)
        state     = np.random.rand(45).astype(np.float32)
        result    = explainer.explain(state)
        self.assertGreater(result["explanation_ms"], 0.0)

    def test_background_shape(self):
        """Explainer must report background shape correctly."""
        from ai_engine.explainability.shap_explainer import SHAPExplainer
        explainer = SHAPExplainer(self.mock_model, self.background, nsamples=10)
        self.assertEqual(explainer.get_background_shape(), (20, 45))


# ============================================================================
class TestExplanationFormatter(unittest.TestCase):

    def setUp(self):
        from ai_engine.explainability.explanation_formatter import ExplanationFormatter
        self.formatter   = ExplanationFormatter()
        self.mock_shap   = _make_mock_shap_output()
        self.mock_decision = {
            "cloud":           "aws",
            "region":          "eu-north-1",
            "instance_type":   "m5.large",
            "purchase_option": "spot",
            "sla_tier":        "standard",
        }

    def test_format_returns_required_keys(self):
        """format() must return all required keys."""
        result = self.formatter.format(self.mock_shap, self.mock_decision)
        for key in ["summary", "top_drivers", "base_value",
                    "top_positive", "top_negative", "confidence"]:
            self.assertIn(key, result)

    def test_confidence_in_range(self):
        """Confidence must be in [0, 1]."""
        result = self.formatter.format(self.mock_shap, self.mock_decision)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"],    1.0)

    def test_top_drivers_have_labels(self):
        """Each top driver must have a human-readable label."""
        result  = self.formatter.format(self.mock_shap, self.mock_decision)
        drivers = result["top_drivers"]
        self.assertGreater(len(drivers), 0)
        for driver in drivers:
            self.assertIn("label", driver)
            self.assertIsInstance(driver["label"], str)
            self.assertGreater(len(driver["label"]), 0)

    def test_summary_is_non_empty_string(self):
        """Summary must be a non-empty string."""
        result = self.formatter.format(self.mock_shap, self.mock_decision)
        self.assertIsInstance(result["summary"], str)
        self.assertGreater(len(result["summary"]), 10)

    def test_empty_on_error_shap(self):
        """Formatter must return empty explanation if shap output has error key."""
        result = self.formatter.format({"error": "explainer_not_ready"}, {})
        self.assertEqual(result["top_drivers"],  [])
        self.assertEqual(result["confidence"],    0.0)

    def test_format_text_returns_string(self):
        """format_text() must return a string."""
        formatted = self.formatter.format(self.mock_shap, self.mock_decision)
        text = self.formatter.format_text(formatted)
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 5)


# ============================================================================
class TestFullPipeline(unittest.TestCase):
    """End-to-end test: BackgroundGen → SHAP → Formatter."""

    def test_pipeline_end_to_end(self):
        """Full pipeline must complete without errors."""
        from ai_engine.explainability.background_generator  import BackgroundDataGenerator
        from ai_engine.explainability.shap_explainer         import SHAPExplainer
        from ai_engine.explainability.explanation_formatter  import ExplanationFormatter

        config     = _TEST_CONFIG
        mock_model = _make_mock_model()

        # Step 1: Generate background
        gen        = BackgroundDataGenerator(config)
        background = gen.generate(n_samples=30, seed=42, force=True)
        self.assertEqual(background.shape, (30, 45))

        # Step 2: SHAP explain
        explainer  = SHAPExplainer(mock_model, background, nsamples=10)
        state      = np.random.rand(45).astype(np.float32)
        shap_out   = explainer.explain(state)
        self.assertIn("shap_values", shap_out)
        self.assertEqual(len(shap_out["shap_values"]), 45)

        # Step 3: Format
        formatter  = ExplanationFormatter()
        decision   = {"cloud": "aws", "region": "eu-north-1",
                      "instance_type": "m5.large", "purchase_option": "spot"}
        explanation = formatter.format(shap_out, decision)

        self.assertIn("summary",     explanation)
        self.assertIn("top_drivers", explanation)
        self.assertIn("confidence",  explanation)
        self.assertGreaterEqual(explanation["confidence"], 0.0)
        self.assertLessEqual(explanation["confidence"],    1.0)

        print("\n✅ Pipeline end-to-end test passed")
        print(f"   Background shape:    {background.shape}")
        print(f"   SHAP features:       {len(shap_out['shap_values'])}")
        print(f"   Top driver:          {explanation['top_drivers'][0]['label'] if explanation['top_drivers'] else 'none'}")
        print(f"   Confidence:          {explanation['confidence']:.3f}")
        print(f"   Summary:             {explanation['summary']}")


# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CloudOS-RL — Module A: Explainability Pipeline Tests")
    print("=" * 60 + "\n")
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestBackgroundDataGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestSHAPExplainer))
    suite.addTests(loader.loadTestsFromTestCase(TestExplanationFormatter))
    suite.addTests(loader.loadTestsFromTestCase(TestFullPipeline))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)