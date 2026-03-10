"""
Data Normalizer
================
Merges raw outputs from all three fetchers and writes canonical JSON files.

Files written:
  data/pricing/aws_pricing.json        <- PricingCache reads this on TTL expiry
  data/pricing/aws_actual_costs.json   <- reward calibration, anomaly reference
  data/carbon/carbon_intensity.json    <- CloudOSEnv reads per-episode

All writes are ATOMIC (write to .tmp.json then rename) so readers never
see a partial file even if the writer crashes mid-write.

Compatible with:
  - ai_engine/cloud_adapter/pricing_cache.py  PricingCache._refresh()
  - ai_engine/environment/cloud_env.py        _load_carbon_from_file()
  - ai_engine/environment/state_builder.py    carbon parameter
  - ai_engine/environment/reward.py           pricing parameter
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PRICING_PATH      = Path("data/pricing/aws_pricing.json")
_DEFAULT_ACTUAL_COSTS_PATH = Path("data/pricing/aws_actual_costs.json")
_DEFAULT_CARBON_PATH       = Path("data/carbon/carbon_intensity.json")


class DataNormalizer:
    """
    Stateless transformer and file writer.
    Safe to call from multiple threads — all writes are atomic.
    """

    def __init__(self, config: Dict):
        dp = config.get("data_pipeline", {})
        self._pricing_path      = Path(dp.get("pricing_output_path",      str(_DEFAULT_PRICING_PATH)))
        self._actual_costs_path = Path(dp.get("actual_costs_output_path", str(_DEFAULT_ACTUAL_COSTS_PATH)))
        self._carbon_path       = Path(dp.get("carbon_output_path",       str(_DEFAULT_CARBON_PATH)))

    # -----------------------------------------------------------------------
    # Public: pricing
    # -----------------------------------------------------------------------

    def normalize_pricing(
        self,
        raw_pricing: Dict[str, Dict],
        cur_data:    Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict]:
        """
        Merges Pricing API data with optional CUR blended rates.
        CUR blended_rates override the on_demand price where available
        to give the RL agent the real cost the account is paying.

        Writes result to data/pricing/aws_pricing.json.
        Returns the merged dict.
        """
        blended_rates = (cur_data or {}).get("blended_rates", {})
        merged: Dict[str, Dict] = {}

        for region, instances in raw_pricing.items():
            region_entry: Dict = {}
            nested:       Dict = {}
            vcpu_rates = []

            for key, value in instances.items():
                # Skip meta keys
                if key.startswith("_") or key == "on_demand_per_vcpu_hr":
                    continue

                # Colon-key variants (already built by AWSPricingFetcher)
                if ":" in key:
                    region_entry[key] = value
                    continue

                # Flat instance key — this is on_demand price
                if isinstance(value, (int, float)):
                    on_demand = float(value)

                    # Override with real CUR blended rate if available
                    blended = blended_rates.get(region, {}).get(key)
                    calibrated_price = blended if blended else on_demand

                    # Retrieve full nested prices if fetcher built _nested
                    nested_data = instances.get("_nested", {}).get(key, {})

                    spot      = nested_data.get("spot",         on_demand * 0.33)
                    sav_plan  = nested_data.get("savings_plan", on_demand * 0.55)
                    res_1yr   = nested_data.get("reserved_1yr", on_demand * 0.60)
                    res_3yr   = nested_data.get("reserved_3yr", on_demand * 0.40)

                    # Flat on_demand key — PricingCache primary lookup
                    region_entry[key] = round(calibrated_price, 6)

                    # Colon keys for PricingCache.get_price()
                    region_entry[f"{key}:on_demand"]    = round(on_demand,        6)
                    region_entry[f"{key}:spot"]         = round(spot,             6)
                    region_entry[f"{key}:savings_plan"] = round(sav_plan,         6)
                    region_entry[f"{key}:reserved_1yr"] = round(res_1yr,          6)
                    region_entry[f"{key}:reserved_3yr"] = round(res_3yr,          6)
                    region_entry[f"{key}:blended"]      = round(calibrated_price, 6)

                    nested[key] = {
                        "on_demand":     round(on_demand,        6),
                        "spot":          round(spot,             6),
                        "savings_plan":  round(sav_plan,         6),
                        "reserved_1yr":  round(res_1yr,          6),
                        "reserved_3yr":  round(res_3yr,          6),
                        "blended":       round(calibrated_price, 6),
                    }

                    # Collect for per-vcpu average computation
                    from ai_engine.data_pipeline.aws_pricing_fetcher import INSTANCE_VCPU
                    vcpus = INSTANCE_VCPU.get(key, 2)
                    vcpu_rates.append(on_demand / vcpus)

            # on_demand_per_vcpu_hr — StateBuilder state[10:20]
            if "on_demand_per_vcpu_hr" in instances:
                region_entry["on_demand_per_vcpu_hr"] = round(
                    float(instances["on_demand_per_vcpu_hr"]), 6
                )
            elif vcpu_rates:
                region_entry["on_demand_per_vcpu_hr"] = round(
                    sum(vcpu_rates) / len(vcpu_rates), 6
                )

            region_entry["_nested"]  = nested
            region_entry["_region"]  = region
            region_entry["_updated"] = datetime.now(tz=timezone.utc).isoformat()
            merged[region] = region_entry

        self._atomic_write(self._pricing_path, merged)
        logger.info("DataNormalizer: wrote pricing → %s (%d regions)", self._pricing_path, len(merged))
        return merged

    # -----------------------------------------------------------------------
    # Public: actual costs
    # -----------------------------------------------------------------------

    def normalize_actual_costs(self, cur_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Writes CUR blended_rates + usage_summary to aws_actual_costs.json.
        Used for reward calibration and cost anomaly monitoring.
        """
        out = {
            "blended_rates":   cur_data.get("blended_rates",  {}),
            "usage_summary":   cur_data.get("usage_summary",  {}),
            "status":          cur_data.get("status",         "unknown"),
            "fetch_timestamp": cur_data.get("fetch_timestamp", ""),
            "period_days":     cur_data.get("period_days",    30),
        }

        anomalies = [
            r for r, s in out["usage_summary"].items()
            if s.get("anomaly", False)
        ]
        if anomalies:
            logger.warning("Cost anomalies detected in: %s", anomalies)

        self._atomic_write(self._actual_costs_path, out)
        logger.info("DataNormalizer: wrote actual costs → %s", self._actual_costs_path)
        return out

    # -----------------------------------------------------------------------
    # Public: carbon
    # -----------------------------------------------------------------------

    def normalize_carbon(self, raw_carbon: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Normalises carbon data, ensures gco2_per_kwh flat key exists,
        and writes to data/carbon/carbon_intensity.json.
        """
        out: Dict[str, Dict] = {}

        for region, data in raw_carbon.items():
            # Ensure both key names present for cross-compatibility
            ci = data.get("carbon_intensity_gco2_per_kwh") or data.get("gco2_per_kwh", 400.0)
            out[region] = {
                **data,
                "carbon_intensity_gco2_per_kwh": round(float(ci), 2),
                "gco2_per_kwh":                  round(float(ci), 2),
            }

        self._atomic_write(self._carbon_path, out)

        live  = sum(1 for v in out.values() if "live"   in v.get("data_source", ""))
        stat  = sum(1 for v in out.values() if "static" in v.get("data_source", ""))
        logger.info(
            "DataNormalizer: wrote carbon → %s (%d live, %d static)",
            self._carbon_path, live, stat,
        )
        return out

    # -----------------------------------------------------------------------
    # Public: convenience reader
    # -----------------------------------------------------------------------

    def get_flat_carbon(self) -> Dict[str, float]:
        """
        Reads carbon JSON and returns {region: gco2_per_kwh} flat dict.
        Compatible with StateBuilder.build() carbon parameter.
        Falls back to static values if file not found.
        """
        try:
            with open(self._carbon_path) as fh:
                data = json.load(fh)
            return {
                region: float(entry.get("gco2_per_kwh", 400.0))
                for region, entry in data.items()
            }
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            from ai_engine.data_pipeline.carbon_api_client import STATIC_CARBON_INTENSITY
            logger.debug("DataNormalizer.get_flat_carbon: file not found, using static.")
            return {r: v["gco2_kwh"] for r, v in STATIC_CARBON_INTENSITY.items()}

    # -----------------------------------------------------------------------
    # Private: atomic write
    # -----------------------------------------------------------------------

    @staticmethod
    def _atomic_write(path: Path, data: Dict):
        """
        Writes JSON to a temp file then renames to final path.
        This ensures readers never see a partial file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.json")
        try:
            with open(tmp, "w") as fh:
                json.dump(data, fh, indent=2, default=str)
            tmp.replace(path)
        except Exception as exc:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise exc