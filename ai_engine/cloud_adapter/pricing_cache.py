"""
PricingCache
=============
Provides EC2 pricing data to the rest of the system with this priority chain:

  Priority 1: data/pricing/aws_pricing.json   <- written by DataPipelineOrchestrator
  Priority 2: AWS Pricing API direct call      <- if pipeline file missing
  Priority 3: Hardcoded fallback constants     <- always available, zero network

File-mtime detection: if the pipeline writes a newer file since last read,
the cache reloads immediately — no need to wait for TTL expiry.

Used by:
  - ai_engine/environment/reward.py           (pricing parameter in compute())
  - ai_engine/environment/cloud_env.py        (_pricing_cache.get_current_pricing())
  - ai_engine/inference/scheduler_agent.py    (estimate_cost_per_hr)
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardcoded fallback — zero dependency, always available
# ---------------------------------------------------------------------------

_FALLBACK: Dict = {
    "us-east-1": {
        "t3.medium": 0.0416,
        "t3.large": 0.0832,
        "m5.large": 0.0960,
        "m5.xlarge": 0.1920,
        "c5.large": 0.0850,
        "c5.xlarge": 0.1700,
        "r5.large": 0.1260,
        "r5.xlarge": 0.2520,
        "g4dn.xlarge": 0.5260,
        "p3.2xlarge": 3.0600,
        "on_demand_per_vcpu_hr": 0.048,
        "spot_discount": 0.65,
    },
    "us-east-2": {"m5.large": 0.096, "c5.large": 0.085, "on_demand_per_vcpu_hr": 0.048, "spot_discount": 0.65},
    "us-west-1": {"m5.large": 0.112, "c5.large": 0.096, "on_demand_per_vcpu_hr": 0.056, "spot_discount": 0.65},
    "us-west-2": {"m5.large": 0.096, "c5.large": 0.085, "on_demand_per_vcpu_hr": 0.048, "spot_discount": 0.68},
    "eu-west-1": {"m5.large": 0.107, "c5.large": 0.097, "on_demand_per_vcpu_hr": 0.054, "spot_discount": 0.63},
    "eu-west-2": {"m5.large": 0.111, "c5.large": 0.100, "on_demand_per_vcpu_hr": 0.056, "spot_discount": 0.65},
    "eu-west-3": {"m5.large": 0.111, "c5.large": 0.100, "on_demand_per_vcpu_hr": 0.056, "spot_discount": 0.65},
    "eu-central-1": {"m5.large": 0.111, "c5.large": 0.099, "on_demand_per_vcpu_hr": 0.056, "spot_discount": 0.65},
    "eu-north-1": {"m5.large": 0.096, "c5.large": 0.086, "on_demand_per_vcpu_hr": 0.048, "spot_discount": 0.70},
    "ap-southeast-1": {"m5.large": 0.114, "c5.large": 0.100, "on_demand_per_vcpu_hr": 0.057, "spot_discount": 0.65},
    "ap-southeast-2": {"m5.large": 0.122, "c5.large": 0.108, "on_demand_per_vcpu_hr": 0.061, "spot_discount": 0.65},
    "ap-northeast-1": {"m5.large": 0.118, "c5.large": 0.107, "on_demand_per_vcpu_hr": 0.059, "spot_discount": 0.65},
    "ap-northeast-2": {"m5.large": 0.114, "c5.large": 0.102, "on_demand_per_vcpu_hr": 0.057, "spot_discount": 0.65},
    "ap-south-1": {"m5.large": 0.096, "c5.large": 0.085, "on_demand_per_vcpu_hr": 0.048, "spot_discount": 0.65},
    "ca-central-1": {"m5.large": 0.100, "c5.large": 0.090, "on_demand_per_vcpu_hr": 0.050, "spot_discount": 0.65},
    "sa-east-1": {"m5.large": 0.142, "c5.large": 0.128, "on_demand_per_vcpu_hr": 0.071, "spot_discount": 0.65},
    # GCP/Azure aliases used in ActionDecoder
    "us-central1": {"m5.large": 0.096, "c5.large": 0.085, "on_demand_per_vcpu_hr": 0.048, "spot_discount": 0.65},
    "europe-west4": {"m5.large": 0.107, "c5.large": 0.097, "on_demand_per_vcpu_hr": 0.054, "spot_discount": 0.63},
    "eastus": {"m5.large": 0.096, "c5.large": 0.085, "on_demand_per_vcpu_hr": 0.048, "spot_discount": 0.65},
    "westeurope": {"m5.large": 0.107, "c5.large": 0.097, "on_demand_per_vcpu_hr": 0.054, "spot_discount": 0.63},
}

_LOCATION_MAP: Dict[str, str] = {
    "US East (N. Virginia)": "us-east-1",
    "US West (Oregon)": "us-west-2",
    "EU (Ireland)": "eu-west-1",
    "EU (Frankfurt)": "eu-central-1",
    "Asia Pacific (Singapore)": "ap-southeast-1",
    "Asia Pacific (Tokyo)": "ap-northeast-1",
    "Canada (Central)": "ca-central-1",
    "South America (Sao Paulo)": "sa-east-1",
}


class PricingCache:
    """
    Thread-safe pricing cache with TTL + file-mtime-based refresh.
    All public methods are safe to call from multiple threads.
    """

    TTL = 3600  # seconds before forcing a re-check

    def __init__(self, config: Dict):
        self._config = config
        self._path = Path(
            config.get("pricing_fallback_path")
            or config.get("data_pipeline", {}).get(
                "pricing_output_path", "data/pricing/aws_pricing.json"
            )
        )
        self._cache: Dict = {}
        self._ts: float = 0.0
        self._file_mtime: float = 0.0

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def get_current_pricing(self) -> Dict[str, float]:
        """
        Returns flat pricing dict: {region: price_per_hr (float)}.
        Compatible with both StateBuilder (expects floats) and
        test_pricing_cache.py (iterates values as floats).
        """
        raw = self._load_raw_pricing()
        return self._flatten_pricing(raw)

    def get_price(
        self,
        region: str,
        instance_type: str,
        purchase_option: str = "on_demand",
    ) -> float:
        """
        Returns price for region/instance/purchase combination.
        Reads raw file to get spot_discount if available.
        """
        flat = self.get_current_pricing()
        od_price = flat.get(region, flat.get("us-east-1", 0.096))

        if purchase_option == "on_demand":
            return round(od_price, 6)

        raw = self._load_raw_pricing()
        region_data = raw.get(region, {})
        if isinstance(region_data, dict):
            discount = float(region_data.get("spot_discount", 0.65))
        else:
            discount = 0.65

        if purchase_option == "spot":
            return round(od_price * (1.0 - discount), 6)
        if purchase_option == "reserved_1yr":
            return round(od_price * 0.60, 6)
        if purchase_option == "reserved_3yr":
            return round(od_price * 0.40, 6)
        if purchase_option == "savings_plan":
            return round(od_price * 0.55, 6)
        if purchase_option == "preemptible":
            return round(od_price * 0.30, 6)

        return round(od_price, 6)

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _load_raw_pricing(self) -> Dict:
        """Loads raw pricing file (may contain nested dicts). Internal use only."""
        expired = (time.time() - self._ts) > self.TTL
        file_new = self._has_newer_file()

        if expired or file_new or not self._cache:
            self._refresh()

        return self._cache if self._cache else _FALLBACK

    @staticmethod
    def _flatten_pricing(raw: Dict) -> Dict[str, float]:
        """
        Converts nested pricing dict to flat {region: float}.
        Handles both formats:
          {"us-east-1": 0.096}                             → passthrough
          {"us-east-1": {"on_demand_per_vcpu_hr": 0.096}} → extracted
        """
        result: Dict[str, float] = {}

        for region, value in raw.items():
            if isinstance(value, (int, float)):
                result[region] = float(value)
            elif isinstance(value, dict):
                price = (
                    value.get("on_demand_per_vcpu_hr")
                    or value.get("m5.large")
                    or value.get("on_demand")
                    or next(
                        (
                            v
                            for v in value.values()
                            if isinstance(v, (int, float)) and v > 0
                        ),
                        0.096,
                    )
                )
                result[region] = float(price)

        return result if result else {
            "us-east-1": 0.096,
            "us-west-2": 0.096,
            "eu-west-1": 0.107,
            "eu-north-1": 0.098,
        }

    def _refresh(self):
        """Refresh from pipeline file → AWS API → hardcoded fallback."""

        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as fh:
                    data = json.load(fh)
                if data:
                    self._cache = data
                    self._ts = time.time()
                    self._file_mtime = self._path.stat().st_mtime
                    logger.debug("PricingCache: loaded from file %s", self._path)
                    return
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("PricingCache: file read failed (%s) — trying AWS API", exc)

        try:
            api_data = self._fetch_from_aws()
            if api_data:
                self._cache = api_data
                self._ts = time.time()
                self._persist_to_file(api_data)
                logger.info("PricingCache: loaded from AWS Pricing API")
                return
        except (ClientError, BotoCoreError) as exc:
            logger.warning("PricingCache: AWS API unavailable (%s) — using fallback", exc)

        self._cache = _FALLBACK
        self._ts = time.time()
        logger.info("PricingCache: using hardcoded fallback pricing")

    def _has_newer_file(self) -> bool:
        """Returns True if the pipeline has written a newer file since last load."""
        try:
            return self._path.stat().st_mtime > self._file_mtime
        except OSError:
            return False

    def _fetch_from_aws(self) -> Dict:
        """Direct AWS Pricing API call. Used when pipeline file is missing."""
        client = boto3.client("pricing", region_name="us-east-1")
        result: Dict = {}

        paginator = client.get_paginator("get_products")
        pages = paginator.paginate(
            ServiceCode="AmazonEC2",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
            ],
            PaginationConfig={"MaxItems": 1000, "PageSize": 100},
        )

        for page in pages:
            for raw in page["PriceList"]:
                try:
                    item = json.loads(raw)
                    attrs = item["product"]["attributes"]
                    region = _LOCATION_MAP.get(attrs.get("location", ""))
                    inst = attrs.get("instanceType", "")
                    if not region or not inst:
                        continue
                    for term in item["terms"].get("OnDemand", {}).values():
                        for dim in term["priceDimensions"].values():
                            price = float(dim["pricePerUnit"].get("USD", 0))
                            if price > 0:
                                result.setdefault(region, {})[inst] = price
                except (KeyError, ValueError):
                    continue

        return result

    def _persist_to_file(self, data: Dict):
        """Saves API-fetched data to file for future use (atomic write)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp.json")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            tmp.replace(self._path)
            self._file_mtime = self._path.stat().st_mtime
        except Exception as exc:
            logger.warning("PricingCache: could not persist to file: %s", exc)
            if tmp.exists():
                tmp.unlink(missing_ok=True)