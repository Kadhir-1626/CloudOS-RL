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
from typing import Dict, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardcoded fallback — zero dependency, always available
# ---------------------------------------------------------------------------

_FALLBACK: Dict = {
    "us-east-1": {
        "t3.medium":   0.0416, "t3.large":   0.0832,
        "m5.large":    0.0960, "m5.xlarge":  0.1920,
        "c5.large":    0.0850, "c5.xlarge":  0.1700,
        "r5.large":    0.1260, "r5.xlarge":  0.2520,
        "g4dn.xlarge": 0.5260, "p3.2xlarge": 3.0600,
        "on_demand_per_vcpu_hr": 0.048,
    },
    "us-east-2":      {"m5.large": 0.096, "c5.large": 0.085, "on_demand_per_vcpu_hr": 0.048},
    "us-west-1":      {"m5.large": 0.112, "c5.large": 0.096, "on_demand_per_vcpu_hr": 0.056},
    "us-west-2":      {"m5.large": 0.096, "c5.large": 0.085, "on_demand_per_vcpu_hr": 0.048},
    "eu-west-1":      {"m5.large": 0.107, "c5.large": 0.097, "on_demand_per_vcpu_hr": 0.054},
    "eu-west-2":      {"m5.large": 0.111, "c5.large": 0.100, "on_demand_per_vcpu_hr": 0.056},
    "eu-west-3":      {"m5.large": 0.111, "c5.large": 0.100, "on_demand_per_vcpu_hr": 0.056},
    "eu-central-1":   {"m5.large": 0.111, "c5.large": 0.099, "on_demand_per_vcpu_hr": 0.056},
    "eu-north-1":     {"m5.large": 0.096, "c5.large": 0.086, "on_demand_per_vcpu_hr": 0.048},
    "ap-southeast-1": {"m5.large": 0.114, "c5.large": 0.100, "on_demand_per_vcpu_hr": 0.057},
    "ap-southeast-2": {"m5.large": 0.122, "c5.large": 0.108, "on_demand_per_vcpu_hr": 0.061},
    "ap-northeast-1": {"m5.large": 0.118, "c5.large": 0.107, "on_demand_per_vcpu_hr": 0.059},
    "ap-northeast-2": {"m5.large": 0.114, "c5.large": 0.102, "on_demand_per_vcpu_hr": 0.057},
    "ap-south-1":     {"m5.large": 0.096, "c5.large": 0.085, "on_demand_per_vcpu_hr": 0.048},
    "ca-central-1":   {"m5.large": 0.100, "c5.large": 0.090, "on_demand_per_vcpu_hr": 0.050},
    "sa-east-1":      {"m5.large": 0.142, "c5.large": 0.128, "on_demand_per_vcpu_hr": 0.071},
    # GCP/Azure aliases used in ActionDecoder
    "us-central1":    {"m5.large": 0.096, "c5.large": 0.085, "on_demand_per_vcpu_hr": 0.048},
    "europe-west4":   {"m5.large": 0.107, "c5.large": 0.097, "on_demand_per_vcpu_hr": 0.054},
    "eastus":         {"m5.large": 0.096, "c5.large": 0.085, "on_demand_per_vcpu_hr": 0.048},
    "westeurope":     {"m5.large": 0.107, "c5.large": 0.097, "on_demand_per_vcpu_hr": 0.054},
}

_LOCATION_MAP: Dict[str, str] = {
    "US East (N. Virginia)":        "us-east-1",
    "US West (Oregon)":             "us-west-2",
    "EU (Ireland)":                 "eu-west-1",
    "EU (Frankfurt)":               "eu-central-1",
    "Asia Pacific (Singapore)":     "ap-southeast-1",
    "Asia Pacific (Tokyo)":         "ap-northeast-1",
    "Canada (Central)":             "ca-central-1",
    "South America (Sao Paulo)":    "sa-east-1",
}


class PricingCache:
    """
    Thread-safe pricing cache with TTL + file-mtime-based refresh.
    All public methods are safe to call from multiple threads.
    """

    TTL = 3600   # seconds before forcing a re-check

    def __init__(self, config: Dict):
        self._path           = Path(config.get("pricing_fallback_path", "data/pricing/aws_pricing.json"))
        self._cache:         Dict  = {}
        self._ts:            float = 0.0
        self._file_mtime:    float = 0.0

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def get_current_pricing(self) -> Dict:
        """
        Returns pricing dict.
        Refreshes automatically if TTL expired or pipeline wrote a newer file.
        """
        expired  = (time.time() - self._ts) > self.TTL
        file_new = self._has_newer_file()

        if expired or file_new or not self._cache:
            self._refresh()

        return self._cache if self._cache else _FALLBACK

    def get_price(
        self,
        region:        str,
        instance_type: str,
        purchase:      str = "on_demand",
    ) -> float:
        """
        Returns $/hr for a specific region/instance/purchase combination.
        Lookup chain: nested → colon-key → flat → fallback.
        """
        pricing     = self.get_current_pricing()
        region_data = pricing.get(region, pricing.get("us-east-1", {}))

        # Try _nested (set by DataNormalizer)
        nested = region_data.get("_nested", {})
        if instance_type in nested and purchase in nested[instance_type]:
            return float(nested[instance_type][purchase])

        # Try colon-key format (set by AWSPricingFetcher + DataNormalizer)
        colon_key = f"{instance_type}:{purchase}"
        if colon_key in region_data:
            return float(region_data[colon_key])

        # Try flat key (on-demand only)
        if instance_type in region_data:
            return float(region_data[instance_type])

        # Absolute last resort
        return _FALLBACK.get("us-east-1", {}).get(instance_type, 0.096)

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _refresh(self):
        """Refresh from pipeline file → AWS API → hardcoded fallback."""

        # Priority 1: Pipeline-written file
        if self._path.exists():
            try:
                with open(self._path) as fh:
                    data = json.load(fh)
                if data:
                    self._cache       = data
                    self._ts          = time.time()
                    self._file_mtime  = self._path.stat().st_mtime
                    logger.debug("PricingCache: loaded from file %s", self._path)
                    return
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("PricingCache: file read failed (%s) — trying AWS API", exc)

        # Priority 2: AWS Pricing API direct call
        try:
            api_data = self._fetch_from_aws()
            if api_data:
                self._cache = api_data
                self._ts    = time.time()
                self._persist_to_file(api_data)
                logger.info("PricingCache: loaded from AWS Pricing API")
                return
        except (ClientError, BotoCoreError) as exc:
            logger.warning("PricingCache: AWS API unavailable (%s) — using fallback", exc)

        # Priority 3: Hardcoded fallback
        self._cache = _FALLBACK
        self._ts    = time.time()
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
                {"Type": "TERM_MATCH", "Field": "tenancy",          "Value": "Shared"},
                {"Type": "TERM_MATCH", "Field": "preInstalledSw",   "Value": "NA"},
                {"Type": "TERM_MATCH", "Field": "capacitystatus",   "Value": "Used"},
            ],
            PaginationConfig={"MaxItems": 1000, "PageSize": 100},
        )

        for page in pages:
            for raw in page["PriceList"]:
                try:
                    item   = json.loads(raw)
                    attrs  = item["product"]["attributes"]
                    region = _LOCATION_MAP.get(attrs.get("location", ""))
                    inst   = attrs.get("instanceType", "")
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
            with open(tmp, "w") as fh:
                json.dump(data, fh, indent=2)
            tmp.replace(self._path)
            self._file_mtime = self._path.stat().st_mtime
        except Exception as exc:
            logger.warning("PricingCache: could not persist to file: %s", exc)
            if tmp.exists():
                tmp.unlink(missing_ok=True)