"""
AWS Pricing Fetcher
===================
Fetches real-time EC2 pricing from two AWS sources:
  1. AWS Pricing API (us-east-1 global endpoint)  → on-demand rates
  2. EC2 API per region                            → live spot prices

IAM permissions required:
  - pricing:GetProducts
  - ec2:DescribeSpotPriceHistory

WITHOUT valid AWS credentials → returns empty dict. PricingCache uses fallback.
WITH    valid AWS credentials → returns pricing for all available regions.

Output schema compatible with:
  - ai_engine/cloud_adapter/pricing_cache.py   (flat key read)
  - ai_engine/environment/state_builder.py     (on_demand_per_vcpu_hr)
  - ai_engine/environment/reward.py            (instance on_demand price)
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

# ---------------------------------------------------------------------------
# boto3 import with clear error message
# ---------------------------------------------------------------------------
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
except ImportError:
    raise ImportError(
        "\n\nboto3 is not installed.\n"
        "Fix: run   pip install boto3   then try again.\n"
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRACKED_INSTANCES: List[str] = [
    "t3.medium", "t3.large",
    "m5.large",  "m5.xlarge",
    "c5.large",  "c5.xlarge",
    "r5.large",  "r5.xlarge",
    "g4dn.xlarge",
    "p3.2xlarge",
]

LOCATION_TO_REGION: Dict[str, str] = {
    "US East (N. Virginia)":         "us-east-1",
    "US East (Ohio)":                "us-east-2",
    "US West (N. California)":       "us-west-1",
    "US West (Oregon)":              "us-west-2",
    "EU (Ireland)":                  "eu-west-1",
    "EU (London)":                   "eu-west-2",
    "EU (Paris)":                    "eu-west-3",
    "EU (Frankfurt)":                "eu-central-1",
    "EU (Stockholm)":                "eu-north-1",
    "Asia Pacific (Singapore)":      "ap-southeast-1",
    "Asia Pacific (Sydney)":         "ap-southeast-2",
    "Asia Pacific (Tokyo)":          "ap-northeast-1",
    "Asia Pacific (Seoul)":          "ap-northeast-2",
    "Asia Pacific (Mumbai)":         "ap-south-1",
    "Canada (Central)":              "ca-central-1",
    "South America (Sao Paulo)":     "sa-east-1",
}

PURCHASE_DISCOUNT: Dict[str, float] = {
    "on_demand":    1.000,
    "spot":         0.330,
    "preemptible":  0.300,
    "savings_plan": 0.550,
    "reserved_1yr": 0.600,
    "reserved_3yr": 0.400,
}

INSTANCE_VCPU: Dict[str, int] = {
    "t3.medium":   2,
    "t3.large":    2,
    "m5.large":    2,
    "m5.xlarge":   4,
    "c5.large":    2,
    "c5.xlarge":   4,
    "r5.large":    2,
    "r5.xlarge":   4,
    "g4dn.xlarge": 4,
    "p3.2xlarge":  8,
}


class AWSPricingFetcher:
    """
    Fetches and merges on-demand + spot EC2 pricing.
    Returns empty dict (not an error) if AWS is unreachable.
    PricingCache handles the empty-dict case by falling back to constants.
    """

    _PRICING_REGION = "us-east-1"

    def __init__(self, config: Dict):
        self._default_region = config.get("aws", {}).get("region", "us-east-1")

    def fetch(self) -> Dict[str, Dict]:
        logger.info("AWSPricingFetcher: fetching on-demand pricing ...")
        t0 = time.perf_counter()

        on_demand = self._fetch_on_demand()

        if not on_demand:
            logger.warning(
                "AWSPricingFetcher: on-demand pricing returned empty. "
                "Check AWS credentials (run: aws sts get-caller-identity) "
                "and IAM permission: pricing:GetProducts"
            )
            return {}

        spot_data = self._fetch_spot_parallel(list(on_demand.keys()))

        result: Dict[str, Dict] = {}
        for region, instances in on_demand.items():
            result[region] = self._build_region_entry(
                region, instances, spot_data.get(region, {})
            )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("AWSPricingFetcher: done — %d regions in %.0f ms", len(result), elapsed)
        return result

    # -----------------------------------------------------------------------
    # Private: on-demand
    # -----------------------------------------------------------------------

    def _fetch_on_demand(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        try:
            client    = boto3.client("pricing", region_name=self._PRICING_REGION)
            paginator = client.get_paginator("get_products")
            pages     = paginator.paginate(
                ServiceCode="AmazonEC2",
                Filters=[
                    {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                    {"Type": "TERM_MATCH", "Field": "tenancy",          "Value": "Shared"},
                    {"Type": "TERM_MATCH", "Field": "preInstalledSw",   "Value": "NA"},
                    {"Type": "TERM_MATCH", "Field": "capacitystatus",   "Value": "Used"},
                ],
                PaginationConfig={"MaxItems": 5000, "PageSize": 100},
            )
            for page in pages:
                for raw in page["PriceList"]:
                    self._parse_item(raw, result)

        except NoCredentialsError:
            logger.error(
                "AWSPricingFetcher: No AWS credentials found. "
                "Run 'aws configure' to set them up."
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            logger.error("AWSPricingFetcher: AWS ClientError [%s]: %s", code, exc)
        except BotoCoreError as exc:
            logger.error("AWSPricingFetcher: BotoCoreError: %s", exc)

        return result

    @staticmethod
    def _parse_item(raw: str, out: Dict[str, Dict[str, float]]):
        try:
            item   = json.loads(raw)
            attrs  = item["product"]["attributes"]
            region = LOCATION_TO_REGION.get(attrs.get("location", ""))
            inst   = attrs.get("instanceType", "")
            if not region or inst not in TRACKED_INSTANCES:
                return
            for term in item["terms"].get("OnDemand", {}).values():
                for dim in term["priceDimensions"].values():
                    usd = float(dim["pricePerUnit"].get("USD", 0.0))
                    if usd > 0.0:
                        out.setdefault(region, {})[inst] = usd
                        return
        except (KeyError, ValueError, json.JSONDecodeError):
            pass

    # -----------------------------------------------------------------------
    # Private: spot
    # -----------------------------------------------------------------------

    def _fetch_spot_parallel(
        self, regions: List[str], max_workers: int = 6
    ) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}

        def _one(region: str):
            return region, self._fetch_spot_region(region)

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="spot") as pool:
            futures = {pool.submit(_one, r): r for r in regions}
            for fut in as_completed(futures):
                try:
                    region, data = fut.result(timeout=30)
                    result[region] = data
                except Exception as exc:
                    logger.warning("Spot fetch [%s]: %s", futures[fut], exc)

        return result

    def _fetch_spot_region(self, region: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        try:
            ec2  = boto3.client("ec2", region_name=region)
            resp = ec2.describe_spot_price_history(
                InstanceTypes=TRACKED_INSTANCES,
                ProductDescriptions=["Linux/UNIX"],
                MaxResults=len(TRACKED_INSTANCES) * 3,
            )
            seen = set()
            for entry in resp.get("SpotPriceHistory", []):
                inst = entry["InstanceType"]
                if inst not in seen:
                    out[inst] = float(entry["SpotPrice"])
                    seen.add(inst)
        except (ClientError, BotoCoreError, NoCredentialsError) as exc:
            logger.debug("Spot unavailable [%s]: %s", region, exc)
        return out

    # -----------------------------------------------------------------------
    # Private: build output
    # -----------------------------------------------------------------------

    def _build_region_entry(
        self,
        region:    str,
        instances: Dict[str, float],
        spot_data: Dict[str, float],
    ) -> Dict:
        entry:       Dict       = {}
        nested:      Dict       = {}
        vcpu_rates: List[float] = []

        for inst, od_price in instances.items():
            if inst not in TRACKED_INSTANCES:
                continue

            spot = spot_data.get(inst, od_price * PURCHASE_DISCOUNT["spot"])

            prices = {
                "on_demand":    round(od_price, 6),
                "spot":         round(spot, 6),
                "savings_plan": round(od_price * PURCHASE_DISCOUNT["savings_plan"], 6),
                "reserved_1yr": round(od_price * PURCHASE_DISCOUNT["reserved_1yr"], 6),
                "reserved_3yr": round(od_price * PURCHASE_DISCOUNT["reserved_3yr"], 6),
            }

            entry[inst] = prices["on_demand"]
            for purchase, price in prices.items():
                entry[f"{inst}:{purchase}"] = price

            nested[inst] = prices

            vcpus = INSTANCE_VCPU.get(inst, 2)
            vcpu_rates.append(od_price / vcpus)

        if vcpu_rates:
            entry["on_demand_per_vcpu_hr"] = round(
                sum(vcpu_rates) / len(vcpu_rates), 6
            )

        entry["_nested"] = nested
        entry["_region"] = region
        return entry