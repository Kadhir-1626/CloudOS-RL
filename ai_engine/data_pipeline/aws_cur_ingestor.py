"""
AWS Cost & Usage Report (CUR) Ingestor
=======================================
Pulls ACTUAL billed costs from AWS Cost Explorer API.

IAM permissions required:
  - ce:GetCostAndUsage
  - ce:GetDimensionValues

Cost Explorer must be enabled in the AWS Console first:
  https://console.aws.amazon.com/cost-management/home

WITHOUT credentials or Cost Explorer: returns status=failed gracefully.
Data still written to aws_actual_costs.json with empty blended_rates.
This is expected behaviour for fresh AWS accounts.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
except ImportError:
    raise ImportError(
        "\n\nboto3 is not installed.\n"
        "Fix: run   pip install boto3   then try again.\n"
    )

logger = logging.getLogger(__name__)

_CE_REGION = "us-east-1"

_CE_LOCATION_TO_REGION: Dict[str, str] = {
    "US East (N. Virginia)":         "us-east-1",
    "US East (Ohio)":                "us-east-2",
    "US West (N. California)":       "us-west-1",
    "US West (Oregon)":              "us-west-2",
    "EU (Ireland)":                  "eu-west-1",
    "EU (London)":                   "eu-west-2",
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

_EC2_FILTER = {
    "Dimensions": {
        "Key":    "SERVICE",
        "Values": ["Amazon Elastic Compute Cloud - Compute"],
    }
}


class AWSCURIngestor:

    def __init__(self, config: Dict):
        self._anomaly_threshold = float(
            config.get("data_pipeline", {}).get("anomaly_threshold_pct", 50.0)
        )

    def fetch(self) -> Dict[str, Any]:
        logger.info("AWSCURIngestor: querying AWS Cost Explorer ...")
        t0 = time.perf_counter()

        try:
            rows_30d      = self._fetch_by_region_instance(days=30)
            totals_30d    = self._fetch_regional_totals(days=30)
            totals_7d     = self._fetch_regional_totals(days=7)

            blended       = self._compute_blended_rates(rows_30d)
            usage_summary = self._build_usage_summary(totals_30d, totals_7d)

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info("AWSCURIngestor: done — %d regions in %.0f ms", len(blended), elapsed)

            return {
                "blended_rates":   blended,
                "usage_summary":   usage_summary,
                "fetch_timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "period_days":     30,
                "status":          "ok",
            }

        except NoCredentialsError:
            logger.warning(
                "AWSCURIngestor: No AWS credentials. "
                "Run 'aws configure' to set them. Skipping CUR fetch."
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "OptInRequired":
                logger.warning(
                    "AWSCURIngestor: Cost Explorer not enabled on this account. "
                    "Enable it at https://console.aws.amazon.com/cost-management/home "
                    "(takes up to 24h to activate). Skipping CUR fetch."
                )
            elif code == "AccessDeniedException":
                logger.warning(
                    "AWSCURIngestor: Missing IAM permission ce:GetCostAndUsage. "
                    "Add it to your IAM policy. Skipping CUR fetch."
                )
            else:
                logger.error("AWSCURIngestor: ClientError [%s]: %s", code, exc)
        except BotoCoreError as exc:
            logger.error("AWSCURIngestor: BotoCoreError: %s", exc)
        except Exception as exc:
            logger.error("AWSCURIngestor: Unexpected error: %s", exc)

        return {
            "blended_rates":   {},
            "usage_summary":   {},
            "fetch_timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "period_days":     30,
            "status":          "failed",
        }

    def _client(self) -> Any:
        return boto3.client("ce", region_name=_CE_REGION)

    def _date_range(self, days: int) -> Tuple[str, str]:
        end   = datetime.now(tz=timezone.utc).date()
        start = end - timedelta(days=days)
        return start.isoformat(), end.isoformat()

    def _fetch_by_region_instance(self, days: int = 30) -> List[Dict]:
        start, end = self._date_range(days)
        resp = self._client().get_cost_and_usage(
            TimePeriod={"Start": start, "End": end},
            Granularity="MONTHLY",
            GroupBy=[
                {"Type": "DIMENSION", "Key": "REGION"},
                {"Type": "DIMENSION", "Key": "INSTANCE_TYPE"},
            ],
            Filter=_EC2_FILTER,
            Metrics=["BlendedCost", "UsageQuantity"],
        )
        rows: List[Dict] = []
        for period in resp.get("ResultsByTime", []):
            rows.extend(period.get("Groups", []))
        return rows

    def _fetch_regional_totals(self, days: int) -> Dict[str, float]:
        start, end = self._date_range(days)
        resp = self._client().get_cost_and_usage(
            TimePeriod={"Start": start, "End": end},
            Granularity="MONTHLY",
            GroupBy=[{"Type": "DIMENSION", "Key": "REGION"}],
            Filter=_EC2_FILTER,
            Metrics=["BlendedCost"],
        )
        totals: Dict[str, float] = {}
        for period in resp.get("ResultsByTime", []):
            for group in period.get("Groups", []):
                ce_region = group["Keys"][0]
                region    = _CE_LOCATION_TO_REGION.get(ce_region, ce_region)
                cost      = float(group["Metrics"]["BlendedCost"]["Amount"])
                totals[region] = totals.get(region, 0.0) + cost
        return totals

    def _compute_blended_rates(self, rows: List[Dict]) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for group in rows:
            keys = group.get("Keys", [])
            if len(keys) < 2:
                continue
            ce_region = keys[0]
            inst_type = keys[1]
            region    = _CE_LOCATION_TO_REGION.get(ce_region, ce_region)
            if inst_type in ("", "NoInstanceType"):
                continue
            metrics   = group.get("Metrics", {})
            cost      = float(metrics.get("BlendedCost",   {}).get("Amount", 0.0))
            usage_hrs = float(metrics.get("UsageQuantity", {}).get("Amount", 0.0))
            if cost <= 0.0 or usage_hrs <= 0.0:
                continue
            result.setdefault(region, {})[inst_type] = round(cost / usage_hrs, 6)
        return result

    def _build_usage_summary(
        self,
        totals_30d: Dict[str, float],
        totals_7d:  Dict[str, float],
    ) -> Dict[str, Dict]:
        summary: Dict[str, Dict] = {}
        for region in set(totals_30d) | set(totals_7d):
            cost_30  = totals_30d.get(region, 0.0)
            cost_7   = totals_7d.get(region, 0.0)
            daily_30 = cost_30 / 30.0
            daily_7  = cost_7  / 7.0
            spike    = (daily_7 - daily_30) / daily_30 * 100.0 if daily_30 > 0.0 else 0.0
            anomaly  = spike > self._anomaly_threshold
            if anomaly:
                logger.warning(
                    "COST ANOMALY [%s]: 7d daily=$%.2f vs 30d daily=$%.2f (+%.0f%%)",
                    region, daily_7, daily_30, spike,
                )
            summary[region] = {
                "total_cost_30d": round(cost_30,  4),
                "total_cost_7d":  round(cost_7,   4),
                "daily_avg_30d":  round(daily_30, 4),
                "daily_avg_7d":   round(daily_7,  4),
                "spike_pct":      round(spike, 2),
                "anomaly":        anomaly,
            }
        return summary