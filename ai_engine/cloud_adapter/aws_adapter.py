import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class AWSAdapter:
    """Thin, stateless wrapper over boto3 for CloudOS-RL execution."""

    def __init__(self, config: Dict):
        self._region = config.get("aws", {}).get("region", "us-east-1")
        self._clients: Dict[str, object] = {}

    # ── client factory ──────────────────────────────────────────────────────

    def _client(self, service: str, region: Optional[str] = None) -> object:
        key = f"{service}:{region or self._region}"
        if key not in self._clients:
            self._clients[key] = boto3.client(service, region_name=region or self._region)
        return self._clients[key]

    # ── pricing ─────────────────────────────────────────────────────────────

    def get_spot_price(self, instance_type: str, region: Optional[str] = None) -> Optional[float]:
        try:
            ec2 = self._client("ec2", region)
            resp = ec2.describe_spot_price_history(
                InstanceTypes=[instance_type],
                ProductDescriptions=["Linux/UNIX"],
                MaxResults=1,
            )
            history = resp.get("SpotPriceHistory", [])
            return float(history[0]["SpotPrice"]) if history else None
        except ClientError as exc:
            logger.error("spot_price [%s/%s]: %s", region, instance_type, exc)
            return None

    # ── instance management ─────────────────────────────────────────────────

    def launch_spot(
        self,
        instance_type: str,
        ami_id: str,
        subnet_id: str,
        sg_ids: List[str],
        region: Optional[str] = None,
        user_data: str = "",
    ) -> Optional[str]:
        try:
            ec2 = self._client("ec2", region)
            resp = ec2.run_instances(
                ImageId=ami_id,
                InstanceType=instance_type,
                MinCount=1,
                MaxCount=1,
                SubnetId=subnet_id,
                SecurityGroupIds=sg_ids,
                UserData=user_data,
                InstanceMarketOptions={
                    "MarketType": "spot",
                    "SpotOptions": {
                        "SpotInstanceType": "one-time",
                        "InstanceInterruptionBehavior": "terminate",
                    },
                },
                TagSpecifications=[{
                    "ResourceType": "instance",
                    "Tags": [{"Key": "ManagedBy", "Value": "CloudOS-RL"}],
                }],
            )
            iid = resp["Instances"][0]["InstanceId"]
            logger.info("Launched spot %s in %s", iid, region or self._region)
            return iid
        except ClientError as exc:
            logger.error("launch_spot: %s", exc)
            return None

    def terminate(self, instance_id: str, region: Optional[str] = None) -> bool:
        try:
            self._client("ec2", region).terminate_instances(InstanceIds=[instance_id])
            return True
        except ClientError as exc:
            logger.error("terminate [%s]: %s", instance_id, exc)
            return False

    # ── monitoring ──────────────────────────────────────────────────────────

    def get_instance_metrics(
        self,
        instance_id: str,
        region: Optional[str] = None,
        lookback_minutes: int = 5,
    ) -> Dict[str, float]:
        cw    = self._client("cloudwatch", region)
        end   = datetime.now(tz=timezone.utc)
        start = end - timedelta(minutes=lookback_minutes)
        out   = {}

        for metric in ("CPUUtilization", "NetworkIn", "NetworkOut"):
            try:
                resp = cw.get_metric_statistics(
                    Namespace="AWS/EC2",
                    MetricName=metric,
                    Dimensions=[{"Name": "InstanceId", "Value": instance_id}],
                    StartTime=start,
                    EndTime=end,
                    Period=60,
                    Statistics=["Average"],
                )
                pts = resp.get("Datapoints", [])
                out[metric] = pts[-1]["Average"] if pts else 0.0
            except ClientError:
                out[metric] = 0.0

        return out