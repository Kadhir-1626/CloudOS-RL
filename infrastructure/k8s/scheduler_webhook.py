"""
Lightweight admission webhook / operator that intercepts CloudOSWorkload
creation events, calls the RL API, and patches the status subresource.
"""
import json
import logging
import os
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request, Response

logger = logging.getLogger(__name__)
app    = FastAPI(title="CloudOS Scheduler Webhook")

API_URL = os.getenv("CLOUDOS_API_URL", "http://cloudos-api:8000")


@app.post("/schedule")
async def webhook(request: Request) -> Response:
    body = await request.json()
    meta = body.get("metadata", {})
    spec = body.get("spec",     {})

    payload = {
        "workload_id":              meta.get("name", "unknown"),
        "cpu_request":              spec.get("cpuRequest",            1.0),
        "memory_request_gb":        spec.get("memoryRequestGb",       2.0),
        "gpu_count":                spec.get("gpuCount",              0),
        "storage_gb":               spec.get("storageGb",             100.0),
        "network_bandwidth_gbps":   spec.get("networkBandwidthGbps",  1.0),
        "expected_duration_hours":  spec.get("expectedDurationHours", 1.0),
        "priority":                 spec.get("priority",              2),
        "sla_latency_ms":           spec.get("slaLatencyMs",          200.0),
        "workload_type":            spec.get("workloadType",          "batch"),
        "is_spot_tolerant":         spec.get("isSpotTolerant",        False),
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.post(f"{API_URL}/api/v1/schedule/", json=payload)
            resp.raise_for_status()
            d = resp.json()

            patch = {"status": {
                "phase":              "Scheduled",
                "decisionId":         d["decision_id"],
                "assignedCloud":      d["cloud"],
                "assignedRegion":     d["region"],
                "instanceType":       d["instance_type"],
                "estimatedCostPerHr": d["estimated_cost_per_hr"],
                "costSavingsPct":     d["cost_savings_pct"],
                "carbonSavingsPct":   d["carbon_savings_pct"],
                "scheduledAt":        datetime.now(tz=timezone.utc).isoformat(),
            }}
        except Exception as exc:
            logger.error("Webhook scheduling error: %s", exc)
            patch = {"status": {"phase": "Failed"}}

    return Response(
        content=json.dumps(patch),
        media_type="application/json",
        status_code=200,
    )