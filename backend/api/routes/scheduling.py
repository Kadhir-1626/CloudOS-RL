import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from prometheus_client import Counter, Histogram

from ai_engine.inference.scheduler_agent import SchedulerAgent
from backend.api.models.schemas import SchedulingDecision, WorkloadRequest

logger    = logging.getLogger(__name__)
router    = APIRouter()

_REQUESTS  = Counter("cloudos_schedule_requests_total",   "Total scheduling requests")
_LATENCY   = Histogram("cloudos_schedule_latency_seconds", "Scheduling latency")
_COST_SAVE = Histogram("cloudos_cost_savings_ratio",       "Cost savings fraction", buckets=[0, .05, .1, .2, .3, .5, 1])

_agent: Optional[SchedulerAgent] = None


def _get_agent() -> SchedulerAgent:
    global _agent
    if _agent is None:
        _agent = SchedulerAgent.load()
    return _agent


@router.post("/", response_model=SchedulingDecision)
async def schedule(
    payload: WorkloadRequest,
    bg: BackgroundTasks,
    req: Request,
):
    _REQUESTS.inc()
    t0 = time.perf_counter()

    try:
        agent    = _get_agent()
        state    = agent.build_state(payload.model_dump())
        decision, explanation = agent.decide(state)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        _LATENCY.observe(latency_ms / 1000.0)

        result = SchedulingDecision(
            decision_id         = str(uuid.uuid4()),
            workload_id         = payload.workload_id,
            cloud               = decision["cloud"],
            region              = decision["region"],
            instance_type       = decision["instance_type"],
            scaling_level       = decision["scaling_level"],
            purchase_option     = decision["purchase_option"],
            sla_tier            = decision["sla_tier"],
            estimated_cost_per_hr = agent.estimate_cost_per_hr(decision),
            cost_savings_pct    = agent.cost_savings_pct(decision),
            carbon_savings_pct  = agent.carbon_savings_pct(decision),
            latency_ms          = latency_ms,
            explanation         = explanation,
        )

        _COST_SAVE.observe(result.cost_savings_pct / 100.0)
        bg.add_task(req.app.state.metrics_store.record_decision, result.model_dump())
        return result

    except Exception as exc:
        logger.exception("Scheduling failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))