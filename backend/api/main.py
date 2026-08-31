"""
CloudOS-RL FastAPI Application
================================
Entry point for the scheduling API.
"""

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.scheduling import router as scheduling_router
from backend.auth.router import router as auth_router
from backend.core.agent_singleton import get_agent, is_ready, startup_initialise

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-40s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

_CPU_COUNT = os.cpu_count() or 4

# Thread pool sizing rationale:
#   Previous value: cpu_count × 2
#   New value: cpu_count × 3 gives slightly more thread capacity for this mixed
#   workload while keeping a firm cap to avoid runaway oversubscription.
#   Ceiling of 24 prevents excessive thread growth on larger machines.
_POOL_SIZE = min(32, max(8, _CPU_COUNT * 3))

_INFERENCE_POOL = ThreadPoolExecutor(
    max_workers=_POOL_SIZE,
    thread_name_prefix="cloudos-inf",
)

logger.info(
    "CloudOS-RL: thread pool sized to %d workers (%d CPUs detected)",
    _POOL_SIZE,
    _CPU_COUNT,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
      - install the default thread pool executor
      - initialise the RL agent / Kafka producer

    Shutdown:
      - gracefully stop the executor
    """
    loop = asyncio.get_running_loop()
    loop.set_default_executor(_INFERENCE_POOL)
    logger.info(
        "CloudOS-RL API: thread pool configured (max_workers=%d)",
        _POOL_SIZE,
    )

    logger.info("CloudOS-RL API starting — loading RL agent ...")
    startup_initialise()

    # Wait for agent to be ready before accepting traffic (max 30s)
    import time as _time
    from backend.core.agent_singleton import is_ready
    _deadline = _time.monotonic() + 30.0
    while not is_ready() and _time.monotonic() < _deadline:
        await asyncio.sleep(0.5)
    if is_ready():
        logger.info("CloudOS-RL API: agent ready, accepting traffic")
    else:
        logger.warning("CloudOS-RL API: agent not ready after 30s, accepting anyway")

    try:
        yield
    finally:
        logger.info("CloudOS-RL API shutting down.")
        _INFERENCE_POOL.shutdown(wait=False)


app = FastAPI(
    title="CloudOS-RL Scheduling API",
    description="AI-native multi-cloud workload scheduler — PPO + SHAP + Kafka.",
    version="1.0.0",
    lifespan=lifespan,
)

import os

ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://velox-multicloud-scheduler.vercel.app",
    "https://velox-multicloud-scheduler-*.vercel.app",
]

# Also allow any origin in development
if os.environ.get("CLOUDOS_AUTH_REQUIRED", "false").lower() == "false":
    ALLOWED_ORIGINS.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Routes
app.include_router(scheduling_router)
app.include_router(auth_router)


# Health
@app.get("/health", tags=["health"])
async def health():
    agent = get_agent()
    return {
        "status": "ok",
        "agent_loaded": agent is not None,
        "shap_ready": agent._explainer is not None if agent else False,
        "ready": is_ready(),
    }


@app.get("/", tags=["health"])
async def root():
    return {"service": "cloudos-rl-api", "version": "1.0.0"}