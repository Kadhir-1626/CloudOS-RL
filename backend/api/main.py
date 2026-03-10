import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from backend.api.routes import metrics, scheduling
from backend.core.config import Settings
from backend.core.metrics_store import MetricsStore

logger   = logging.getLogger(__name__)
settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    app.state.metrics_store = MetricsStore()
    app.state.start_time    = time.time()
    logger.info("CloudOS-RL API started")
    yield
    logger.info("CloudOS-RL API shutting down")


app = FastAPI(
    title="CloudOS-RL",
    description="RL-driven multi-cloud workload scheduler",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/metrics", make_asgi_app())

app.include_router(scheduling.router, prefix="/api/v1/schedule",  tags=["scheduling"])
app.include_router(metrics.router,    prefix="/api/v1/metrics",   tags=["metrics"])


@app.get("/health", tags=["system"])
async def health():
    return {
        "status": "ok",
        "uptime_s": round(time.time() - app.state.start_time, 2),
    }