from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.get("/aggregate")
async def aggregate(
    req: Request,
    window_hours: float = Query(default=1.0, ge=0.1, le=168.0),
):
    return req.app.state.metrics_store.aggregate(window_hours)


@router.get("/recent")
async def recent(
    req: Request,
    n: int = Query(default=50, ge=1, le=500),
):
    return {"decisions": req.app.state.metrics_store.recent(n)}