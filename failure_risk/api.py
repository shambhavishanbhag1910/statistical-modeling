from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from failure_risk.config import DEFAULT_MODEL_PATH
from failure_risk.inference import FailureRiskService

REQUEST_COUNT = Counter("failure_risk_api_requests_total", "API requests", ["path", "method", "status"])
REQUEST_LATENCY = Histogram("failure_risk_api_request_seconds", "API request latency", ["path"])

service: FailureRiskService | None = None


class PredictionRequest(BaseModel):
    current_age_months: float = Field(gt=0, le=120)
    monthly_km: float = Field(ge=0, le=20_000)
    engine_hours_per_month: float = Field(ge=0, le=500)
    load_factor: float = Field(ge=0, le=1)
    temperature_exposure: float = Field(ge=0, le=1)
    prior_repairs: int = Field(ge=0, le=50)
    service_delay_days: float = Field(ge=0, le=365)
    route_severity: float = Field(ge=0, le=1)
    preventive_maintenance_score: float = Field(ge=0, le=1)
    vehicle_class: Literal["Light", "Medium", "Heavy"]


class PredictionResponse(BaseModel):
    risk_30d: float
    risk_60d: float
    risk_90d: float
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    if model_path.exists():
        service = FailureRiskService(model_path)
    yield


app = FastAPI(
    title="Vehicle Component Failure Risk API",
    version="1.0.0",
    description="Conditional 30/60/90-day failure risk from survival models.",
    lifespan=lifespan,
)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    REQUEST_COUNT.labels(request.url.path, request.method, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(time.perf_counter() - start)
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "healthy" if service else "model_not_loaded"}


@app.get("/model-info")
def model_info() -> dict:
    if service is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded. Run training first.")
    return service.model_info()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if service is None:
        raise HTTPException(status_code=503, detail="Model artifact is not loaded. Run training first.")
    pred = service.predict_one(request.model_dump())
    return PredictionResponse(**pred.__dict__)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
