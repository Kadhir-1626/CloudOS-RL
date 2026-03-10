from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class WorkloadType(str, Enum):
    BATCH       = "batch"
    REALTIME    = "realtime"
    ML_TRAINING = "ml_training"
    WEB_SERVICE = "web_service"


class WorkloadRequest(BaseModel):
    workload_id:              str
    cpu_request:              float = Field(...,   ge=0.1,    le=512.0)
    memory_request_gb:        float = Field(...,   ge=0.5,    le=4096.0)
    gpu_count:                int   = Field(0,     ge=0,      le=16)
    storage_gb:               float = Field(100.0, ge=1.0,    le=50_000.0)
    network_bandwidth_gbps:   float = Field(1.0,   ge=0.1,    le=100.0)
    expected_duration_hours:  float = Field(...,   ge=0.01,   le=8760.0)
    priority:                 int   = Field(2,     ge=1,      le=4)
    sla_latency_ms:           float = Field(200.0, ge=1.0,    le=5000.0)
    workload_type:            WorkloadType = WorkloadType.BATCH
    is_spot_tolerant:         bool  = False
    excluded_regions:         List[str] = Field(default_factory=list)

    @field_validator("cpu_request")
    @classmethod
    def cpu_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("cpu_request must be > 0")
        return v

    model_config = {"use_enum_values": True}


class SchedulingDecision(BaseModel):
    decision_id:         str
    workload_id:         str
    cloud:               str
    region:              str
    instance_type:       str
    scaling_level:       int
    purchase_option:     str
    sla_tier:            int
    estimated_cost_per_hr: float
    cost_savings_pct:    float
    carbon_savings_pct:  float
    latency_ms:          float
    explanation:         Dict[str, Any]