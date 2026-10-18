"""Component 2: Adaptive Digital Twin endpoints.

All metrics are served directly from the component's held-out validation
artifacts (see ``app.services.component2_service``), so the dashboard
always matches ``02_component_2_digital_twin`` results.
"""

from typing import List

from fastapi import APIRouter, HTTPException

from app.schemas.component2 import (
    ConfigComparisonRow,
    DetectionRecord,
    DigitalTwinRequest,
    DigitalTwinResponse,
    LeadTimeBand,
    Overview,
    RulMetrics,
    SweepConfig,
)
from app.services import component2_service as service

router = APIRouter()


@router.get("/overview", response_model=Overview, tags=["Component 2: Digital Twin"])
def get_overview() -> Overview:
    """Headline held-out metrics + deployed detector configuration."""
    return service.get_overview()


@router.get(
    "/detections",
    response_model=List[DetectionRecord],
    tags=["Component 2: Digital Twin"],
)
def get_detections() -> List[DetectionRecord]:
    """Per-HDD lead-time results for every failed HDD in the test split."""
    return service.get_detections()


@router.get(
    "/lead-time-distribution",
    response_model=List[LeadTimeBand],
    tags=["Component 2: Digital Twin"],
)
def get_lead_time_distribution() -> List[LeadTimeBand]:
    """Distribution of detection lead times across bands."""
    return service.get_lead_time_distribution()


@router.get("/sweep", response_model=List[SweepConfig], tags=["Component 2: Digital Twin"])
def get_sweep() -> List[SweepConfig]:
    """Threshold-sweep trade-off curve (tuning set: train+validation)."""
    return service.get_sweep()


@router.get(
    "/config-comparison",
    response_model=List[ConfigComparisonRow],
    tags=["Component 2: Digital Twin"],
)
def get_config_comparison() -> List[ConfigComparisonRow]:
    """Tuning-selected vs actually-deployed config, both on the held-out test."""
    return service.get_config_comparison()


@router.get("/rul-metrics", response_model=RulMetrics, tags=["Component 2: Digital Twin"])
def get_rul_metrics() -> RulMetrics:
    """RUL prediction accuracy by adaptive state and by checkpoint."""
    return service.get_rul_metrics()


@router.post("/twin", response_model=DigitalTwinResponse, tags=["Component 2: Digital Twin"])
def get_digital_twin(request: DigitalTwinRequest) -> DigitalTwinResponse:
    """Per-HDD digital twin detail for one serial number."""
    detail = service.get_twin_detail(request.serial_number)
    if detail is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Serial '{request.serial_number}' not found in the held-out "
                "digital twin results."
            ),
        )
    return detail
