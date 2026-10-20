"""Component 2: Adaptive Digital Twin endpoints.

All metrics are served directly from the component's held-out validation
artifacts (see ``app.services.component2_service``), so the dashboard
always matches ``02_component_2_digital_twin`` results.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, UploadFile, File
import io
import logging
import pandas as pd

from app.schemas.component2 import (
    BatchInferenceResponse,
    ConfigComparisonRow,
    DetectionRecord,
    DigitalTwinRequest,
    DigitalTwinResponse,
    LeadTimeBand,
    Overview,
    RulMetrics,
    SweepConfig,
    TwinLearnRequest,
    TwinResetResponse,
    TwinSyncRequest,
)
from app.services import component2_service as service
from app.services import component2_inference_service as inference_service
from app.services import twin_sync_service as sync_service

logger = logging.getLogger(__name__)

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


@router.post("/inference/batch", response_model=BatchInferenceResponse, tags=["Component 2: Digital Twin"])
async def inference_batch(file: UploadFile = File(...)) -> BatchInferenceResponse:
    """Run batch inference on uploaded CSV using trained Digital Twin model."""
    if not file.filename or not file.filename.lower().endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported"
        )
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        results = inference_service.predict_batch(df)
        return {"count": len(results), "predictions": results}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")


# --- Live Digital Twin: synchronization + enhanced features ---


@router.post("/twin/sync", tags=["Component 2: Digital Twin"])
def sync_twin(request: TwinSyncRequest) -> Dict[str, Any]:
    """Ingest one observation and return the full live twin state.

    Runs adaptive thresholds, RUL + online adaptation, conformal confidence,
    data-quality and the multi-horizon forecast in one call.

    A malformed payload is rejected with **422** *before* the twin is mutated
    (the observation lifecycle is validate -> dedup -> compute -> persist), so a
    rejected request never leaves a partially updated twin.  If the resulting
    state cannot be durably persisted the request fails with **503** and the
    in-memory twin is rolled back: a sync is never reported as successful unless
    it was written to disk.
    """
    try:
        return sync_service.sync_observation(
            request.serial_number, request.observation
        )
    except sync_service.TwinPersistenceError as e:
        # Technical cause is logged in the service; the client only sees a
        # stable, non-sensitive message (no paths, no traceback).
        logger.error("twin sync persistence failure: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Twin state could not be persisted",
        )
    except sync_service.ObservationValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Twin sync failed: {str(e)}")


@router.get("/twin/{serial_number}/state", tags=["Component 2: Digital Twin"])
def get_twin_state(serial_number: str) -> Dict[str, Any]:
    """Read the current live twin state without ingesting data."""
    state = sync_service.get_state(serial_number)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"No live twin state for serial '{serial_number}'.",
        )
    return state


@router.post("/twin/{serial_number}/learn", tags=["Component 2: Digital Twin"])
def learn_twin_outcome(serial_number: str, request: TwinLearnRequest) -> Dict[str, Any]:
    """Feed a realised RUL back so the twin adapts online (RLS + conformal)."""
    try:
        return sync_service.learn_outcome(
            serial_number, request.actual_rul_days
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Online learning update failed: {str(e)}"
        )


@router.delete("/twin/{serial_number}", tags=["Component 2: Digital Twin"])
def reset_twin(serial_number: str) -> TwinResetResponse:
    """Clear a live twin's state and history."""
    return TwinResetResponse(
        serial_number=serial_number,
        reset=sync_service.reset_serial(serial_number),
    )


@router.get("/twin/serials", tags=["Component 2: Digital Twin"])
def list_twin_serials() -> Dict[str, Any]:
    """List serials that currently have live twin state."""
    serials = sync_service.list_serials()
    return {"count": len(serials), "serials": serials}


@router.get("/twin/calibration", tags=["Component 2: Digital Twin"])
def get_twin_calibration() -> Dict[str, Any]:
    """Conformal calibration + online-adapter learning diagnostics."""
    return sync_service.calibration_snapshot()
