from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List, Dict, Any
from app.schemas.component3 import ExplainRequest, ExplainResponse, RootCauseRequest, RootCauseResponse
from app.services import component3_service

router = APIRouter()

@router.get("/health")
def get_health():
    """Check Component 3 service health and data availability."""
    return component3_service.get_health_status()

@router.get("/summary")
def get_summary():
    """Return dashboard summary metrics from Component 3 outputs."""
    return component3_service.get_summary()

@router.get("/dashboard")
def get_dashboard():
    """Return complete dashboard payload."""
    return component3_service.get_dashboard()

@router.get("/factors")
def get_factors():
    """Return factor evidence catalog of all canonical SMART/workload factors."""
    return component3_service.get_factors()

@router.get("/factors/{factor_id}")
def get_factor_by_id(factor_id: str):
    """Return detailed evidence for a specific factor (case-insensitive ID)."""
    return component3_service.get_factor_by_id(factor_id)

@router.get("/root-causes")
def get_root_causes():
    """Return integrated root-cause candidate factors."""
    return component3_service.get_root_causes()

@router.get("/temporal-pathways")
def get_temporal_pathways():
    """Return degradation temporal pathways (PCMCI stability relationships)."""
    return component3_service.get_temporal_pathways()

@router.get("/matched-effects")
def get_matched_effects():
    """Return matched 30-day failure risk evidence (ATT analysis)."""
    return component3_service.get_matched_effects()

@router.get("/cases")
def get_cases():
    """Return available representative local explanation cases (TP, FP, FN, TN)."""
    return component3_service.get_cases()

@router.get("/cases/{case_type}")
def get_case_by_type(case_type: str):
    """Return representative local case payload for TP, FP, FN, or TN (case-insensitive)."""
    return component3_service.get_case_by_type(case_type)

@router.get("/figures")
def get_figures():
    """Return metadata for available whitelisted Component 3 research figures."""
    return component3_service.get_figures_metadata()

@router.get("/figures/{figure_key}")
def get_figure_by_key(figure_key: str):
    """Serve whitelisted research figure image (PNG). Security: Only whitelisted figure keys allowed."""
    file_path = component3_service.get_figure_filepath(figure_key)
    return FileResponse(path=file_path, media_type="image/png", filename=file_path.name)

# Legacy POST endpoints for backward compatibility
@router.post("/explain", response_model=ExplainResponse)
def explain_hdd_failure(request: ExplainRequest):
    return component3_service.generate_explanation(request)

@router.post("/root-cause", response_model=RootCauseResponse)
def analyze_root_cause(request: RootCauseRequest):
    return component3_service.evaluate_root_cause(request)
