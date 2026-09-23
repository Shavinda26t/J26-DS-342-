from fastapi import APIRouter
from app.schemas.component3 import ExplainRequest, ExplainResponse, RootCauseRequest, RootCauseResponse
from app.services.component3_service import generate_explanation, evaluate_root_cause

router = APIRouter()

@router.post("/explain", response_model=ExplainResponse)
def explain_hdd_failure(request: ExplainRequest):
    return generate_explanation(request)

@router.post("/root-cause", response_model=RootCauseResponse)
def analyze_root_cause(request: RootCauseRequest):
    return evaluate_root_cause(request)
