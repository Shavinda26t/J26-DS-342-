from fastapi import APIRouter
from app.schemas.component1 import DriftAnalysisRequest, DriftAnalysisResponse
from app.services.component1_service import analyze_fleet_drift

router = APIRouter()

@router.post("/analyze", response_model=DriftAnalysisResponse)
def analyze_drift(request: DriftAnalysisRequest):
    return analyze_fleet_drift(request)
