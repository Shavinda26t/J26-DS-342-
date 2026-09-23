from fastapi import APIRouter
from app.services.integration_service import run_integrated_analysis

router = APIRouter()

@router.post("/analyze")
def full_system_analysis(drive_id: str):
    return run_integrated_analysis(drive_id)
