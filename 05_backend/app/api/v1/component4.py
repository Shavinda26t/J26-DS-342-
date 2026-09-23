from fastapi import APIRouter
from app.schemas.component4 import OptimizationRequest, OptimizationResponse
from app.services.component4_service import optimize_storage

router = APIRouter()

@router.post("/optimize", response_model=OptimizationResponse)
def optimize(request: OptimizationRequest):
    return optimize_storage(request)
