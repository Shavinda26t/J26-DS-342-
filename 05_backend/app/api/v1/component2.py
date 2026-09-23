from fastapi import APIRouter
from app.schemas.component2 import DigitalTwinRequest, DigitalTwinResponse
from app.services.component2_service import process_digital_twin

router = APIRouter()

@router.post("/twin", response_model=DigitalTwinResponse)
def get_digital_twin(request: DigitalTwinRequest):
    return process_digital_twin(request)
