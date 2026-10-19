from pydantic import BaseModel
from typing import Dict

class DigitalTwinRequest(BaseModel):
    serial_number: str

class DigitalTwinResponse(BaseModel):
    serial_number: str
    health_index: float
    estimated_rul_days: float
    degradation_curve: Dict[str, float]
