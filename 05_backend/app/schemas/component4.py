from pydantic import BaseModel
from typing import List

class OptimizationRequest(BaseModel):
    drive_id: str
    current_state: List[float]

class OptimizationResponse(BaseModel):
    drive_id: str
    recommended_action: str
    safety_constraint_violated: bool
