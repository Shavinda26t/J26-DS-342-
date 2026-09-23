from pydantic import BaseModel
from typing import List, Dict

class DriftAnalysisRequest(BaseModel):
    fleet_id: str

class DriftAnalysisResponse(BaseModel):
    fleet_id: str
    drift_score: float
    status: str
    summary_report: str
