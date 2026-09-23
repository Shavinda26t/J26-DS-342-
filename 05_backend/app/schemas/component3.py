from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ExplainRequest(BaseModel):
    serial_number: str
    hdd_model: Optional[str] = None

class ExplainResponse(BaseModel):
    serial_number: str
    prediction_risk: float
    global_explanation: Dict[str, float]
    local_shap_values: Dict[str, float]
    important_smart_attributes: List[str]
    causal_relationships: List[Dict[str, Any]]
    causal_effect_estimates: Dict[str, float]
    candidate_root_causes: List[str]
    confidence_score: float
    smart_trend_info: Dict[str, List[float]]

class RootCauseRequest(BaseModel):
    serial_number: str

class RootCauseResponse(BaseModel):
    serial_number: str
    primary_root_cause: str
    causal_chain: List[str]
    evidence_score: float
