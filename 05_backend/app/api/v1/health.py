from fastapi import APIRouter

router = APIRouter()

@router.get("")
def get_health():
    return {"status": "online", "system": "HDD Failure Prediction Framework"}
