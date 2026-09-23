from fastapi import APIRouter
from app.api.v1 import health, component1, component2, component3, component4, system

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(component1.router, prefix="/component1", tags=["Component 1: Behavioral Drift"])
api_router.include_router(component2.router, prefix="/component2", tags=["Component 2: Digital Twin"])
api_router.include_router(component3.router, prefix="/component3", tags=["Component 3: Causal XAI"])
api_router.include_router(component4.router, prefix="/component4", tags=["Component 4: Safe RL"])
api_router.include_router(system.router, prefix="/system", tags=["Integrated System"])
