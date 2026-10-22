"""API router assembly.

Each component router is imported defensively: a component whose service
layer is still an unimplemented stub must not prevent the whole API from
booting (previously a single missing service function crashed the app at
import time). Skipped components are recorded in ``DISABLED_COMPONENTS``
and surfaced by the root endpoint so the gap stays visible.
"""

import importlib
import logging

from fastapi import APIRouter

from app.api.v1 import health

logger = logging.getLogger(__name__)

api_router = APIRouter()
api_router.include_router(health.router, prefix="/health", tags=["Health"])

# module name -> (url prefix, tag)
COMPONENT_ROUTERS = {
    "component1": ("/component1", "Component 1: Behavioral Drift"),
    "component2": ("/component2", "Component 2: Digital Twin"),
    "component3": ("/component3", "Component 3: Causal XAI"),
    "component4": ("/component4", "Component 4: Safe RL"),
    "component4_decision": ("/component4", "Component 4: Safe Decisions"),
    "system": ("/system", "Integrated System"),
}

# module name -> import error message (empty dict = all routers healthy)
DISABLED_COMPONENTS: dict = {}


def _include(module_name: str, prefix: str, tag: str) -> None:
    try:
        module = importlib.import_module(f"app.api.v1.{module_name}")
    except Exception as exc:  # noqa: BLE001 - report and keep booting
        logger.warning("Router '%s' disabled: %s", module_name, exc)
        DISABLED_COMPONENTS[module_name] = str(exc)
        return
    api_router.include_router(module.router, prefix=prefix, tags=[tag])


for _module, (_prefix, _tag) in COMPONENT_ROUTERS.items():
    _include(_module, _prefix, _tag)
