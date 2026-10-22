"""Component 1 research shadow API. No warning or healing dispatch."""
import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from app.services import backblaze_replay_service as replay
from app.services.component1_service import get_service, IntakeConflict, analyze_fleet_drift

router = APIRouter()
logger = logging.getLogger(__name__)


def execute(operation):
    try:
        service = get_service()
    except Exception as exc:
        logger.exception("Frozen scorer initialization failed")
        raise HTTPException(503, "Frozen scorer unavailable; inspect artifact verification logs") from exc
    try:
        return operation(service)
    except IntakeConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (OSError, RuntimeError, KeyError) as exc:
        logger.exception("Shadow scorer unavailable")
        raise HTTPException(503, "Shadow scorer or durable state unavailable; inspect backend logs") from exc


@router.post("/shadow/records")
def ingest_shadow_record(record: dict, replay_id: str | None = None):
    return execute(lambda service: replay.ingest(service, replay_id, record) if replay_id else service.ingest(record))


@router.get("/shadow")
def shadow_states():
    return execute(lambda service: {"mode": "shadow", "drives": service.list_states(),
                                   "supported_models": service.scorer.models,
                                   "agent_schema_verified": False})


@router.get("/shadow/history")
def shadow_history(serial_number: str):
    history = execute(lambda service: service.get_history(serial_number))
    if history is None:
        raise HTTPException(404, "Live shadow device not found")
    return history


@router.post("/analyze")
def analyze_drift():
    return analyze_fleet_drift()


@router.post("/replay/prepare")
async def prepare_replay(request: Request):
    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > replay.MAX_CSV_BYTES:
            raise HTTPException(413, "Only upload the extracted 30-row CSV (64 KiB maximum); process the ZIP locally")
        content.extend(chunk)
    return execute(lambda service: replay.prepare(service, bytes(content)))


@router.get("/replay/{replay_id}")
def replay_status(replay_id: str):
    return execute(lambda service: replay.status(service, replay_id))


@router.get("/replay/{replay_id}/csv")
def download_replay(replay_id: str):
    content = execute(lambda service: replay.csv_export(service, replay_id))
    return Response(content, media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="backblaze_window.csv"'})
