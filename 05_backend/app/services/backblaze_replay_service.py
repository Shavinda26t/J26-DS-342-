"""Small validated replay manifests; rolling state lives in the existing ShadowStore."""
import csv
import io
import json
import os
from uuid import uuid4, UUID
from app.services.component1_service import ShadowStore, IntakeConflict
from tools.backblaze_replay import COLUMNS, FEATURES, validate_rows

MAX_CSV_BYTES = 65536


def parse_csv(content, supported):
    if len(content) > MAX_CSV_BYTES:
        raise ValueError("Upload only the extracted 30-row CSV (maximum 64 KiB), not the quarterly ZIP")
    try:
        reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig"), newline=""))
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("CSV must contain exactly: " + ", ".join(COLUMNS))
        rows = []
        for row in reader:
            if len(rows) >= 30 or None in row or any(v is None for v in row.values()):
                raise ValueError("Expected exactly 30 complete records")
            rows.append(row)
        validate_rows(rows)
        if rows[0]["model"] not in supported:
            raise ValueError("Unsupported model")
        return [{**{k: r[k] for k in COLUMNS[:3]}, **{k: float(r[k]) for k in FEATURES}} for r in rows]
    except (UnicodeError, csv.Error, OverflowError) as exc:
        raise ValueError("Invalid CSV encoding or raw SMART data") from exc


def directory(service):
    path = service.path.parent / (service.path.stem + "_replays")
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare(service, content):
    rows = parse_csv(content, service.scorer.models)
    token = uuid4().hex
    manifest = {"replay_id": token, "replay_identity": "REPLAY_" + token,
                "source_serial": rows[0]["serial_number"], "model": rows[0]["model"],
                "start_date": rows[0]["date"], "end_date": rows[-1]["date"],
                "scorer_version": service.scorer.version, "rows": rows,
                "purpose": "Historical compatibility test only; no training or threshold tuning"}
    path = directory(service) / (token + ".manifest.json")
    tmp = path.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf8") as handle:
            json.dump(manifest, handle, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return manifest


def load(service, token):
    try:
        if UUID(token).hex != token:
            raise ValueError()
    except (ValueError, AttributeError):
        raise ValueError("Invalid replay ID")
    path = directory(service) / (token + ".manifest.json")
    if not path.exists():
        raise ValueError("Replay session not found; prepare the CSV again")
    manifest = json.loads(path.read_text(encoding="utf8"))
    if manifest["scorer_version"] != service.scorer.version:
        raise RuntimeError("Replay scorer version mismatch")
    store = ShadowStore(directory(service) / (token + ".state.json"), service.scorer)
    return manifest, store


def status(service, token):
    manifest, store = load(service, token)
    states = store.list_states()
    result = states[0] if states else None
    return {**manifest, "completed_days": result["history_days"] if result else 0, "result": result}


def ingest(service, token, record):
    manifest, store = load(service, token)
    if record not in manifest["rows"]:
        raise ValueError("Record does not match the validated replay window")
    # The outer file lock and ShadowStore's reentrant lock make ordering + write atomic.
    with store.lock:
        states = store.list_states()
        count = states[0]["history_days"] if states else 0
        index = manifest["rows"].index(record)
        if index not in (count, count - 1):
            raise IntakeConflict("Replay must be chronological; resume from completed_days or prepare a new replay")
        return store.ingest({**record, "serial_number": manifest["replay_identity"]})


def csv_export(service, token):
    manifest, _ = load(service, token)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(manifest["rows"])
    return stream.getvalue()
