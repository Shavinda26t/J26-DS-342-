"""Durable shadow-only daily intake; isolated from healing/notification paths."""
import json
import math
import os
import re
import threading
from filelock import FileLock
from datetime import date, timedelta
from pathlib import Path
from app.ml.frozen_hdd import FEATURES, FrozenHDD


class IntakeConflict(ValueError):
    pass


class ShadowStore:
    def __init__(self, path=None, scorer=None):
        self.path = Path(path or os.environ.get("HDD_SHADOW_STATE", Path(__file__).resolve().parents[1] / "runtime/hdd_shadow.json"))
        self.scorer = scorer or FrozenHDD()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path) + ".lock", timeout=10)

    def _read(self):
        if not self.path.exists():
            return {}
        try:
            state = json.loads(self.path.read_text(encoding="utf8"))
        except (ValueError, OSError) as exc:
            raise RuntimeError("Persisted shadow state cannot be read") from exc
        if not isinstance(state, dict) or not isinstance(state.get("drives"), dict):
            raise RuntimeError("Invalid persisted shadow state structure")
        if state.get("version") != self.scorer.version:
            raise RuntimeError("Persisted scorer version mismatch; explicit migration required")
        return state["drives"]

    def _write(self, drives):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf8") as handle:
                json.dump({"version": self.scorer.version, "drives": drives}, handle, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        finally:
            tmp.unlink(missing_ok=True)

    def list_states(self):
        with self.lock:
            return [v["result"] for v in self._read().values()]

    def get_history(self, serial_number):
        """Return the accepted raw daily history for one live shadow drive.

        State stores raw values in consecutive order and the latest accepted
        calendar day in ``result.end_date``. Dates are therefore reconstructed
        from that enforced invariant without inventing observations.
        """
        with self.lock:
            record = self._read().get(serial_number)
            if record is None:
                return None
            result = record["result"]
            history = record["history"]
            end_day = date.fromisoformat(result["end_date"])
            start_day = end_day - timedelta(days=max(0, len(history) - 1))
            rows = []
            for offset, values in enumerate(history):
                rows.append({
                    "date": str(start_day + timedelta(days=offset)),
                    **dict(zip(FEATURES, values)),
                })
            return {
                "serial_number": serial_number,
                "model": result["model"],
                "start_date": rows[0]["date"] if rows else None,
                "end_date": result["end_date"],
                "history_days": len(rows),
                "rows": rows,
                "result": result,
            }

    def ingest(self, row):
        # Contract intake only: never guess a live agent schema or convert timestamps.
        serial, model, stamp = (row.get(k) for k in ("serial_number", "model", "date"))
        if not isinstance(serial, str) or not serial.strip():
            raise ValueError("serial_number must be a nonempty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if not isinstance(stamp, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", stamp):
            raise ValueError("date must be an explicit calendar day YYYY-MM-DD; timezone conversion is not inferred")
        day = date.fromisoformat(stamp)
        base = {"serial_number": serial, "model": model, "end_date": stamp,
                "mode": "shadow", "scorer_version": self.scorer.version,
                "regime_id": None, "drift_event": None,
                "anomaly_score": None, "frozen_threshold": None, "candidate_flag": None,
                "agent_schema_verified": False}
        with self.lock:
            drives = self._read()
            prior = drives.get(serial)
            if prior and prior["result"]["model"] != model:
                raise IntakeConflict("model changed for serial")
            if model not in self.scorer.models:
                result = {**base, "status": "unsupported_model", "history_days": 0}
                drives[serial] = {"history": [], "result": result}
                self._write(drives)
                return result
            values = []
            for field in FEATURES:
                value = row.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or value > 3.4028234663852886e38 or not math.isfinite(value):
                    raise ValueError(f"{field} requires a finite nonnegative observed raw number; no imputation")
                values.append(value)
            history = prior["history"] if prior else []
            gap = False
            if prior:
                previous_day = date.fromisoformat(prior["result"]["end_date"])
                if day < previous_day:
                    raise IntakeConflict("out-of-order record; backfill is not accepted")
                if day == previous_day:
                    # Compare original raw numbers, not rounded float32 measurements.
                    if values != history[-1]:
                        raise IntakeConflict("conflicting duplicate day")
                    return {**prior["result"], "duplicate": True}
                gap = (day - previous_day).days != 1
                if gap:
                    history = []
            history = (history + [values])[-30:]
            result = {**base, "status": "warming", "history_days": len(history),
                      "duplicate": False, "reset_reason": "gap" if gap else None}
            if len(history) == 30:
                result.update(self.scorer.score(history, model))
                result["status"] = "scored"
            drives[serial] = {"history": history, "result": result}
            self._write(drives)
            return result


_service = None
_service_lock = threading.Lock()


def get_service():
    global _service
    with _service_lock:
        if _service is None:
            _service = ShadowStore()
        return _service


def analyze_fleet_drift(request=None):
    return {"status": "unavailable", "drift_score": None, "regime_id": None,
            "summary_report": "Drift and regime detection are not validated."}
