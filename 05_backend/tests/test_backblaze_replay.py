"""Streaming ZIP extraction -> isolated real frozen scorer, with no labels exported."""
import csv
import io
import json
import subprocess
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api.v1 import component1
from app.ml.frozen_hdd import FrozenHDD
from app.services.component1_service import ShadowStore
from app.services import backblaze_replay_service as replay
from tools.backblaze_replay import COLUMNS, source_entries, choose_serial, collect, last_window, validate_window


@pytest.fixture(scope="module")
def scorer():
    return FrozenHDD()


def archive(path, kind="valid"):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for n in range(30):
            if kind == "gap" and n == 12:
                continue
            day = str(date(2022, 10, 1) + timedelta(days=n))
            stream = io.StringIO()
            writer = csv.writer(stream)
            writer.writerow([*COLUMNS, "failure", "smart_5_normalized"] if kind != "columns" else ["date"])
            values = ["SOURCE-DRIVE", "ST4000DM000" if kind != "unsupported" else "SSD", day, 0, 1000 + n*24, "" if kind == "missing" and n == 12 else 28, 0, 0, 1, 100]
            writer.writerow(values)
            if kind == "duplicate" and n == 12:
                writer.writerow(values)
            z.writestr(f"daily/{day}.csv", stream.getvalue())


@pytest.fixture
def extracted(tmp_path):
    path = tmp_path / "synthetic.zip"
    archive(path)
    output = tmp_path / "window.csv"
    script = Path(__file__).resolve().parents[1] / "tools/backblaze_replay.py"
    result = subprocess.run([sys.executable, str(script), "extract", str(path), "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "failure" not in output.read_text()
    assert len(validate_window(output)) == 30
    return output


@pytest.mark.parametrize("kind", ["gap", "columns", "missing", "duplicate", "unsupported"])
def test_reject_bad_archive(tmp_path, kind):
    path = tmp_path / "bad.zip"
    archive(path, kind)
    with pytest.raises(ValueError):
        choose_serial(source_entries([path]), {"ST4000DM000"}, 30, None)


def test_duplicate_daily_member(tmp_path):
    path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("a/2022-10-01.csv", "")
        z.writestr("b/2022-10-01.csv", "")
    with pytest.raises(ValueError, match="Duplicate daily"):
        source_entries([path])


def test_api_replay_restart_isolation_and_repeat(extracted, tmp_path, scorer):
    service = ShadowStore(tmp_path / "live.json", scorer)
    app = FastAPI()
    app.include_router(component1.router, prefix="/api/v1/component1")
    prefix = "/api/v1/component1"
    with patch.object(component1, "get_service", return_value=service), TestClient(app) as client:
        response = client.post(prefix + "/replay/prepare", content=extracted.read_bytes())
        assert response.status_code == 200
        manifest = response.json()
        token = manifest["replay_id"]
        assert len(manifest["rows"]) == 30
        for n, row in enumerate(manifest["rows"]):
            response = client.post(prefix + "/shadow/records", params={"replay_id": token}, json=row)
            assert response.status_code == 200, response.text
            assert response.json()["history_days"] == n + 1
            assert response.json()["status"] == ("warming" if n < 29 else "scored")
            if n == 14:
                service = ShadowStore(tmp_path / "live.json", scorer)
                # Load a new store from disk; no cached rolling state required.
                assert replay.status(service, token)["completed_days"] == 15
        result = response.json()
        assert result["serial_number"] == manifest["replay_identity"]
        assert result["model_version"] == scorer.checkpoint_hash
        assert result["candidate_flag"] == (result["anomaly_score"] > result["frozen_threshold"])
        assert client.get(prefix + "/shadow").json()["drives"] == []
        assert client.get(prefix + f"/replay/{token}").json()["result"] == result
        assert client.post(prefix + "/shadow/records", params={"replay_id": token}, json=manifest["rows"][-1]).json()["duplicate"]
        assert client.post(prefix + "/shadow/records", params={"replay_id": token}, json=manifest["rows"][0]).status_code == 409
        changed = {**manifest["rows"][-1], "smart_5_raw": 999}
        assert client.post(prefix + "/shadow/records", params={"replay_id": token}, json=changed).status_code == 422
        fresh = client.post(prefix + "/replay/prepare", content=extracted.read_bytes()).json()
        assert fresh["replay_id"] != token
        assert client.get(prefix + f"/replay/{fresh['replay_id']}").json()["completed_days"] == 0
        downloaded = client.get(prefix + f"/replay/{token}/csv")
        assert downloaded.status_code == 200 and "failure" not in downloaded.text
        assert len(list(csv.DictReader(io.StringIO(downloaded.text)))) == 30
        assert client.post(prefix + "/replay/prepare", content=b"x" * 65537).status_code == 413
        assert client.post(prefix + "/replay/prepare", content=extracted.read_bytes()[:80]).status_code == 422
        assert client.get(prefix + "/replay/not-a-uuid").status_code == 422


@pytest.mark.parametrize("change", ["nan", "missing", "unsupported", "duplicate", "negative", "normalized", "empty"])
def test_preview_validation(extracted, scorer, change):
    rows = list(csv.DictReader(io.StringIO(extracted.read_text())))
    if change == "nan": rows[4]["smart_9_raw"] = "NaN"
    if change == "missing": rows[4]["smart_194_raw"] = ""
    if change == "unsupported":
        for row in rows: row["model"] = "SSD"
    if change == "duplicate": rows[5]["date"] = rows[4]["date"]
    if change == "negative": rows[5]["smart_5_raw"] = "-1"
    if change == "empty":
        for row in rows: row["serial_number"] = ""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=COLUMNS)
    writer.writeheader(); writer.writerows(rows)
    content = stream.getvalue().replace("smart_5_raw", "smart_5_normalized") if change == "normalized" else stream.getvalue()
    with pytest.raises(ValueError):
        replay.parse_csv(content.encode(), scorer.models)
