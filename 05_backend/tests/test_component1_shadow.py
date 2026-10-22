"""Frozen notebook parity and durable shadow API contract tests (no Q4 labels)."""
import json
from collections import deque
from datetime import date, timedelta
from unittest.mock import patch
import re
import numpy as np
import pandas as pd
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.ml.frozen_hdd import FrozenHDD, FEATURES, HIDDEN, ROOT, digest
from app.services.component1_service import ShadowStore, IntakeConflict
from app.api.v1 import component1


@pytest.fixture(scope="module")
def scorer():
    torch.set_num_threads(1)
    return FrozenHDD()


def row(n=0, model="ST4000DM000", **updates):
    return {"serial_number": "synthetic-contract-check", "model": model,
            "date": (date(2021, 9, 1) + timedelta(days=n)).isoformat(),
            **dict(zip(FEATURES, [0, 1000 + n * 24, 28, 0, 0])), **updates}


@pytest.fixture
def store(tmp_path, scorer):
    return ShadowStore(tmp_path / "state.json", scorer)


def test_warmup_restart_day30_gap_duplicate(store, scorer):
    for n in range(29):
        result = store.ingest(row(n))
        assert result["status"] == "warming"
        assert result["history_days"] == n + 1
    restarted = ShadowStore(store.path, scorer)
    result = restarted.ingest(row(29))
    assert result["status"] == "scored"
    assert result["candidate_flag"] == (result["anomaly_score"] > result["frozen_threshold"])
    assert restarted.ingest(row(29)) == {**result, "duplicate": True}
    assert store.list_states() == [result]
    with pytest.raises(IntakeConflict, match="conflicting"):
        store.ingest(row(29, smart_5_raw=1))
    with pytest.raises(IntakeConflict, match="out-of-order"):
        store.ingest(row(10))
    result = store.ingest(row(31))
    assert result["history_days"] == 1 and result["reset_reason"] == "gap"
    assert result["anomaly_score"] is None


@pytest.mark.parametrize("value", [None, -1, float("nan"), float("inf"), True, "0", 1e100])
@pytest.mark.parametrize("field", FEATURES)
def test_invalid_raw_does_not_mutate(store, field, value):
    store.ingest(row())
    before = store.path.read_bytes()
    with pytest.raises(ValueError):
        store.ingest(row(1, **{field: value}))
    assert store.path.read_bytes() == before


def test_missing_normalized_and_timestamp_not_substituted(store):
    record = row()
    del record["smart_5_raw"]
    record["smart_5_normalized"] = 100
    with pytest.raises(ValueError, match="smart_5_raw"):
        store.ingest(record)
    with pytest.raises(ValueError, match="calendar day"):
        store.ingest(row(date="2021-09-01T23:59:00Z"))
    assert store.list_states() == []


def test_unsupported_and_model_change(store):
    assert store.ingest(row(model="SSD"))["status"] == "unsupported_model"
    assert store.list_states()[0]["anomaly_score"] is None
    with pytest.raises(IntakeConflict, match="model changed"):
        store.ingest(row(1))


def test_failed_persistence_and_corruption(store):
    store.ingest(row())
    before = store.path.read_bytes()
    with patch("app.services.component1_service.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            store.ingest(row(1))
    assert store.path.read_bytes() == before
    store.path.write_text("broken")
    with pytest.raises(RuntimeError, match="cannot be read"):
        store.list_states()


def test_hash_tampering(tmp_path):
    # Verify the lock before any checkpoint loading.
    folder = tmp_path / "results/single_model_q1q2_2021_v1"
    (folder / "q4_research_review_and_handoff_v1").mkdir(parents=True)
    (folder / "q4_blind_candidate_protocol_v1").mkdir()
    (folder / "q4_research_review_and_handoff_v1/research_integration_handoff.json").write_text(json.dumps({"candidate_lock_sha256": "bad"}))
    (folder / "q4_blind_candidate_protocol_v1/q4_candidate_lock.json").write_text("{}")
    with pytest.raises(RuntimeError, match="lock hash"):
        FrozenHDD(tmp_path)


def test_notebook24_numerical_parity_all_models(store, scorer):
    # Execute the original notebook definitions independently, without its data-reading cells.
    notebook = json.loads(next((ROOT / "notebooks").glob("HDD_24_*")).read_text(encoding="utf8"))
    code = "".join(notebook["cells"][3]["source"])
    env = {"nn": torch.nn, "torch": torch, "np": np, "pd": pd, "re": re, "deque": deque,
           "models": scorer.models, "prep": scorer.prep, "FEATURES": FEATURES, "HIDDEN": HIDDEN,
           "model_index": {m: i for i, m in enumerate(scorer.models)}, "DEVICE": torch.device("cpu"),
           "thresholds": scorer.thresholds, "sha256": digest}
    exec(code[:code.index("network=MaskedTelemetryTransformer")], env)
    network = env["MaskedTelemetryTransformer"](5)
    network.load_state_dict(scorer.network.state_dict())
    network.eval()
    env["network"] = network
    env["checkpoint_path"] = ROOT / "models/trained/single_model_q1q2_2021_v1/q3_five_model_masked_transformer_v1/best_model.pt"
    exec(code[code.index("@torch.no_grad()"):code.index("print('Frozen daily")], env)
    reference = env["FrozenDailyScorer"]()
    for model in scorer.models:
        for n in range(35):
            record = row(n, model=model, serial_number=model, smart_5_raw=max(0, n-27), smart_194_raw=28+n%5)
            expected = reference.ingest(record)
            actual = store.ingest(record)
            assert actual["status"] == expected["status"]
            if n >= 29:
                for key in ("scores", "sensor_residuals"):
                    np.testing.assert_allclose(list(actual[key].values()), list(expected[key].values()), rtol=1e-5, atol=1e-6)
                assert actual["flags"] == expected["flags"]


def test_api_daily_flow(store):
    app = FastAPI()
    app.include_router(component1.router, prefix="/api/v1/component1")
    with patch.object(component1, "get_service", return_value=store), TestClient(app) as client:
        for n in range(30):
            response = client.post("/api/v1/component1/shadow/records", json=row(n))
            assert response.status_code == 200
        displayed = client.get("/api/v1/component1/shadow").json()["drives"][0]
        assert displayed == response.json()
        assert displayed["mode"] == "shadow" and displayed["regime_id"] is None and displayed["drift_event"] is None
        assert client.post("/api/v1/component1/shadow/records", json=row(0)).status_code == 409
        assert client.post("/api/v1/component1/shadow/records", json=row(30, smart_9_raw=None)).status_code == 422
        assert client.post("/api/v1/component1/analyze").json()["status"] == "unavailable"


def test_live_history_exposes_only_persisted_raw_daily_values(store):
    first = row(0, smart_5_raw=2, smart_194_raw=29)
    second = row(1, smart_5_raw=3, smart_194_raw=30)
    store.ingest(first)
    store.ingest(second)
    history = store.get_history(first["serial_number"])
    assert history["start_date"] == first["date"]
    assert history["end_date"] == second["date"]
    assert history["history_days"] == 2
    assert history["rows"] == [
        {"date": first["date"], **{field: first[field] for field in FEATURES}},
        {"date": second["date"], **{field: second[field] for field in FEATURES}},
    ]
    assert "failure" not in json.dumps(history)
    restarted = ShadowStore(store.path, store.scorer)
    assert restarted.get_history(first["serial_number"]) == history


def test_live_history_api_and_unknown_drive(store):
    record = row()
    store.ingest(record)
    app = FastAPI()
    app.include_router(component1.router, prefix="/api/v1/component1")
    with patch.object(component1, "get_service", return_value=store), TestClient(app) as client:
        response = client.get("/api/v1/component1/shadow/history", params={"serial_number": record["serial_number"]})
        assert response.status_code == 200
        assert response.json()["rows"][0]["smart_9_raw"] == record["smart_9_raw"]
        assert client.get("/api/v1/component1/shadow/history", params={"serial_number": "missing"}).status_code == 404


def test_independent_writers_preserve_drives(tmp_path, scorer):
    from concurrent.futures import ThreadPoolExecutor
    stores = [ShadowStore(tmp_path / "shared.json", scorer) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda i: stores[i].ingest(row(serial_number=f"drive-{i}")), range(4)))
    assert len(results) == len(stores[0].list_states()) == 4


def test_duplicate_raw_precision_and_missing_day(store):
    store.ingest(row(smart_9_raw=16777216))
    with pytest.raises(IntakeConflict, match="conflicting"):
        store.ingest(row(smart_9_raw=16777217))
    with pytest.raises(ValueError):
        store.ingest(row(1, smart_194_raw=None))
    assert store.ingest(row(2))["reset_reason"] == "gap"


def test_state_version_mismatch(store):
    store.ingest(row())
    state = json.loads(store.path.read_text())
    state["version"] = "other"
    store.path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="version mismatch"):
        store.list_states()


def test_api_missing_artifact_is_service_unavailable():
    app = FastAPI()
    app.include_router(component1.router)
    with patch.object(component1, "get_service", side_effect=FileNotFoundError("checkpoint")), TestClient(app) as client:
        assert client.get("/shadow").status_code == 503


@pytest.mark.parametrize("artifact", ["transformer_checkpoint", "transformer_preprocessing", "transformer_thresholds"])
def test_artifact_hash_tampering(tmp_path, artifact):
    import shutil
    result_root = "results/single_model_q1q2_2021_v1/"
    lock_relative = result_root + "q4_blind_candidate_protocol_v1/q4_candidate_lock.json"
    handoff_relative = result_root + "q4_research_review_and_handoff_v1/research_integration_handoff.json"
    lock = json.loads((ROOT / lock_relative).read_text())
    relatives = [lock_relative, handoff_relative] + [v["path"].replace("\\", "/") for k, v in lock["artifacts"].items() if k.startswith("transformer_") or k == "stream_contract"]
    for relative in relatives:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    target = tmp_path / lock["artifacts"][artifact]["path"].replace("\\", "/")
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        FrozenHDD(tmp_path)
