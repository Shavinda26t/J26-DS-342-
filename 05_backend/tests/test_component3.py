from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_component3_health():
    response = client.get("/api/component3/health")
    assert response.status_code == 200
    data = response.json()
    assert data["component_id"] == "C3"
    assert data["status"] == "ok"
    assert data["api_payload_available"] is True
    assert data["dashboard_payload_available"] is True

def test_component3_summary():
    response = client.get("/api/component3/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["canonical_factor_count"] == 16
    assert data["primary_root_cause_candidate"] == "Reported Uncorrectable Errors"

def test_component3_dashboard():
    response = client.get("/api/component3/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "root_cause_candidates" in data

def test_component3_factors():
    response = client.get("/api/component3/factors")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_component3_factor_by_id():
    # Test valid lookup (case-insensitive)
    response = client.get("/api/component3/factors/REPORTED_UNCORRECTABLE")
    assert response.status_code == 200
    data = response.json()
    assert data["factor_id"] == "REPORTED_UNCORRECTABLE"

    response_lower = client.get("/api/component3/factors/reported_uncorrectable")
    assert response_lower.status_code == 200
    assert response_lower.json()["factor_id"] == "REPORTED_UNCORRECTABLE"

    # Test invalid lookup
    invalid_response = client.get("/api/component3/factors/NON_EXISTENT_FACTOR")
    assert invalid_response.status_code == 404

def test_component3_root_causes():
    response = client.get("/api/component3/root-causes")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    factor_names = [rc["factor_name"] for rc in data]
    assert "Reported Uncorrectable Errors" in factor_names

def test_component3_temporal_pathways():
    response = client.get("/api/component3/temporal-pathways")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_component3_matched_effects():
    response = client.get("/api/component3/matched-effects")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

def test_component3_cases():
    response = client.get("/api/component3/cases")
    assert response.status_code == 200
    data = response.json()
    assert "cases" in data
    assert "TP" in data["cases"]

def test_component3_case_by_type():
    # Valid case types (case-insensitive)
    for c_type in ["TP", "fp", "FN", "tn"]:
        res = client.get(f"/api/component3/cases/{c_type}")
        assert res.status_code == 200
        assert "case_type" in res.json()

    # Invalid case type
    res_inv = client.get("/api/component3/cases/INVALID_TYPE")
    assert res_inv.status_code == 400

def test_component3_figures_metadata():
    response = client.get("/api/component3/figures")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 5

def test_component3_figure_delivery():
    # Test valid figure
    response = client.get("/api/component3/figures/integrated-evidence")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    # Test invalid figure key
    invalid_response = client.get("/api/component3/figures/invalid-figure-key")
    assert invalid_response.status_code == 400
