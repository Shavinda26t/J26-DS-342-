import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import HTTPException
from app.config import settings

# In-memory caches for static payload files
_payload_cache: Dict[str, Any] = {}

def get_c3_export_dir() -> Path:
    return settings.HDD_PROJECT_ROOT / "03_component_3_causal_xai" / "results" / "api_exports"

def get_c3_figures_dir() -> Path:
    return settings.HDD_PROJECT_ROOT / "03_component_3_causal_xai" / "figures" / "integrated_evidence" / "final"

def _load_json_file(file_path: Path) -> Dict[str, Any]:
    cache_key = str(file_path.resolve())
    if cache_key in _payload_cache:
        return _payload_cache[cache_key]
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Required Component 3 data artifact not found: {file_path.name}"
        )
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            _payload_cache[cache_key] = data
            return data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read Component 3 data file {file_path.name}: {str(e)}"
        )

def load_api_payload() -> Dict[str, Any]:
    file_path = get_c3_export_dir() / "component3_api_payload.json"
    return _load_json_file(file_path)

def load_dashboard_payload() -> Dict[str, Any]:
    file_path = get_c3_export_dir() / "component3_dashboard_payload.json"
    return _load_json_file(file_path)

FIGURE_WHITELIST = {
    "integrated-evidence": "01_integrated_factor_evidence.png",
    "matched-att": "02_matched_att_forest_plot.png",
    "pcmci-network": "03_pcmci_stable_temporal_network.png",
    "degradation-pathway": "04_degradation_root_cause_pathway.png",
    "evidence-classes": "05_integrated_evidence_class_summary.png"
}

def get_health_status() -> Dict[str, Any]:
    api_payload_path = get_c3_export_dir() / "component3_api_payload.json"
    dashboard_payload_path = get_c3_export_dir() / "component3_dashboard_payload.json"
    figures_dir = get_c3_figures_dir()
    
    fig_count = 0
    if figures_dir.exists() and figures_dir.is_dir():
        fig_count = len([f for f in figures_dir.glob("*.png") if f.is_file()])
        
    return {
        "component_id": "C3",
        "component_name": "Causal Explainable AI for HDD Failure Analysis",
        "status": "ok",
        "api_payload_available": api_payload_path.exists(),
        "dashboard_payload_available": dashboard_payload_path.exists(),
        "figure_count": fig_count
    }

def get_summary() -> Dict[str, Any]:
    payload = load_dashboard_payload()
    summary = payload.get("summary")
    if not summary:
        raise HTTPException(status_code=404, detail="Dashboard summary section not found in payload.")
    return summary

def get_dashboard() -> Dict[str, Any]:
    return load_dashboard_payload()

def get_factors() -> List[Dict[str, Any]]:
    payload = load_api_payload()
    catalog = payload.get("factor_evidence_catalog")
    if catalog is None:
        dashboard = load_dashboard_payload()
        catalog = dashboard.get("integrated_factor_catalog", [])
    return catalog

def get_factor_by_id(factor_id: str) -> Dict[str, Any]:
    factors = get_factors()
    target_id = factor_id.strip().lower()
    for factor in factors:
        f_id = str(factor.get("factor_id", "")).strip().lower()
        if f_id == target_id:
            return factor
    raise HTTPException(
        status_code=404,
        detail=f"Factor with ID '{factor_id}' not found in Component 3 evidence catalog."
    )

def get_root_causes() -> List[Dict[str, Any]]:
    payload = load_dashboard_payload()
    return payload.get("root_cause_candidates", [])

def get_temporal_pathways() -> List[Dict[str, Any]]:
    payload = load_dashboard_payload()
    return payload.get("temporal_pathways", [])

def get_matched_effects() -> List[Dict[str, Any]]:
    payload = load_dashboard_payload()
    return payload.get("matched_failure_risk_evidence", [])

def get_cases() -> Dict[str, Any]:
    case_types = ["TP", "FP", "FN", "TN"]
    available_cases = {}
    cases_dir = get_c3_export_dir() / "representative_cases"
    
    for c_type in case_types:
        case_file = cases_dir / f"{c_type.lower()}_dashboard_payload.json"
        if case_file.exists():
            try:
                data = _load_json_file(case_file)
                available_cases[c_type] = {
                    "case_type": c_type,
                    "available": True,
                    "hdd": data.get("hdd", {}),
                    "prediction": data.get("prediction", {})
                }
            except Exception:
                available_cases[c_type] = {"case_type": c_type, "available": True}
        else:
            available_cases[c_type] = {"case_type": c_type, "available": False}
            
    return {
        "description": "Representative local explanation cases",
        "cases": available_cases
    }

def get_case_by_type(case_type: str) -> Dict[str, Any]:
    normalized_type = case_type.strip().lower()
    valid_types = ["tp", "fp", "fn", "tn"]
    if normalized_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported case type '{case_type}'. Supported types are TP, FP, FN, TN."
        )
        
    case_file = get_c3_export_dir() / "representative_cases" / f"{normalized_type}_dashboard_payload.json"
    if not case_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Representative case payload for '{case_type.upper()}' not found."
        )
        
    return _load_json_file(case_file)

def get_figures_metadata() -> List[Dict[str, Any]]:
    figures_dir = get_c3_figures_dir()
    result = []
    
    descriptions = {
        "integrated-evidence": "Overview of factor-level global SHAP, PCMCI temporal degree, and matched ATT risk difference.",
        "matched-att": "Forest plot of incident 30-day matched failure-risk differences with clustered-bootstrap 95% CIs.",
        "pcmci-network": "Stable temporal conditional dependency network derived from PCMCI analysis.",
        "degradation-pathway": "Identified degradation pathway showing physical and error relationship dynamics.",
        "evidence-classes": "Summary distribution of factors across integrated evidence classifications."
    }
    
    titles = {
        "integrated-evidence": "Integrated Factor Evidence Overview",
        "matched-att": "Matched ATT Forest Plot",
        "pcmci-network": "PCMCI Stable Temporal Network",
        "degradation-pathway": "Degradation Root-Cause Pathway",
        "evidence-classes": "Integrated Evidence Class Summary"
    }

    for key, filename in FIGURE_WHITELIST.items():
        file_path = figures_dir / filename
        result.append({
            "key": key,
            "title": titles.get(key, key),
            "filename": filename,
            "available": file_path.exists(),
            "endpoint": f"/api/component3/figures/{key}",
            "description": descriptions.get(key, "")
        })
        
    return result

def get_figure_filepath(figure_key: str) -> Path:
    normalized_key = figure_key.strip().lower()
    if normalized_key not in FIGURE_WHITELIST:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid figure key '{figure_key}'. Whitelisted keys: {list(FIGURE_WHITELIST.keys())}"
        )
        
    filename = FIGURE_WHITELIST[normalized_key]
    file_path = get_c3_figures_dir() / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Figure image file '{filename}' not found."
        )
        
    return file_path

# Backward compatibility functions
def generate_explanation(request: Any) -> Dict[str, Any]:
    # Legacy wrapper matching existing signature if needed
    summary = get_summary()
    root_causes = get_root_causes()
    return {
        "serial_number": getattr(request, "serial_number", "UNKNOWN"),
        "prediction_risk": 0.0,
        "global_explanation": {},
        "local_shap_values": {},
        "important_smart_attributes": [rc.get("factor_name", "") for rc in root_causes],
        "causal_relationships": get_temporal_pathways(),
        "causal_effect_estimates": {},
        "candidate_root_causes": [rc.get("factor_name", "") for rc in root_causes],
        "confidence_score": 1.0,
        "smart_trend_info": {}
    }

def evaluate_root_cause(request: Any) -> Dict[str, Any]:
    summary = get_summary()
    return {
        "serial_number": getattr(request, "serial_number", "UNKNOWN"),
        "primary_root_cause": summary.get("primary_root_cause_candidate", "Reported Uncorrectable Errors"),
        "causal_chain": ["Reallocated Sectors", "Reported Uncorrectable Errors"],
        "evidence_score": 1.0
    }
