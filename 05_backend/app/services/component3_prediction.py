import io
import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from fastapi import HTTPException

from app.config import settings
from app.services import component3_service


# ==========================================================
# CONSTANTS (mirrors 07c_temporal_feature_engineering_q3_q4)
# ==========================================================

SMART_FEATURES = [
    "smart_1_raw",
    "smart_4_raw",
    "smart_5_raw",
    "smart_7_raw",
    "smart_9_raw",
    "smart_12_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_192_raw",
    "smart_193_raw",
    "smart_194_raw",
    "smart_197_raw",
    "smart_199_raw",
    "smart_240_raw",
    "smart_241_raw",
    "smart_242_raw",
]

WINDOWS = [7, 14, 30]

HORIZONS = [7, 14, 30]

MAX_SERIALS = 25

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

SMART_LABELS = {
    "smart_1_raw": "Read Error Rate",
    "smart_4_raw": "Seek Error Rate",
    "smart_5_raw": "Reallocated Sectors",
    "smart_7_raw": "Seek Time / Retryable Errors",
    "smart_9_raw": "Power-On Hours",
    "smart_12_raw": "Power Cycle Count",
    "smart_187_raw": "Command Timeout",
    "smart_188_raw": "Unused Reserved Blocks",
    "smart_192_raw": "Unsafe Shutdown Count",
    "smart_193_raw": "Load/Unload Cycle Count",
    "smart_194_raw": "Temperature",
    "smart_197_raw": "Current Pending Sectors",
    "smart_199_raw": "UDMA CRC Errors",
    "smart_240_raw": "Workload (Host HM)",
    "smart_241_raw": "Total LBAs Written",
    "smart_242_raw": "Total LBAs Read",
}

SMART_TO_FACTOR = {
    "smart_1_raw": "READ_ERROR_RATE",
    "smart_4_raw": "SEEK_ERROR_RATE",
    "smart_5_raw": "REALLOCATED_SECTORS",
    "smart_12_raw": "POWER_CYCLE_ACTIVITY",
    "smart_192_raw": "POWEROFF_RETRACT_ACTIVITY",
    "smart_193_raw": "LOAD_CYCLE_ACTIVITY",
    "smart_194_raw": "TEMPERATURE",
    "smart_197_raw": "PENDING_SECTORS",
    "smart_241_raw": "WRITE_WORKLOAD",
    "smart_242_raw": "READ_WORKLOAD",
}


def _c3_root() -> Path:
    return settings.HDD_PROJECT_ROOT / "03_component_3_causal_xai"


def _load_feature_list() -> List[str]:
    manifest_path = (
        _c3_root()
        / "data"
        / "interim"
        / "feature_leakage_audit"
        / "model_feature_manifest.csv"
    )
    manifest = pd.read_csv(manifest_path)
    return (
        manifest.loc[
            manifest["approved_for_reference_model"] == True,
            "feature",
        ]
        .tolist()
    )


_model_cache: Dict[int, Any] = {}
_threshold_cache: Dict[int, float] = {}
_feature_list_cache: List[str] = []


def _load_model(horizon: int):
    if horizon not in _model_cache:
        model_path = (
            _c3_root()
            / "models"
            / "reference_model"
            / f"{horizon}d"
            / "xgboost_reference.joblib"
        )
        if not model_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Reference model for {horizon}d horizon not found on server.",
            )
        _model_cache[horizon] = joblib.load(model_path)
    return _model_cache[horizon]


def _load_threshold(horizon: int) -> float:
    if horizon not in _threshold_cache:
        metrics_path = (
            _c3_root()
            / "results"
            / "reference_model"
            / f"{horizon}d"
            / "metrics.csv"
        )
        metrics = pd.read_csv(metrics_path)
        validation_row = metrics[metrics["split"] == "validation"].iloc[0]
        _threshold_cache[horizon] = float(validation_row["threshold"])
    return _threshold_cache[horizon]


def _get_feature_list() -> List[str]:
    global _feature_list_cache
    if not _feature_list_cache:
        _feature_list_cache = _load_feature_list()
    return _feature_list_cache


# ==========================================================
# FEATURE ENGINEERING (exact 07c semantics, latest row only)
# ==========================================================


def engineer_latest_rows(df: pd.DataFrame) -> pd.DataFrame:

    work = df.copy()

    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    work = work.dropna(subset=["date", "serial_number"])

    work = work.sort_values(
        ["serial_number", "date"]
    ).reset_index(drop=True)

    for feature in SMART_FEATURES:
        if feature not in work.columns:
            work[feature] = np.nan
        work[feature] = pd.to_numeric(
            work[feature], errors="coerce"
        ).astype("float32")

    # Every uploaded row counts as one observation (07c counts
    # the non-null failure label; uploads are unlabeled).
    work["failure"] = 1

    latest_records = []

    for serial, group in work.groupby("serial_number", sort=False):

        group = group.sort_values("date").reset_index(drop=True)

        record = group.iloc[-1].to_dict()

        # --------------------------------------------------
        # 1-day differences (only when previous row is
        # exactly one calendar day earlier)
        # --------------------------------------------------

        differences = group[SMART_FEATURES].diff()
        previous_gap = group["date"].diff().dt.days
        valid_1d = (previous_gap == 1).to_numpy()

        for feature in SMART_FEATURES:
            values = differences[feature].to_numpy(dtype="float32")
            values[~valid_1d] = np.nan
            record[f"{feature}_diff_1d"] = values[-1]

        # --------------------------------------------------
        # Calendar rolling windows (min_periods: 1 for
        # count/mean/max, 2 for std — same as 07c)
        # --------------------------------------------------

        for window in WINDOWS:

            record[f"observations_{window}d"] = int(
                group.rolling(
                    window=f"{window}D",
                    on="date",
                    min_periods=1,
                )["failure"].count().iloc[-1]
            )

            rolling = group.rolling(
                window=f"{window}D",
                on="date",
                min_periods=1,
            )

            rolling_mean = rolling[SMART_FEATURES].mean()
            rolling_max = rolling[SMART_FEATURES].max()

            rolling_std = group.rolling(
                window=f"{window}D",
                on="date",
                min_periods=2,
            )[SMART_FEATURES].std()

            for feature in SMART_FEATURES:
                record[f"{feature}_mean_{window}d"] = (
                    rolling_mean[feature].iloc[-1]
                )
                record[f"{feature}_std_{window}d"] = (
                    rolling_std[feature].iloc[-1]
                )
                record[f"{feature}_max_{window}d"] = (
                    rolling_max[feature].iloc[-1]
                )

        record["has_sufficient_7d_history"] = bool(
            record["observations_7d"] >= 6
        )
        record["has_sufficient_14d_history"] = bool(
            record["observations_14d"] >= 12
        )
        record["has_sufficient_30d_history"] = bool(
            record["observations_30d"] >= 24
        )

        latest_records.append(record)

    return pd.DataFrame(latest_records)


# ==========================================================
# SHAP + CAUSAL CONTEXT
# ==========================================================


def _top_shap_factors(horizon: int, row_df: pd.DataFrame) -> List[Dict[str, Any]]:

    import shap

    model = _load_model(horizon)

    features = _get_feature_list()

    X_row = row_df.reindex(columns=features).astype("float32")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_row)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_row = np.asarray(shap_values).reshape(-1)

    order = np.argsort(np.abs(shap_row))[::-1][:5]

    top = []

    for index in order:
        feature_name = features[int(index)]
        smart_base = feature_name.split("_diff_")[0].split("_mean_")[0]
        smart_base = smart_base.split("_std_")[0].split("_max_")[0]
        top.append(
            {
                "feature": feature_name,
                "smart_attribute": smart_base,
                "smart_label": SMART_LABELS.get(smart_base, smart_base),
                "feature_value": float(X_row.iloc[0, int(index)])
                if pd.notna(X_row.iloc[0, int(index)])
                else None,
                "shap_value": float(shap_row[int(index)]),
            }
        )

    return top


def _causal_context(top_factors: List[Dict[str, Any]]) -> List[str]:

    notes = []

    try:
        pathways = component3_service.get_temporal_pathways()
        effects = component3_service.get_matched_effects()
    except HTTPException:
        return [
            "Causal evidence payload unavailable; "
            "showing model-only explanation."
        ]

    factor_ids = set()

    for factor in top_factors:
        mapped = SMART_TO_FACTOR.get(factor["smart_attribute"])
        if mapped:
            factor_ids.add(mapped)

    for pathway in pathways:
        source_id = pathway.get("source_factor_id")
        target_id = pathway.get("target_factor_id")
        if source_id in factor_ids or target_id in factor_ids:
            runs = pathway.get("strongest_detection_rate")
            runs_text = (
                f"{int(round(runs * 5))}/5 PCMCI runs" if runs else "stable"
            )
            notes.append(
                f"Causal pathway: {pathway.get('source_factor_name')}"
                f" \u2192 {pathway.get('target_factor_name')} "
                f"(temporal dependency, detected in {runs_text}; "
                f"temporal ordering evidence only, not a proven mechanism)."
            )

    for effect in effects:
        if effect.get("factor_id") in factor_ids:
            diff = effect.get("risk_difference_pp")
            if diff is not None:
                notes.append(
                    f"Matched evidence: {effect.get('factor_name')} "
                    f"associated with {diff:+.1f} pp 30-day failure-risk "
                    f"difference (observational ATT, propensity-matched)."
                )

    if not notes:
        notes.append(
            "Top drivers do not map to a factor with stable PCMCI "
            "pathways or matched ATT evidence; treat this as a "
            "model-only prediction."
        )

    return notes


# ==========================================================
# PREDICTION ENTRY POINT
# ==========================================================


def predict_from_csv_bytes(raw_bytes: bytes) -> Dict[str, Any]:

    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"CSV too large (limit {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
        )

    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse CSV: {exc}",
        )

    df.columns = [str(col).strip() for col in df.columns]

    required = {"serial_number", "date"}

    if not required.issubset(set(df.columns)):
        raise HTTPException(
            status_code=400,
            detail=(
                "CSV must contain 'serial_number' and 'date' columns. "
                f"Optional SMART columns: {', '.join(SMART_FEATURES)}"
            ),
        )

    smart_present = [col for col in SMART_FEATURES if col in df.columns]

    if not smart_present:
        raise HTTPException(
            status_code=400,
            detail=(
                "CSV contains no recognized SMART columns. "
                f"Provide at least one of: {', '.join(SMART_FEATURES)}"
            ),
        )

    df["serial_number"] = df["serial_number"].astype(str).str.strip()

    df = df[df["serial_number"] != ""]

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="CSV contains no data rows.",
        )

    serial_count = df["serial_number"].nunique()

    if serial_count > MAX_SERIALS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many drives in one upload (limit {MAX_SERIALS}).",
        )

    engineered = engineer_latest_rows(df)

    features = _get_feature_list()

    X = engineered.reindex(columns=features).astype("float32")

    predictions = []

    for position, (_, row) in enumerate(engineered.iterrows()):

        serial = str(row["serial_number"])

        as_of_date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")

        history_rows = int(
            (df["serial_number"] == serial).sum()
        )

        row_df = X.iloc[[position]]

        warnings = []

        if not row["has_sufficient_7d_history"]:
            warnings.append(
                "Less than 7 days of history; short-horizon "
                "features are less reliable."
            )
        if not row["has_sufficient_30d_history"]:
            warnings.append(
                "Less than 30 days of history; 30-day features "
                "are based on a partial window (30+ days recommended)."
            )

        risks = {}

        top_factors_all = []

        for horizon in HORIZONS:

            model = _load_model(horizon)

            probability = float(
                model.predict_proba(row_df)[:, 1][0]
            )

            threshold = _load_threshold(horizon)

            risks[f"{horizon}d"] = {
                "risk_probability": probability,
                "risk_percent": round(probability * 100, 3),
                "decision_threshold": threshold,
                "flagged": bool(probability >= threshold),
            }

            top_factors_all.append(
                {
                    "horizon": f"{horizon}d",
                    "top_factors": _top_shap_factors(horizon, row_df),
                }
            )

        primary_top = next(
            (
                entry["top_factors"]
                for entry in top_factors_all
                if entry["horizon"] == "30d"
            ),
            top_factors_all[0]["top_factors"] if top_factors_all else [],
        )

        predictions.append(
            {
                "serial_number": serial,
                "as_of_date": as_of_date,
                "history_rows": history_rows,
                "sufficient_history": {
                    "7d": bool(row["has_sufficient_7d_history"]),
                    "14d": bool(row["has_sufficient_14d_history"]),
                    "30d": bool(row["has_sufficient_30d_history"]),
                },
                "risks": risks,
                "top_factors_by_horizon": top_factors_all,
                "causal_context": _causal_context(primary_top),
                "warnings": warnings,
            }
        )

    return {
        "predictions": predictions,
        "model_info": {
            "model": "XGBoost reference (09B), early-stopping ablation available separately",
            "features": len(features),
            "smart_columns_recognized": smart_present,
            "thresholds_source": "validation best-F1 (per horizon)",
            "note": (
                "Risk scores come from an out-of-time validated model "
                "(trained Aug-Sep 2025). Absolute probabilities are "
                "conservative; use ranking and driver explanation, "
                "not the raw percentage alone."
            ),
        },
    }
