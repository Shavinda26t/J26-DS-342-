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
    "smart_4_raw": "Start/Stop Activity",
    "smart_5_raw": "Reallocated Sectors",
    "smart_7_raw": "Seek Error Rate",
    "smart_9_raw": "Power-On Hours",
    "smart_12_raw": "Power Cycle Count",
    "smart_187_raw": "Reported Uncorrectable Errors",
    "smart_188_raw": "Command Timeout",
    "smart_192_raw": "Power-Off Retract Activity",
    "smart_193_raw": "Load/Unload Cycle Count",
    "smart_194_raw": "Temperature",
    "smart_197_raw": "Current Pending Sectors",
    "smart_199_raw": "UDMA CRC Errors",
    "smart_240_raw": "Head Flying Hours",
    "smart_241_raw": "Total LBAs Written",
    "smart_242_raw": "Total LBAs Read",
}

# Canonical 16-factor mapping (source: 13b_create_canonical_factor_mapping,
# identical to factor_evidence_catalog in the exported API payload).
RAW_TO_FACTOR = {
    "smart_1_raw": "READ_ERROR_RATE",
    "smart_4_raw": "START_STOP_ACTIVITY",
    "smart_5_raw": "REALLOCATED_SECTORS",
    "smart_7_raw": "SEEK_ERROR_RATE",
    "smart_9_raw": "DRIVE_AGE",
    "smart_12_raw": "POWER_CYCLE_ACTIVITY",
    "smart_187_raw": "REPORTED_UNCORRECTABLE",
    "smart_188_raw": "COMMAND_TIMEOUT",
    "smart_192_raw": "POWEROFF_RETRACT_ACTIVITY",
    "smart_193_raw": "LOAD_CYCLE_ACTIVITY",
    "smart_194_raw": "TEMPERATURE",
    "smart_197_raw": "PENDING_SECTORS",
    "smart_199_raw": "CRC_ERRORS",
    "smart_240_raw": "HEAD_FLYING_HOURS",
    "smart_241_raw": "WRITE_WORKLOAD",
    "smart_242_raw": "READ_WORKLOAD",
}

FACTOR_NAMES = {
    "READ_ERROR_RATE": "Read Error Rate",
    "START_STOP_ACTIVITY": "Start/Stop Activity",
    "REALLOCATED_SECTORS": "Reallocated Sectors",
    "SEEK_ERROR_RATE": "Seek Error Rate",
    "DRIVE_AGE": "Drive Age / Power-On Hours",
    "POWER_CYCLE_ACTIVITY": "Power Cycle Activity",
    "REPORTED_UNCORRECTABLE": "Reported Uncorrectable Errors",
    "COMMAND_TIMEOUT": "Command Timeout",
    "POWEROFF_RETRACT_ACTIVITY": "Power-Off Retract Activity",
    "LOAD_CYCLE_ACTIVITY": "Load Cycle Activity",
    "TEMPERATURE": "Drive Temperature",
    "PENDING_SECTORS": "Current Pending Sectors",
    "CRC_ERRORS": "CRC / Interface Errors",
    "HEAD_FLYING_HOURS": "Head Flying Hours",
    "WRITE_WORKLOAD": "Write Workload",
    "READ_WORKLOAD": "Read Workload",
}

# Longest-first so smart_1_raw cannot shadow smart_1_raw_xxx handling.
_RAW_BASES = sorted(RAW_TO_FACTOR, key=len, reverse=True)


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

_explainer_cache: Dict[int, Any] = {}


def _get_explainer(horizon: int):
    import shap

    if horizon not in _explainer_cache:
        _explainer_cache[horizon] = shap.TreeExplainer(
            _load_model(horizon)
        )
    return _explainer_cache[horizon]


def _feature_shap_values(horizon: int, row_df: pd.DataFrame):
    """SHAP values for one row against one horizon's model."""

    features = _get_feature_list()

    X_row = row_df.reindex(columns=features).astype("float32")

    shap_values = _get_explainer(horizon).shap_values(X_row)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_row = np.asarray(shap_values).reshape(-1)

    return X_row, shap_row


def _map_feature_to_factor(feature_name: str):
    """176 engineered features -> canonical factor_id (or None)."""

    for raw_base in _RAW_BASES:
        if (
            feature_name == raw_base
            or feature_name.startswith(raw_base + "_")
        ):
            return RAW_TO_FACTOR[raw_base]

    return None


def _top_shap_factors(horizon: int, row_df: pd.DataFrame) -> List[Dict[str, Any]]:

    X_row, shap_row = _feature_shap_values(horizon, row_df)

    features = _get_feature_list()

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
        mapped = RAW_TO_FACTOR.get(factor["smart_attribute"])
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
# LOCAL FACTOR AGGREGATION + FLEET-EVIDENCE ENRICHMENT
# ==========================================================


def _fleet_evidence_index():
    """Static fleet-level evidence, keyed by factor_id.

    This is READ-ONLY reuse of the precomputed PCMCI pathway and
    matched-ATT payloads. A single-drive upload never triggers new
    PCMCI or matched-ATT analysis; fleet-level research evidence
    stays exactly as exported.
    """

    evidence = {}

    try:
        factors = component3_service.get_factors()
        pathways = component3_service.get_temporal_pathways()
        effects = component3_service.get_matched_effects()
    except HTTPException:
        return {}

    for entry in factors:
        factor_id = entry.get("factor_id")
        if not factor_id:
            continue
        evidence[factor_id] = {
            "category": entry.get("category"),
            "semantic_status": entry.get("semantic_status"),
            "predictive_evidence": entry.get("predictive_evidence"),
            "temporal_evidence": entry.get("temporal_evidence"),
            "matched_failure_risk_evidence": entry.get(
                "matched_failure_risk_evidence"
            ),
            "integrated_evidence": entry.get("integrated_evidence"),
            "temporal_pathways": [],
            "matched_effects": [],
        }

    for pathway in pathways:
        for side, other in (
            ("source_factor_id", "target_factor_id"),
            ("target_factor_id", "source_factor_id"),
        ):
            factor_id = pathway.get(side)
            if factor_id in evidence:
                other_id = pathway.get(other)
                evidence[factor_id]["temporal_pathways"].append(
                    {
                        "direction": "outgoing"
                        if side == "source_factor_id"
                        else "incoming",
                        "counterpart_factor_id": other_id,
                        "counterpart_factor_name": FACTOR_NAMES.get(
                            other_id, other_id
                        ),
                        "strongest_detection_rate": pathway.get(
                            "strongest_detection_rate"
                        ),
                        "strongest_sign_consistency": pathway.get(
                            "strongest_sign_consistency"
                        ),
                        "strongest_lag_days": pathway.get(
                            "strongest_lag_days"
                        ),
                    }
                )

    for effect in effects:
        factor_id = effect.get("factor_id")
        if factor_id in evidence:
            evidence[factor_id]["matched_effects"].append(
                {
                    "risk_difference_pp": effect.get("risk_difference_pp"),
                    "cluster_bootstrap_95ci_pp": effect.get(
                        "cluster_bootstrap_95ci_pp"
                    ),
                    "mcnemar_exact_p_value": effect.get(
                        "mcnemar_exact_p_value"
                    ),
                    "evidence_class": effect.get("evidence_class"),
                    "analysis_role": effect.get("analysis_role"),
                }
            )

    return evidence


def _local_factor_ranking(
    horizon: int,
    row_df: pd.DataFrame,
    evidence_index: Dict[str, Any],
) -> List[Dict[str, Any]]:

    X_row, shap_row = _feature_shap_values(horizon, row_df)

    features = _get_feature_list()

    factor_totals: Dict[str, float] = {}
    factor_dominant: Dict[str, Dict[str, Any]] = {}

    for index, feature_name in enumerate(features):

        factor_id = _map_feature_to_factor(feature_name)
        if factor_id is None:
            continue

        shap_value = float(shap_row[index])
        abs_value = abs(shap_value)

        factor_totals[factor_id] = (
            factor_totals.get(factor_id, 0.0) + abs_value
        )

        current = factor_dominant.get(factor_id)
        if current is None or abs_value > current["abs_shap"]:
            raw_base = next(
                (base for base in _RAW_BASES
                 if feature_name == base
                 or feature_name.startswith(base + "_")),
                feature_name,
            )
            raw_value = X_row.iloc[0, index]
            factor_dominant[factor_id] = {
                "abs_shap": abs_value,
                "metric": smart_metric_suffix(feature_name, raw_base),
                "smart_attribute": raw_base,
                "smart_label": SMART_LABELS.get(raw_base, raw_base),
                "feature": feature_name,
                "shap_value": shap_value,
                "feature_value": float(raw_value)
                if pd.notna(raw_value)
                else None,
            }

    total_mass = sum(factor_totals.values()) or 1.0

    ranking = []

    for factor_id, total_abs in sorted(
        factor_totals.items(), key=lambda item: item[1], reverse=True
    ):

        dominant = factor_dominant[factor_id]

        fleet = evidence_index.get(factor_id) or {}

        matched = fleet.get("matched_failure_risk_evidence") or {}

        temporal = fleet.get("temporal_evidence") or {}

        predictive = fleet.get("predictive_evidence") or {}

        integrated = fleet.get("integrated_evidence") or {}

        ranking.append(
            {
                "factor_id": factor_id,
                "factor_name": FACTOR_NAMES.get(factor_id, factor_id),
                "category": fleet.get("category"),
                "local_shap_total_abs": round(total_abs, 6),
                "local_shap_share": round(total_abs / total_mass, 6),
                "local_shap_share_pct": round(
                    total_abs / total_mass * 100, 3
                ),
                "local_direction": (
                    "increases_risk"
                    if dominant["shap_value"] >= 0
                    else "decreases_risk"
                ),
                "dominant_metric": dominant["metric"],
                "dominant_feature": dominant["feature"],
                "dominant_feature_value": dominant["feature_value"],
                "dominant_smart_attribute": dominant["smart_attribute"],
                "dominant_smart_label": dominant["smart_label"],
                "fleet_evidence": {
                    "global_shap_rank": predictive.get("consensus_rank"),
                    "global_shap_share_pct": predictive.get(
                        "normalized_shap_share_pct"
                    ),
                    "predictive_band": predictive.get("band"),
                    "temporal_connectivity_class": temporal.get(
                        "connectivity_class"
                    ),
                    "strongest_temporal_relation": temporal.get(
                        "strongest_relation"
                    ),
                    "matched_risk_difference_pp": matched.get(
                        "risk_difference_pp"
                    ),
                    "matched_ci_pp": fleet.get("matched_effects", [{}])[0].get(
                        "cluster_bootstrap_95ci_pp"
                    )
                    if fleet.get("matched_effects")
                    else None,
                    "matched_evidence_class": matched.get("evidence_class"),
                    "integrated_classification": integrated.get(
                        "classification"
                    ),
                    "root_cause_candidate": integrated.get(
                        "root_cause_candidate"
                    ),
                    "temporal_pathways": fleet.get("temporal_pathways", []),
                },
            }
        )

    for position, entry in enumerate(ranking, start=1):
        entry["local_rank"] = position

    return ranking


def smart_metric_suffix(feature_name: str, raw_base: str) -> str:
    suffix = feature_name[len(raw_base):].lstrip("_")
    return suffix or "raw"


def _local_root_cause_explanation(
    serial: str,
    as_of_date: str,
    risks: Dict[str, Any],
    ranking: List[Dict[str, Any]],
) -> Dict[str, Any]:

    primary = risks.get("30d") or next(iter(risks.values()), None)

    threshold = primary.get("decision_threshold") if primary else None
    probability = primary.get("risk_probability") if primary else None
    flagged = primary.get("flagged") if primary else None

    lead = []

    top = ranking[:3]

    for entry in top:
        lead.append(entry["factor_name"])

    if probability is not None:
        decision_text = (
            "above the validation-selected decision threshold "
            f"({threshold:.4f}) and is flagged as high risk"
            if flagged
            else "below the validation-selected decision threshold "
            f"({threshold:.4f}) and is not flagged"
        )
        headline = (
            f"Drive {serial} as of {as_of_date} has a 30-day failure "
            f"probability of {probability * 100:.2f}%, which is "
            f"{decision_text}."
        )
    else:
        headline = (
            f"Drive {serial} as of {as_of_date}: model scoring "
            "unavailable."
        )

    if top:
        dominant = top[0]
        driver_text = (
            f"The dominant local driver is {dominant['factor_name']} "
            f"({dominant['local_shap_share_pct']:.1f}% of local SHAP mass, "
            f"{dominant['local_direction'].replace('_', ' ')}, "
            f"metric {dominant['dominant_metric']})."
        )
    else:
        driver_text = "No factor-level local drivers could be mapped."

    fleet_bits = []

    for entry in top:
        fleet = entry["fleet_evidence"]
        if fleet.get("matched_risk_difference_pp") is not None:
            fleet_bits.append(
                f"{entry['factor_name']} is associated with a "
                f"{fleet['matched_risk_difference_pp']:+.1f} pp fleet-level "
                "matched failure-risk difference"
            )
        if fleet.get("strongest_temporal_relation"):
            fleet_bits.append(
                f"fleet PCMCI links {fleet['strongest_temporal_relation']}"
            )

    if fleet_bits:
        fleet_text = (
            "Fleet-level evidence alignment: " + "; ".join(fleet_bits) + "."
        )
    else:
        fleet_text = (
            "Top local drivers do not align with strong fleet-level "
            "matched or temporal evidence; treat as model-only."
        )

    scope = (
        "Local explanation operates on this drive's uploaded history "
        "only. Fleet-level agreement is reused from static PCMCI and "
        "matched-ATT research evidence and is not recomputed per upload."
    )

    return {
        "headline": headline,
        "dominant_driver": driver_text,
        "fleet_alignment": fleet_text,
        "scope": scope,
        "text": " ".join([headline, driver_text, fleet_text, scope]),
    }


# ==========================================================
# DYNAMIC VISUALIZATION PAYLOAD
# ==========================================================

# Factors counted as "locally important" for a single uploaded HDD.
LOCAL_IMPORTANCE_SHARE_PCT = 5.0
LOCAL_IMPORTANCE_MAX = 6


def _locally_important_factors(
    ranking: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    important = [
        entry
        for entry in ranking
        if entry["local_shap_share_pct"] >= LOCAL_IMPORTANCE_SHARE_PCT
    ][:LOCAL_IMPORTANCE_MAX]

    if not important and ranking:
        important = ranking[:3]

    return important


def _smart_history_traces(
    df: pd.DataFrame,
    serial: str,
    ranking: List[Dict[str, Any]],
    max_factors: int = 5,
    max_points: int = 30,
) -> List[Dict[str, Any]]:
    """30-day raw SMART trajectories for the top local factors."""

    history = df[df["serial_number"] == serial].copy()

    history["date"] = pd.to_datetime(history["date"], errors="coerce")

    history = (
        history.dropna(subset=["date"])
        .sort_values("date")
        .tail(max_points)
    )

    traces = []

    for entry in ranking[:max_factors]:

        attribute = entry["dominant_smart_attribute"]

        if attribute not in df.columns:
            continue

        values = pd.to_numeric(history[attribute], errors="coerce")

        points = [
            {
                "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                "value": (None if pd.isna(v) else float(v)),
            }
            for d, v in zip(history["date"], values)
        ]

        numeric = [p["value"] for p in points if p["value"] is not None]

        trend = "flat"

        if len(numeric) >= 2:
            change = numeric[-1] - numeric[0]
            span = max(abs(numeric[0]), 1e-9)
            if change > 0.05 * span:
                trend = "increasing"
            elif change < -0.05 * span:
                trend = "decreasing"

        traces.append(
            {
                "factor_id": entry["factor_id"],
                "factor_name": entry["factor_name"],
                "smart_attribute": attribute,
                "smart_label": entry["dominant_smart_label"],
                "local_rank": entry["local_rank"],
                "local_shap_share_pct": entry["local_shap_share_pct"],
                "trend": trend,
                "points": points,
            }
        )

    return traces


def _att_overlay(ranking: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fleet ATT contrasts, with local-HDD relevance flags.

    ATT values are reused verbatim from the static fleet payload;
    only a boolean 'relevant_to_local_hdd' is added per factor.
    """

    important_ids = {
        entry["factor_id"] for entry in _locally_important_factors(ranking)
    }

    local_by_id = {entry["factor_id"]: entry for entry in ranking}

    factors = []

    for entry in ranking:

        fleet = entry["fleet_evidence"]

        att_pp = fleet.get("matched_risk_difference_pp")

        if att_pp is None:
            continue

        factors.append(
            {
                "factor_id": entry["factor_id"],
                "factor_name": entry["factor_name"],
                "risk_difference_pp": att_pp,
                "ci_pp": fleet.get("matched_ci_pp"),
                "evidence_class": fleet.get("matched_evidence_class"),
                "relevant_to_local_hdd": entry["factor_id"] in important_ids,
                "local_shap_share_pct": local_by_id[entry["factor_id"]][
                    "local_shap_share_pct"
                ],
            }
        )

    factors.sort(
        key=lambda item: (not item["relevant_to_local_hdd"],
                          -(item["risk_difference_pp"] or 0))
    )

    return {
        "title": "Fleet Matched Failure-Risk Evidence Relevant to This HDD",
        "disclaimer": (
            "ATT values are fleet-level observational estimates and are "
            "not individual treatment effects for this HDD."
        ),
        "factors": factors,
    }


def _pcmci_overlay(
    pathways: List[Dict[str, Any]],
    ranking: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Static stable PCMCI network + local-HDD highlight flags."""

    important_ids = {
        entry["factor_id"] for entry in _locally_important_factors(ranking)
    }

    local_by_id = {entry["factor_id"]: entry for entry in ranking}

    def node(factor_id: str) -> Dict[str, Any]:
        entry = local_by_id.get(factor_id)
        return {
            "factor_id": factor_id,
            "factor_name": FACTOR_NAMES.get(factor_id, factor_id),
            "locally_important": factor_id in important_ids,
            "local_shap_share_pct": (
                entry["local_shap_share_pct"] if entry else 0.0
            ),
        }

    nodes: Dict[str, Dict[str, Any]] = {}

    edges = []

    for pathway in pathways:

        source = pathway.get("source_factor_id")
        target = pathway.get("target_factor_id")

        if not source or not target:
            continue

        nodes.setdefault(source, node(source))
        nodes.setdefault(target, node(target))

        edges.append(
            {
                "source_factor_id": source,
                "source_factor_name": FACTOR_NAMES.get(source, source),
                "target_factor_id": target,
                "target_factor_name": FACTOR_NAMES.get(target, target),
                "strongest_detection_rate": pathway.get(
                    "strongest_detection_rate"
                ),
                "strongest_sign_consistency": pathway.get(
                    "strongest_sign_consistency"
                ),
                "strongest_lag_days": pathway.get("strongest_lag_days"),
                "relevant_to_local_hdd": (
                    source in important_ids and target in important_ids
                ),
                "touches_local_hdd": (
                    source in important_ids or target in important_ids
                ),
            }
        )

    return {
        "title": "Fleet Temporal Dependency Network with Local HDD Overlay",
        "disclaimer": (
            "Edges are static fleet-level PCMCI temporal dependencies. "
            "Highlighted nodes/edges correspond to factors locally "
            "important for this uploaded HDD's predictive explanation; "
            "PCMCI is never recomputed from a single drive."
        ),
        "nodes": list(nodes.values()),
        "edges": edges,
    }


def _dynamic_root_cause_pathway(
    ranking: List[Dict[str, Any]],
    pathways: List[Dict[str, Any]],
    risks: Dict[str, Any],
) -> Dict[str, Any]:
    """Root-cause-oriented pathway built from local SHAP + fleet evidence."""

    important = _locally_important_factors(ranking)

    important_ids = {entry["factor_id"] for entry in important}

    primary = risks.get("30d") or next(iter(risks.values()), None)

    flagged = bool(primary.get("flagged")) if primary else False

    # Prefer a fleet PCMCI edge where both endpoints are local-important.
    chosen_edge = None

    for pathway in pathways:
        source = pathway.get("source_factor_id")
        target = pathway.get("target_factor_id")
        if source in important_ids and target in important_ids:
            chosen_edge = pathway
            break

    # Otherwise use the strongest local-important factor's own pathway.
    if chosen_edge is None and important:
        lead_id = important[0]["factor_id"]
        for pathway in pathways:
            if (
                pathway.get("source_factor_id") == lead_id
                or pathway.get("target_factor_id") == lead_id
            ):
                chosen_edge = pathway
                break

    steps = []

    edge_source = chosen_edge.get("source_factor_id") if chosen_edge else None
    edge_target = chosen_edge.get("target_factor_id") if chosen_edge else None

    ordered_ids = []

    if edge_source:
        ordered_ids.append(edge_source)
    if edge_target:
        ordered_ids.append(edge_target)

    for entry in important:
        if entry["factor_id"] not in ordered_ids:
            ordered_ids.append(entry["factor_id"])

    local_by_id = {entry["factor_id"]: entry for entry in ranking}

    for factor_id in ordered_ids:

        entry = local_by_id.get(factor_id)

        if entry is None:
            continue

        fleet = entry["fleet_evidence"]

        temporal_support = (
            [p for p in fleet.get("temporal_pathways", [])] != []
            or bool(fleet.get("strongest_temporal_relation"))
        )

        matched_support = (
            fleet.get("matched_risk_difference_pp") is not None
        )

        steps.append(
            {
                "factor_id": factor_id,
                "factor_name": entry["factor_name"],
                "local_contribution_pct": entry["local_shap_share_pct"],
                "local_direction": entry["local_direction"],
                "temporal_support": temporal_support,
                "matched_risk_support": matched_support,
                "matched_risk_difference_pp": fleet.get(
                    "matched_risk_difference_pp"
                ),
                "evidence_classification": fleet.get(
                    "integrated_classification"
                ),
            }
        )

    active = bool(flagged and any(
        step["local_contribution_pct"] >= LOCAL_IMPORTANCE_SHARE_PCT
        and (step["temporal_support"] or step["matched_risk_support"])
        for step in steps
    ))

    if active:
        summary = (
            "Root-cause-oriented pathway active: locally important "
            "factors for this HDD also carry fleet-level temporal or "
            "matched observational support."
        )
    else:
        summary = (
            "Root-cause-oriented pathway weak/inactive for this HDD: "
            "no locally important factor carries strong fleet-level "
            "temporal or matched support, or the prediction is not "
            "flagged. No root cause is claimed."
        )

    return {
        "active": active,
        "flagged": flagged,
        "summary": summary,
        "steps": steps,
        "edge": (
            {
                "source_factor_id": edge_source,
                "source_factor_name": FACTOR_NAMES.get(edge_source, edge_source),
                "target_factor_id": edge_target,
                "target_factor_name": FACTOR_NAMES.get(edge_target, edge_target),
                "strongest_detection_rate": chosen_edge.get(
                    "strongest_detection_rate"
                ),
                "strongest_sign_consistency": chosen_edge.get(
                    "strongest_sign_consistency"
                ),
                "strongest_lag_days": chosen_edge.get("strongest_lag_days"),
            }
            if chosen_edge
            else None
        ),
        "scope": (
            "Local contribution is the drive's local SHAP share; temporal "
            "and matched risk support are reused fleet-level observational "
            "estimates, not individual causal effects."
        ),
    }


def _local_evidence_class_summary(
    ranking: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Distribution of locally important factors by evidence class."""

    important = _locally_important_factors(ranking)

    counts: Dict[str, int] = {}

    factors = []

    for entry in important:
        classification = (
            entry["fleet_evidence"].get("integrated_classification")
            or "PREDICTIVE_ONLY"
        )
        counts[classification] = counts.get(classification, 0) + 1
        factors.append(
            {
                "factor_id": entry["factor_id"],
                "factor_name": entry["factor_name"],
                "local_shap_share_pct": entry["local_shap_share_pct"],
                "evidence_classification": classification,
            }
        )

    order = [
        "HIGH_PRIORITY_ROOT_CAUSE_CANDIDATE",
        "SUPPORTING_DEGRADATION_FACTOR",
        "EXPLORATORY_ROOT_CAUSE_CANDIDATE",
        "PREDICTIVE_TEMPORAL_CONTEXT",
        "PREDICTIVE_SIGNAL_REQUIRES_SEMANTIC_REVIEW",
        "PREDICTIVE_ONLY",
    ]

    classes = [
        {"classification": name, "count": counts.get(name, 0)}
        for name in order
        if counts.get(name, 0) > 0
    ]

    for name, count in counts.items():
        if name not in order:
            classes.append({"classification": name, "count": count})

    return {
        "total_local_factors": len(important),
        "classes": classes,
        "factors": factors,
    }


def _build_visual_payload(
    df: pd.DataFrame,
    serial: str,
    ranking: List[Dict[str, Any]],
    pathways: List[Dict[str, Any]],
    risks: Dict[str, Any],
) -> Dict[str, Any]:

    important = _locally_important_factors(ranking)

    importance_flags = {
        entry["factor_id"]: True for entry in important
    }

    local_bars = []

    for entry in ranking[:10]:
        fleet = entry["fleet_evidence"]
        local_bars.append(
            {
                "factor_id": entry["factor_id"],
                "factor_name": entry["factor_name"],
                "local_shap_share_pct": entry["local_shap_share_pct"],
                "local_direction": entry["local_direction"],
                "pcmci_support": bool(
                    fleet.get("temporal_pathways")
                    or fleet.get("strongest_temporal_relation")
                ),
                "matched_att_support": (
                    fleet.get("matched_risk_difference_pp") is not None
                ),
                "integrated_evidence_class": fleet.get(
                    "integrated_classification"
                ),
                "locally_important": entry["factor_id"]
                in importance_flags,
            }
        )

    return {
        "local_factor_bars": local_bars,
        "att_overlay": _att_overlay(ranking),
        "pcmci_overlay": _pcmci_overlay(pathways, ranking),
        "dynamic_root_cause_pathway": _dynamic_root_cause_pathway(
            ranking, pathways, risks
        ),
        "local_evidence_class_summary": _local_evidence_class_summary(
            ranking
        ),
        "smart_history_traces": _smart_history_traces(
            df, serial, ranking
        ),
        "rules": {
            "pcmci_recomputed": False,
            "att_recomputed": False,
            "note": (
                "Fleet PCMCI coefficients and matched ATT values are "
                "reused unchanged; only local overlay flags are added."
            ),
        },
    }


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

    evidence_index = _fleet_evidence_index()

    try:
        fleet_pathways = component3_service.get_temporal_pathways()
    except HTTPException:
        fleet_pathways = []

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

        local_ranking = _local_factor_ranking(
            30, row_df, evidence_index
        )

        local_explanation = _local_root_cause_explanation(
            serial,
            as_of_date,
            risks,
            local_ranking,
        )

        visual_payload = _build_visual_payload(
            df,
            serial,
            local_ranking,
            fleet_pathways,
            risks,
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
                "local_factor_ranking": local_ranking,
                "local_root_cause_explanation": local_explanation,
                "visual_analysis": visual_payload,
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
            "dynamic_scope": (
                "Per-drive dynamic: risk scores, local SHAP, local "
                "factor ranking, local root-cause narrative."
            ),
            "static_scope": (
                "Fleet-level research evidence (PCMCI pathways, matched "
                "ATT, evidence catalog, figures) is precomputed and reused "
                "read-only; it is never recomputed on upload."
            ),
            "note": (
                "Risk scores come from an out-of-time validated model "
                "(trained Aug-Sep 2025). Absolute probabilities are "
                "conservative; use ranking and driver explanation, "
                "not the raw percentage alone."
            ),
        },
    }
