
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "digital_twin_rul"

SUMMARY_FILE = RESULTS_DIR / "heldout_adaptive_lead_time_summary.csv"
HDD_RESULTS_FILE = RESULTS_DIR / "heldout_test_adaptive_digital_twin_results.csv"
LEAD_TIME_FILE = RESULTS_DIR / "heldout_adaptive_lead_time_results.csv"

st.set_page_config(
    page_title="HDD Digital Twin Monitor",
    page_icon="💽",
    layout="wide",
)

st.title("HDD Digital Twin Monitoring Dashboard")
st.caption("Adaptive degradation monitoring and RUL estimation")


@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


summary = load_csv(SUMMARY_FILE)
hdd_results = load_csv(HDD_RESULTS_FILE)
lead_times = load_csv(LEAD_TIME_FILE)

# 1. Validation overview
st.header("1. Held-out Warning Validation")

if not summary.empty:
    metrics = summary.iloc[0]
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Test HDDs", int(metrics["test_hdds_in_results"]))
    col2.metric("Warnings detected", int(metrics["detected_within_60_days"]))
    col3.metric("Detection rate", f'{metrics["detection_rate_percent"]:.2f}%')
    col4.metric("Median lead time", f'{metrics["median_lead_time_days"]:.1f} days')

    st.caption("Historical held-out test results, not live monitoring.")

# 2. Detection status chart
st.header("2. Detection Status")

if not lead_times.empty and "status" in lead_times.columns:
    status = lead_times["status"].replace({
        "detected_within_60_days": "Warning detected",
        "not_detected": "No warning detected",
    }).value_counts()

    st.bar_chart(status)
    st.dataframe(
        status.rename_axis("Detection status")
        .reset_index(name="Number of HDDs"),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info("Detection status data is unavailable.")

# 3. Lead-time chart
st.header("3. Warning Lead-Time Distribution")

if not lead_times.empty and "lead_time_days" in lead_times.columns:
    lead_times["lead_time_days"] = pd.to_numeric(
        lead_times["lead_time_days"], errors="coerce"
    )

    detected = lead_times[
        lead_times["lead_time_days"].between(0, 60)
    ].copy()

    if not detected.empty:
        bins = [0, 3, 7, 14, 30, 60]
        labels = ["0–3 days", "4–7 days", "8–14 days", "15–30 days", "31–60 days"]

        detected["Lead-time range"] = pd.cut(
            detected["lead_time_days"],
            bins=[-0.01, 3, 7, 14, 30, 60],
            labels=labels,
            include_lowest=True,
        )

        distribution = (
            detected["Lead-time range"]
            .value_counts(sort=False)
            .reindex(labels, fill_value=0)
        )

        st.bar_chart(distribution)
        st.dataframe(
            distribution.rename_axis("Lead time")
            .reset_index(name="Number of HDDs"),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No valid lead-time records were found.")

# 4. Adaptive health states
st.header("4. Adaptive Health-State Distribution")

if not hdd_results.empty and "adaptive_state" in hdd_results.columns:
    states = hdd_results["adaptive_state"].fillna("Unknown").value_counts()
    st.bar_chart(states)
    st.dataframe(
        states.rename_axis("Health state")
        .reset_index(name="Number of HDDs"),
        hide_index=True,
        use_container_width=True,
    )

    st.caption("Health states are experimental rule-based classifications.")

# 5. Individual HDD results
st.header("5. Individual HDD Results")

if not hdd_results.empty:
    hdd_results = hdd_results.copy()
    hdd_results["serial_number"] = hdd_results["serial_number"].astype(str)

    serials = sorted(hdd_results["serial_number"].unique())
    serial = st.selectbox("Select HDD serial number", serials)

    row = hdd_results[hdd_results["serial_number"] == serial].iloc[0]

    col1, col2, col3 = st.columns(3)

    predicted = pd.to_numeric(
        pd.Series([row.get("predicted_rul_days")]), errors="coerce"
    ).iloc[0]
    actual = pd.to_numeric(
        pd.Series([row.get("actual_rul_days")]), errors="coerce"
    ).iloc[0]
    error = pd.to_numeric(
        pd.Series([row.get("absolute_error_days")]), errors="coerce"
    ).iloc[0]

    col1.metric(
        "Predicted RUL",
        f"{predicted:.2f} days" if pd.notna(predicted) else "Unavailable",
    )
    col2.metric(
        "Actual RUL",
        f"{actual:.2f} days" if pd.notna(actual) else "Unavailable",
    )
    col3.metric(
        "Absolute error",
        f"{error:.2f} days" if pd.notna(error) else "Unavailable",
    )

    st.subheader("Adaptive Health State")
    st.write(f'**State:** {row.get("adaptive_state", "Unknown")}')

    col1, col2, col3 = st.columns(3)

    col1.metric("Current persistence", row.get("current_degradation_persistence", "N/A"))
    col2.metric("Longest persistence", row.get("longest_degradation_persistence", "N/A"))

    available = row.get("health_features_available")
    col3.metric(
        "Available health features",
        f"{available}/5" if pd.notna(available) else "N/A",
    )

    st.write(f'**Latest pre-failure observation:** {row.get("latest_pre_failure_date", "N/A")}')
    st.write(f'**Recorded failure date:** {row.get("failure_date", "N/A")}')

    columns = [
        "serial_number",
        "adaptive_state",
        "predicted_rul_days",
        "actual_rul_days",
        "absolute_error_days",
        "health_features_available",
    ]
    columns = [c for c in columns if c in hdd_results.columns]

    st.dataframe(
        hdd_results[columns],
        hide_index=True,
        use_container_width=True,
    )

st.divider()
st.caption(
    "Research prototype | Historical Backblaze Q1 2023 data | "
    "Predictions and warning states are experimental."
)
