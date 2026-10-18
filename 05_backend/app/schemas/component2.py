"""Pydantic schemas for Component 2: Adaptive Digital Twin.

Every field maps to a column produced by the component's held-out
validation artifacts in ``02_component_2_digital_twin/results/digital_twin_rul``.
"""

from typing import List, Optional

from pydantic import BaseModel


class DigitalTwinRequest(BaseModel):
    serial_number: str


class DeployedConfig(BaseModel):
    """Detector configuration actually deployed in the component."""

    config: str
    threshold_table: str
    min_severity: int
    persistence: int
    matches_production: bool


class OverviewSummary(BaseModel):
    """Held-out lead-time validation summary (per failed HDD)."""

    test_hdds: int
    detected: int
    not_detected: int
    detection_rate_percent: float
    persistence_observations: int
    max_lead_time_window_days: int
    mean_lead_time_days: float
    median_lead_time_days: float
    minimum_lead_time_days: float
    maximum_lead_time_days: float
    median_persistence_span_days: float
    maximum_persistence_span_days: float


class FalseAlarmStats(BaseModel):
    """False-alarm rate on the disjoint healthy control sample (test)."""

    healthy_hdds: int
    healthy_false_alarms: int
    healthy_false_alarm_rate_percent: float


class Overview(BaseModel):
    summary: OverviewSummary
    false_alarms: FalseAlarmStats
    config: DeployedConfig


class DetectionRecord(BaseModel):
    """One failed HDD from the held-out lead-time evaluation."""

    serial_number: str
    failure_date: str
    pre_failure_observations: int
    first_detection_date: Optional[str]
    lead_time_days: Optional[float]
    persistence_span_days: Optional[float]
    longest_persistence: int
    detected_within_60_days: bool
    status: str


class LeadTimeBand(BaseModel):
    lead_time_band: str
    hdd_count: int
    percentage_of_detections: float


class SweepConfig(BaseModel):
    """One row of the detection-threshold sweep (tuning set)."""

    config: str
    threshold_table: str
    use_14d: bool
    min_severity: int
    persistence: int
    matches_production: bool
    detection_rate_percent: float
    healthy_false_alarm_rate_percent: float
    median_lead_time_days: float
    evaluation_set: str


class ConfigComparisonRow(SweepConfig):
    """Selected-vs-deployed comparison evaluated on the held-out test set."""

    role: str
    failed_hdds: int
    detected_within_60_days: int
    healthy_hdds: int
    healthy_false_alarms: int


class RulByState(BaseModel):
    """RUL prediction accuracy grouped by adaptive twin state."""

    adaptive_state: str
    n_predictions: int
    n_hdds: int
    mae_days: float
    rmse_days: float
    bias_days: float
    median_absolute_error_days: float
    mean_actual_rul_days: float
    mean_predicted_rul_days: float


class RulTrajectoryPoint(BaseModel):
    """RUL accuracy at a checkpoint (observations before failure)."""

    checkpoint_observations: int
    n_predictions: int
    n_hdds: int
    mae: float
    rmse: float
    bias: float


class RulMetrics(BaseModel):
    by_state: List[RulByState]
    trajectory: List[RulTrajectoryPoint]


class DigitalTwinResponse(BaseModel):
    """Per-HDD digital twin detail for a single serial number."""

    serial_number: str
    failure_date: Optional[str]
    adaptive_state: str
    actual_rul_days: Optional[float]
    predicted_rul_days: Optional[float]
    absolute_error_days: Optional[float]
    current_degradation_persistence: float
    longest_degradation_persistence: float
    health_features_available: Optional[float]
    detected: bool
    first_detection_date: Optional[str]
    lead_time_days: Optional[float]
    persistence_span_days: Optional[float]
    longest_persistence_run: int
