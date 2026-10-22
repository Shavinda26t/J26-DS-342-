"""
Evaluate the risk-gated, degradation-aware adaptation controller.

Protocol (leakage-free, delayed labels)
---------------------------------------
* The frozen seq_30 BiLSTM produces a raw RUL for every window of the
  validation split (89 drives) and the held-out test split (90 drives).
* Validation windows seed every adaptive method (calibration only).
* Test drives are replayed in failure-date order.  A drive's RUL labels only
  exist once it has failed, so for each drive:
      1. every method predicts all of its windows with its *current* state,
      2. the drive fails -> its labelled windows become one evidence batch,
      3. each method learns / decides from that batch.
  Predictions are therefore always made before the labels are seen.
* Each window is labelled degrading / stable with the production degradation
  detector (combined severity of the 7-observation SMART 1 / SMART 187
  changes >= MIN_SEVERITY), recovered by inverting the stored scaling.

Methods
-------
  frozen            raw BiLSTM, never adapts
  always_adapt      RLS correction updated and applied after every batch
  current_twin      existing DualRulAdapter population path (validated once
                    on the seed hold-out, then updated continuously)
  risk_gated        proposed controller (ADAPT / WAIT / REJECT / ROLLBACK)
  risk_gated_no_degradation, risk_gated_no_drift   ablations

Scenarios
---------
  S0_real                real held-out stream, no intervention
  S1_operating_drift     from drive 30 on the model becomes 10 d over-optimistic
                         and the operating signature (smart_1 volatility) shifts
  S2_noisy_feedback      drives 20-60 report labels with N(0, 20 d) noise
  S4_biased_feedback     drives 20-60 report labels 20 d too late (failure
                         dates logged late) -> learning from them is harmful
  S3_transient_drift     +10 d over-optimism for drives 20-44 only, then the
                         original behaviour returns (tests ROLLBACK)
  S5_mismatched_seed      every adaptive method is seeded on a population whose
                         labels are 10 d higher (twin calibrated on another
                         drive model) -> the seeded correction is harmful
S1-S5 are controlled stress tests on top of the real predictions; evaluation
always uses the true labels.

Null check (empirical validation of the alpha guarantee)
--------------------------------------------------------
On NULL_RUNS synthetic streams the model is perfectly calibrated (label =
raw prediction + symmetric noise), so no correction can help.  The fraction
of runs in which the gate ever ADAPTs must stay <= alpha (0.05).

Outputs: results/risk_gated_adaptation/
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import COMPONENT_ROOT, REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "05_backend"))

from app.ml import calibration_data  # noqa: E402
from app.ml.conformal import data_quality_score  # noqa: E402
from app.ml.online_adapter import DualRulAdapter, OnlineRulAdapter  # noqa: E402
from app.ml.risk_gate import (  # noqa: E402
    ADAPT,
    REJECT,
    ROLLBACK,
    WAIT,
    Outcome,
    RiskGatedAdapter,
)
import degradation_detector as detector  # noqa: E402

OUT_DIR = COMPONENT_ROOT / "results" / "risk_gated_adaptation"
SEED = 42
CRITICAL_RUL = 14
UNSAFE_MARGIN = 14
NULL_RUNS = 200
NULL_NOISE_DAYS = 10.0

# Column indices in the preprocessed seq_30 tensors.
COL_SMART1 = 0
COL_SMART1_CHANGE_7D = 5
COL_SMART1_STD_7 = 7
COL_SMART187_CHANGE_7D = 8
COL_AVAILABLE = slice(11, 16)
COL_VALID = slice(16, 22)


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def load_split(split, predictor, params):
    metadata = calibration_data.load_split_metadata()
    rows = metadata[metadata["split"] == split].reset_index(drop=True)
    X = np.load(calibration_data.preprocessed_dir() / f"X_{split}.npy")
    assert len(X) == len(rows), f"{split}: tensor/metadata mismatch"

    raw = calibration_data._predict_batch(predictor.model, X)
    means = np.array(params["means"])
    stds = np.array(params["stds"])
    last = X[:, -1, :]
    unscale = lambda c: last[:, c] * stds[c] + means[c]  # noqa: E731

    change_1 = unscale(COL_SMART1_CHANGE_7D)
    change_187 = unscale(COL_SMART187_CHANGE_7D)
    available = (last[:, COL_AVAILABLE] * stds[11:16] + means[11:16]).round()

    frame = rows.copy()
    frame["raw_pred"] = raw
    frame["severity"] = [
        detector.combined_severity(a, b) for a, b in zip(change_1, change_187)
    ]
    frame["degrading"] = frame["severity"] >= detector.MIN_SEVERITY
    frame["quality"] = [
        data_quality_score(
            health_features_available=float(a),
            history_length=30,
            history_target=30,
            temporal_valid_fraction=float(v),
        )
        for a, v in zip(available.sum(axis=1), last[:, COL_VALID].mean(axis=1))
    ]
    frame["smart_1_normalized"] = unscale(COL_SMART1)
    frame["smart_1_normalized_rolling_std_7"] = unscale(COL_SMART1_STD_7)
    frame["failure_date"] = pd.to_datetime(frame["failure_date"])
    frame["sequence_end"] = pd.to_datetime(frame["sequence_end"])
    return frame


def outcomes(frame, label_col="label"):
    return [
        Outcome(
            predicted=float(r.raw_pred),
            actual=float(getattr(r, label_col)),
            degrading=bool(r.degrading),
            quality=float(r.quality),
            features={
                "smart_1_normalized": float(r.smart_1_normalized),
                "smart_1_normalized_rolling_std_7": float(
                    r.smart_1_normalized_rolling_std_7
                ),
            },
        )
        for r in frame.itertuples()
    ]


# ------------------------------------------------------------------
# Scenarios (applied on a copy of the real test stream)
# ------------------------------------------------------------------
def apply_scenario(test, name, drive_order, smart1_std_scale):
    frame = test.copy()
    frame["label"] = frame["rul"].astype(float)
    index = {serial: i for i, serial in enumerate(drive_order)}
    position = frame["serial_number"].map(index)
    rng = np.random.default_rng(SEED)

    if name == "S1_operating_drift":
        hit = position >= 30
        frame.loc[hit, "raw_pred"] += 10.0
        frame.loc[hit, "smart_1_normalized_rolling_std_7"] += 1.5 * smart1_std_scale
    elif name == "S2_noisy_feedback":
        hit = (position >= 20) & (position < 60)
        noise = rng.normal(0.0, 20.0, size=int(hit.sum()))
        frame.loc[hit, "label"] = np.maximum(0.0, frame.loc[hit, "label"] + noise)
    elif name == "S4_biased_feedback":
        hit = (position >= 20) & (position < 60)
        frame.loc[hit, "label"] = frame.loc[hit, "label"] + 20.0
    elif name == "S3_transient_drift":
        hit = (position >= 20) & (position < 45)
        frame.loc[hit, "raw_pred"] += 10.0
        frame.loc[hit, "smart_1_normalized_rolling_std_7"] += 1.5 * smart1_std_scale
    return frame


# ------------------------------------------------------------------
# Methods
# ------------------------------------------------------------------
class Frozen:
    def predict(self, o):
        return o.predicted

    def learn(self, batch):
        return None


class AlwaysAdapt:
    def __init__(self, seed_outcomes, center, scale):
        self.adapter = OnlineRulAdapter(forgetting=0.995, center=center, scale=scale)
        self.adapter.validated = True
        for o in seed_outcomes:
            self.adapter.update(o.predicted, o.actual)

    def predict(self, o):
        return self.adapter.adapt(o.predicted)

    def learn(self, batch):
        for o in batch:
            self.adapter.update(o.predicted, o.actual)


class CurrentTwin:
    """Population path of the deployed DualRulAdapter."""

    def __init__(self, seed_pairs, seed_serials):
        self.dual = DualRulAdapter()
        self.diagnostics = self.dual.seed_population(seed_pairs, seed_serials)

    def predict(self, o):
        return self.dual.global_adapter.adapt(o.predicted)

    def learn(self, batch):
        for o in batch:
            self.dual.global_adapter.update(o.predicted, o.actual)


class RiskGated:
    def __init__(self, seed_batches, center, scale, **flags):
        self.gate = RiskGatedAdapter(center=center, scale=scale, **flags)
        self.gate.seed(seed_batches)

    def predict(self, o):
        return self.gate.apply(o.predicted, o.degrading)["adapted"]

    def learn(self, batch):
        return self.gate.observe(batch)


# ------------------------------------------------------------------
def run(test_frame, drive_order, methods):
    rows, decisions = [], []
    for step, serial in enumerate(drive_order):
        drive = test_frame[test_frame["serial_number"] == serial].sort_values(
            "sequence_end"
        )
        batch = outcomes(drive)
        for name, method in methods.items():
            for o, r in zip(batch, drive.itertuples()):
                rows.append({
                    "method": name,
                    "step": step,
                    "serial_number": serial,
                    "actual_rul": float(r.rul),
                    "raw_pred": o.predicted,
                    "pred": float(method.predict(o)),
                    "degrading": o.degrading,
                })
        # The drive fails: its (possibly noisy) labels become evidence.
        for name, method in methods.items():
            decision = method.learn(batch)
            if isinstance(decision, dict):
                decisions.append({"method": name, "step": step,
                                  "serial_number": serial, **{
                                      k: v for k, v in decision.items()
                                      if k != "evidence_before_decision"},
                                  **decision["evidence_before_decision"]})
    return pd.DataFrame(rows), pd.DataFrame(decisions)


def null_check(test, drive_order, validation, center, scale):
    """Empirical false-acceptance rate when no correction can help."""
    rng = np.random.default_rng(SEED)
    drives = [
        test[test["serial_number"] == s].sort_values("sequence_end")
        for s in drive_order
    ]
    rows = []
    for run_id in range(NULL_RUNS):
        def calibrated(frame):
            frame = frame.copy()
            noise = rng.normal(0.0, NULL_NOISE_DAYS, size=len(frame))
            frame["label"] = np.maximum(0.0, frame["raw_pred"] + noise)
            return frame

        seed_batches = [
            outcomes(calibrated(g)) for _, g in validation.groupby("serial_number")
        ]
        gate = RiskGated(seed_batches, center, scale)
        always = AlwaysAdapt([o for b in seed_batches for o in b], center, scale)
        ever, err_gate, err_always, err_frozen = False, [], [], []
        for drive in drives:
            batch = outcomes(calibrated(drive))
            for o in batch:
                err_gate.append(abs(gate.predict(o) - o.actual))
                err_always.append(abs(always.predict(o) - o.actual))
                err_frozen.append(abs(o.predicted - o.actual))
            ever |= gate.learn(batch)["action"] == ADAPT
            always.learn(batch)
        rows.append({
            "run": run_id,
            "ever_adapted": ever,
            "gate_mae_delta": float(np.mean(err_gate) - np.mean(err_frozen)),
            "always_adapt_mae_delta": float(
                np.mean(err_always) - np.mean(err_frozen)
            ),
        })
    return pd.DataFrame(rows)


def summarise(pred, decisions, scenario):
    out = []
    for name, group in pred.groupby("method", sort=False):
        err = group["pred"] - group["actual_rul"]
        critical = group["actual_rul"] <= CRITICAL_RUL
        crit_err = err[critical]
        degr = group["degrading"]
        row = {
            "scenario": scenario,
            "method": name,
            "windows": len(group),
            "mae_days": float(err.abs().mean()),
            "rmse_days": float(np.sqrt((err ** 2).mean())),
            "bias_days": float(err.mean()),
            "critical_mae_days": float(crit_err.abs().mean()),
            "unsafe_overestimate_rate_pct": float(
                (crit_err > UNSAFE_MARGIN).mean() * 100.0
            ),
            "degrading_mae_days": float(err[degr].abs().mean()),
        }
        d = decisions[decisions["method"] == name] if len(decisions) else decisions
        for action in (ADAPT, WAIT, REJECT, ROLLBACK):
            row[f"n_{action.lower()}"] = (
                int((d["action"] == action).sum()) if len(d) else None
            )
        if len(d):
            row["drift_detected_pct"] = float(d["drift_detected"].mean() * 100.0)
        out.append(row)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from app.ml.component2_rul_predictor import Component2RULPredictor

    predictor = Component2RULPredictor(
        model_path=calibration_data.checkpoint_path(),
        preprocessing_path=calibration_data.preprocessing_path(),
    )
    with calibration_data.preprocessing_path().open() as handle:
        params = json.load(handle)

    print("Predicting validation (seed) and test windows ...")
    validation = load_split("validation", predictor, params)
    validation["label"] = validation["rul"].astype(float)
    test = load_split("test", predictor, params)
    assert not set(validation.serial_number) & set(test.serial_number)

    seed = outcomes(validation)
    seed_serials = validation["serial_number"].astype(str).tolist()
    center = float(np.median([o.predicted for o in seed]))
    scale = float(max(np.std([o.predicted for o in seed]), 1e-3))
    smart1_std_scale = float(
        validation.loc[~validation.degrading, "smart_1_normalized_rolling_std_7"].std()
    )

    drive_order = (
        test.groupby("serial_number")["failure_date"].first()
        .sort_values(kind="stable").index.tolist()
    )

    print(f"seed windows={len(seed)} drives={validation.serial_number.nunique()} "
          f"degrading={validation.degrading.mean():.1%}")
    print(f"test windows={len(test)} drives={len(drive_order)} "
          f"degrading={test.degrading.mean():.1%}")

    scenarios = ["S0_real", "S1_operating_drift", "S2_noisy_feedback",
                 "S3_transient_drift", "S4_biased_feedback", "S5_mismatched_seed"]
    summary, all_decisions, all_preds = [], [], []
    for scenario in scenarios:
        frame = apply_scenario(test, scenario, drive_order, smart1_std_scale)
        seed_frame = validation.copy()
        if scenario == "S5_mismatched_seed":
            seed_frame["label"] = seed_frame["label"] + 10.0
        seed = outcomes(seed_frame)
        seed_pairs = [(o.predicted, o.actual) for o in seed]
        seed_batches = [
            outcomes(group.sort_values("sequence_end"))
            for _, group in seed_frame.groupby("serial_number")
        ]
        methods = {
            "frozen": Frozen(),
            "always_adapt": AlwaysAdapt(seed, center, scale),
            "current_twin": CurrentTwin(seed_pairs, seed_serials),
            "risk_gated": RiskGated(seed_batches, center, scale),
            "risk_gated_no_degradation": RiskGated(
                seed_batches, center, scale, degradation_aware=False),
            "risk_gated_no_drift": RiskGated(
                seed_batches, center, scale, drift_aware=False),
        }
        if scenario == "S0_real":
            print("current_twin seed gate:",
                  methods["current_twin"].diagnostics.get("status"),
                  "gain", methods["current_twin"].diagnostics.get("holdout_gain"))
        pred, decisions = run(frame, drive_order, methods)
        pred.insert(0, "scenario", scenario)
        if len(decisions):
            decisions.insert(0, "scenario", scenario)
        summary.extend(summarise(pred, decisions, scenario))
        all_decisions.append(decisions)
        all_preds.append(pred)

    null = null_check(test, drive_order, validation, center, scale)
    null.to_csv(OUT_DIR / "risk_gated_null_check.csv", index=False)

    summary = pd.DataFrame(summary)
    decisions = pd.concat(all_decisions, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)

    summary.to_csv(OUT_DIR / "risk_gated_summary.csv", index=False)
    decisions.to_csv(OUT_DIR / "risk_gated_decisions.csv", index=False)
    preds.to_csv(OUT_DIR / "risk_gated_predictions.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["scenario", "method", "mae_days", "critical_mae_days",
            "unsafe_overestimate_rate_pct", "bias_days",
            "n_adapt", "n_wait", "n_reject", "n_rollback", "drift_detected_pct"]
    print(summary[cols].round(2).to_string(index=False))
    rate = null["ever_adapted"].mean()
    print(f"\nNull check ({len(null)} runs, calibrated model, no update can help):")
    print(f"  risk_gated ever ADAPTed in {rate:.1%} of runs (guarantee <= 5%), "
          f"mean MAE change vs frozen {null['gate_mae_delta'].mean():+.3f} d")
    print(f"  always_adapt applies its update in 100% of runs, mean MAE change "
          f"vs frozen {null['always_adapt_mae_delta'].mean():+.3f} d")
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
