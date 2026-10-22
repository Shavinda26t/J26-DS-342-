# Component 2 — Complete Pipeline Flow
**Adaptive AI-Driven Digital Twin for HDD Health, Degradation & RUL**

78 scripts in `scripts/`, executed in 10 phases. Every phase consumes the previous
phase's artifacts from `data/` and writes evidence to `results/`.

```
RAW SMART DATA
   │
   ▼
[1] AUDIT & EXPLORE ──► [2] FEATURES & HEALTH STATE ──► [3] DEGRADATION DETECTOR
                                                                 │
                                                                 ▼
                                              [4] RUL MODEL (BiLSTM, seq_30)
                                                                 │
                                                                 ▼
                                              [5] HELD-OUT VALIDATION
                                                                 │
                              ┌──────────────────────────────────┴───────────────────┐
                              ▼                                                      ▼
                     [6] LIVE ADAPTIVE TWIN (backend runtime)          [7] RISK-GATED ADAPTATION
                              │                                                      │
                              └──────────────────────┬───────────────────────────────┘
                                                     ▼
                                          [8] API + DASHBOARD
                                                     ▼
                                          [9] TESTS  /  [10] DOCUMENTATION
```

---

## Phase 1 — Data Audit & Exploration
**Purpose:** understand the data before modelling; find which SMART attributes
actually move before a failure.

| Script | What it does | Key output |
|---|---|---|
| `inspect_smart_columns.py` | Lists raw columns, dtypes, scale of the Backblaze export | column inventory |
| `analyze_smart_quality.py` | Missing values, zeros, impossible values, per-attribute quality | data-quality report |
| `analyze_failure_history.py` | Which drives failed, when, how many, fleet base rates | failure base rates |
| `analyze_hdd_observation_gaps.py` | Gaps between daily observations (real telemetry is irregular) | gap statistics |
| `analyze_feature_redundancy.py` | Correlation between SMART attributes → drop redundant ones | redundancy matrix |
| `analyze_individual_degradation.py` | Single-drive degradation curves | per-drive curves |

**Decision output:** SMART 1, 5, 187, 197, 198 are the failure-relevant set; the
rest are noise or highly redundant.

---

## Phase 2 — Feature Engineering & Health-State Construction
**Purpose:** turn raw SMART rows into a normalised *health state* the twin can track.

| Script | What it does | Output |
|---|---|---|
| `build_temporal_features.py` | Rolling windows, deltas, rates of change | `data/features/` |
| `analyze_smart1_trajectory.py` | SMART 1 (raw read error rate) behaviour over life | trajectory insights |
| `analyze_smart5_trajectory.py` | SMART 5 (reallocated sectors) behaviour | trajectory insights |
| `analyze_smart187_trajectory.py` | SMART 187 (reported uncorrectable) behaviour | trajectory insights |
| `analyze_smart197_trajectory.py` | SMART 197 (current pending sector) behaviour | trajectory insights |
| `analyze_smart198_trajectory.py` | SMART 198 (offline pending sector) behaviour | trajectory insights |
| `build_candidate_health_indices.py` | Several candidate composite health indices | candidate indices |
| `evaluate_health_state_candidates.py` | Scores each candidate (separation failed vs healthy) | ranked candidates |
| `select_health_state_features.py` | Picks the final feature subset | selected features |
| `normalize_health_state_features.py` | Fits normalisation on train only | scaler params |
| `rebuild_health_state_normalization.py` | Rebuild after edits (no leakage) | refreshed scaler |
| `build_digital_twin_state.py` → `_v5.py` | **v1→v5 iteration** of the twin state table (v5 = production) | `data/processed/digital_twin_state*` |
| `check_full_temporal_variation.py` | Verifies every window has valid history | validation report |
| `analyze_health_state_distributions.py` | State distribution failed vs healthy | distributions |
| `analyze_health_scaling.py` | Scaling behaviour per attribute | scaling report |

**Why v1→v5:** each version fixed a defect found by the next validation script
(e.g. missing temporal continuity, leaky scaling). The version trail *is* the
methodology.

---

## Phase 3 — Degradation Detection & Streaming State
**Purpose:** detect "this drive is degrading" and keep live per-drive state.

| Script | What it does | Output |
|---|---|---|
| `degradation_detector.py` | Core detector: severity from SMART 1 / 187 changes | detector logic |
| `detect_adaptive_degradation.py` | Per-drive adaptive severity + streak logic | adaptive detector |
| `digital_twin_state_manager.py` | **The runtime class**: holds state, severity, streak, history per serial | reusable module |
| `analyze_degradation_thresholds.py` | Where to put severity cut-offs | threshold analysis |
| `tune_detection_thresholds.py` | Sweeps severity × persistence → detection/FA trade-off | `threshold_sweep_*.csv` |
| `analyze_degradation_persistence.py` | How many consecutive observations confirm degradation | persistence analysis |
| `analyze_persistence_lead_time.py` | Lead time gained per persistence level | lead-time table |
| `compare_persistence_levels.py` | Persistence level comparison | comparison CSV |
| `compare_healthy_persistence_levels.py` | Same, on healthy controls (false-alarm view) | healthy comparison |
| `compare_failed_healthy_state.py` | Failed vs healthy state separation | separation table |
| `analyze_degradation_trajectory.py` | Degradation path over drive life | trajectory stats |
| `adaptive_digital_twin.py` / `adaptive_digital_twin_system.py` | End-to-end adaptive twin prototypes (research stage) | prototypes |

**Key output:** `threshold_sweep_best.csv` → deployed config
(`use_14d`, severity ≥ 2, persistence 4) — **39.9 % detection within 60 days at
19.75 % healthy false-alarm rate**.

---

## Phase 4 — RUL Modelling (BiLSTM)
**Purpose:** predict Remaining Useful Life in days from the health-state sequence.

| Script | What it does | Output |
|---|---|---|
| `build_rul_sequences.py` | Sliding windows → (sequence, RUL label) pairs | sequence dataset |
| `split_rul_dataset.py` | Train / validation / held-out test split (no drive overlap) | `data/splits/` |
| `preprocess_rul_sequences.py` | Scale sequences, build tensors | preprocessed tensors |
| `check_rul_nan_split.py` | Leakage / NaN check on the splits | sanity report |
| `train_rul_lstm.py` | Baseline LSTM | `results/rul_lstm/` |
| `train_rul_lstm_huber.py` | Huber loss (robust to outliers) | Huber variant |
| `train_rul_weighted_lstm.py` | Loss weighted toward critical windows | weighted variant |
| `build_sequence_length_datasets.py` | Datasets for window lengths 7 / 14 / 21 / 30 | 4 datasets |
| `preprocess_sequence_length_experiments.py` | Preprocess each | 4 preprocessed sets |
| `train_sequence_length_experiments.py` | Compare all window lengths | length comparison |
| `train_rul_bilstm.py` | BiLSTM (forward + backward over the window) | BiLSTM v1 |
| `train_rul_bilstm_v2.py` | Improved BiLSTM → **seq_30 production model** | `models/trained/` |
| `build_raw_smart_experiment.py` | Alternative: raw SMART without engineered features | raw dataset |
| `train_raw_smart_bilstm.py` | Train the raw-SMART comparison model | comparison model |
| `diagnose_rul_prediction_sensitivity.py` | Which inputs move the prediction | sensitivity report |
| `compare_repeated_rul_inputs.py` | Determinism check on identical inputs | determinism report |
| `analyze_seq30_rul_errors.py` | Error breakdown of the chosen model | error analysis |

**Winner:** seq_30 BiLSTM (30-observation lookback, bidirectional).

---

## Phase 5 — Validation & Evidence
**Purpose:** prove it works on drives never seen in training.

| Script | What it does | Output |
|---|---|---|
| `test_digital_twin_state_manager.py` | State manager unit checks | pass/fail |
| `test_adaptive_digital_twin.py` | Integrated adaptive-twin checks | pass/fail |
| `test_directional_health_state.py` | Health state moves the right direction | pass/fail |
| `test_rul_trajectory.py` | RUL over a full drive life | trajectory |
| `test_v5_rul_trajectory.py` | Same on the production v5 state | trajectory |
| `test_real_hdd_rul.py` | RUL on real failed drives | per-drive RUL |
| `validate_all_eligible_hdds_rul.py` | Every eligible failed HDD | `all_eligible_hdd_*` |
| `validate_multihdd_rul.py` | Multi-HDD checkpoint metrics | `multihdd_rul_*` |
| `validate_heldout_rul_trajectory.py` | **Held-out trajectory test** | `heldout_rul_trajectory_metrics.csv` |
| `validate_adaptive_degradation.py` | Adaptive detector validation | adaptive validation |
| `validate_adaptive_digital_twin_test_set.py` | Full twin on the test split | `heldout_test_adaptive_digital_twin_results.csv` |
| `validate_heldout_adaptive_lead_time.py` | Lead time to failure, held out | `heldout_adaptive_lead_time_*` |
| `validate_healthy_degradation.py` | Healthy-control false-alarm check | `healthy_degradation_*` |
| `validate_health_state_trajectory.py` | Health state trajectory check | trajectory check |
| `inspect_rul_validation_coverage.py` | Are enough windows covered? | coverage report |
| `analyze_rul_uncertainty.py` | Error percentiles (mae, p75, p90, p95, p99) | `heldout_rul_uncertainty_summary.csv` |
| `analyze_rul_uncertainty_by_state.py` | Uncertainty split by health state | `heldout_rul_uncertainty_by_state.csv` |

**Headline numbers (held-out, 90 drives):**

| Checkpoint | MAE (d) | Bias (d) | Notes |
|---|---|---|---|
| 30 obs | 9.38 | −7.90 | under-predicts far from failure |
| 14 obs | **7.03** | +6.91 | best accuracy |
| 7 obs | 12.45 | +12.45 | over-predicts near failure |
| 1 obs | 16.56 | +16.56 | over-predicts near failure |

Detection: **35.0 %** of failed drives detected within 60 days on held-out
(median lead time 15 days).

---

## Phase 6 — Live Adaptive Twin (backend runtime)
**Purpose:** run the research offline artifacts as a live service.

| Backend module | Responsibility |
|---|---|
| `app/ml/_component2_paths.py` | Single place that resolves component-2 artifacts (detector, state manager) |
| `app/ml/adaptive_thresholds.py` | **Feature 3** — per-drive, noise-scaled thresholds; never looser than global |
| `app/ml/online_adapter.py` | **Feature 1** — RLS residual corrector (personal + population) on the frozen BiLSTM |
| `app/ml/conformal.py` | **Feature 2** — split-conformal interval + data-quality widening + confidence score |
| `app/ml/calibration_data.py` | Leakage-free calibration set (2 925 held-out residuals) |
| `app/ml/forecaster.py` | **Feature 4** — multi-horizon health forecast + failure ETA |
| `app/ml/warning_decision.py` | Combines adaptive + global warnings, never suppresses a global warning |
| `app/ml/risk_gate.py` | **Novelty** — ADAPT / WAIT / REJECT / ROLLBACK controller (e-values) |
| `app/services/twin_sync_service.py` | The integrator: validate → dedup → compute → persist, plus `learn_outcome` feedback |

**Runtime flow of one observation (`POST /api/v1/component2/twin/sync`):**

```
observation
   → validate (422 on malformed, twin untouched)
   → dedup (same serial+date ignored)
   → state manager update (severity, streak, state 0-3)
   → adaptive thresholds (per-drive bounds)
   → BiLSTM RUL (needs 30 obs)
   → RLS online adapter (personal + population correction)
   → conformal interval (quality-gated) + confidence
   → multi-horizon forecast + failure ETA
   → risk-gate check (shadow / production)
   → persist to disk (503 + rollback if write fails)
   → return full state JSON
```

---

## Phase 7 — Risk-Gated Adaptation (research novelty)
**Purpose:** decide *whether* the twin may adapt — not just *how*.

| Element | Detail |
|---|---|
| Controller | `app/ml/risk_gate.py` |
| Actions | ADAPT / WAIT / REJECT / ROLLBACK |
| (a) Degradation-aware safety | On a degrading drive an update may only make RUL **more conservative, never more optimistic** |
| (b) Anytime-valid acceptance | `E ← E·(1+λ·d/C)`; ADAPT at `E ≥ 1/α = 20`; Ville's inequality ⇒ **P(false acceptance) ≤ α = 0.05**, no multiple-testing penalty |
| (c) Drift-aware betting | Feature-only drift score (stable windows only) adjusts the bet cap without breaking the guarantee |
| Data quality | Observations with quality < 0.6 are not used as evidence |
| Evaluation | `scripts/evaluate_risk_gated_adaptation.py` → `results/risk_gated_adaptation/` |

**Evaluation protocol:** validation split (89 drives) seeds the methods; test split
(90 drives, 3 026 windows) is replayed **in failure-date order with delayed
labels** — every method predicts before the label is revealed (test-then-train).

**Results (`risk_gated_summary.csv`):**

| Scenario | Method | MAE (d) | Critical MAE (d) | Unsafe over-estimate % |
|---|---|---|---|---|
| **S0 real stream** | frozen | 10.61 | 11.97 | 33.80 |
| | always_adapt | **7.98** | **9.26** | 23.38 |
| | current_twin | 10.61 | 11.97 | 33.80 |
| | **risk_gated** | 8.45 | 9.28 | **22.24** |
| **S1 operating drift** | frozen | 12.34 | 19.11 | 82.49 |
| | **risk_gated** | 8.23 | 10.09 | 23.64 |
| **S4 biased feedback** | always_adapt | 11.13 | 15.18 | 57.88 |
| | **risk_gated** | **10.82** | **13.55** | **46.85** |
| **S5 mismatched calibration** | current_twin | 12.01 | 17.46 | 61.65 |
| | always_adapt | 8.93 | 11.39 | 30.74 |
| | **risk_gated** | 9.89 | 11.80 | 34.68 |

**Null check** (`risk_gated_null_check.csv`): 200 synthetic streams where no update
can help → the gate **never adapted in 0 % of runs** (bound ≤ 5 %), while
always-adapt adapted in 100 %.

**Ablations:** `risk_gated_no_degradation` and `risk_gated_no_drift` are in the
same CSV — turning the degradation guard off raises the unsafe rate in every
scenario (S0: 22.24 → 23.91; S4: 46.85 → 58.41).

---

## Phase 8 — Serving & Dashboard
**Purpose:** expose everything to the examiner through one UI.

| Layer | Files |
|---|---|
| API | `05_backend/app/api/v1/component2.py` (13 endpoints), schemas in `app/schemas/component2.py` |
| Service | `app/services/component2_service.py` (validation artifacts), `component2_inference_service.py` (batch CSV) |
| Frontend | `06_frontend/src/pages/DigitalTwin.jsx` (3 tabs: Validation / Simulator / Live Twin), `pages/LiveTwin.jsx` (5 feature panels + risk-gate panel), `api/component2Api.js` |

**API surface:** `/overview`, `/detections`, `/lead-time-distribution`, `/sweep`,
`/config-comparison`, `/rul-metrics`, `POST /twin`, `POST /inference/batch`,
`POST /twin/sync`, `GET /twin/{serial}/state`, `POST /twin/{serial}/learn`,
`DELETE /twin/{serial}`, `GET /twin/serials`, `GET /twin/risk-gate`,
`GET /twin/calibration`.

**Important:** the dashboard reads the *same* result CSVs the research scripts
produced — no hard-coded numbers.

---

## Phase 9 — Testing
| Test file | Covers |
|---|---|
| `tests/test_conformal.py` | Interval coverage, quality gating, confidence |
| `tests/test_forecaster.py` | Horizon bands, ETA, escalation logic |
| `tests/test_online_adapter.py` | RLS convergence, personal vs population |
| `tests/test_adaptive_thresholds.py` | Per-drive bounds, global floor |
| `tests/test_warning_decision.py` | Warning combination, no global suppression |
| `tests/test_twin_sync_service.py` | Full sync service incl. persistence/rollback |
| `tests/test_twin_online_feedback.py` | learn_outcome path + risk-gate wiring |
| `tests/test_risk_gate.py` | e-value maths, WAIT/ADAPT/REJECT/ROLLBACK |

---

## Phase 10 — Documentation
| Artifact | Path |
|---|---|
| Validation report | `docs/validation/Component_2_Adaptive_Digital_Twin_Validation_Report.docx` |
| Novelty write-up | `docs/novelty_risk_gated_adaptation.md` |
| This flow | `docs/pipeline_flow.md` |
| Component README | `02_component_2_digital_twin/README.md` |

---

## Data-lineage rules (for the examiner's "is it leakage-free?" question)
1. Scalers fit on **train only**; validation and test reuse them unchanged.
2. Conformal calibration uses **held-out residuals only** (`calibration_data.py`).
3. Splits are **drive-disjoint** — no drive appears in two splits.
4. Online-adaptation evaluation uses **delayed labels** (test-then-train).
5. Every number on the dashboard comes from a CSV written by a research script.
