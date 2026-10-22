
# Component 4 — Safe Reinforcement Learning for HDD Mitigation

## 1. Research Objective

Component 4 investigates how reinforcement learning and
independent safety constraints can support HDD mitigation
decisions under operational limitations.

Research questions include:

1. Can RL policies learn useful HDD mitigation actions?
2. How do DQN, PPO and A2C compare with rule-based policies?
3. Can independent safety filtering prevent unsafe actions?
4. How do failure risk and intervention cost affect decisions?

All current operations are simulation or dry-run only.

No real HDD migration, isolation, throttling, rebalancing
or replacement is performed.

## 2. System Architecture

Component 4 consists of:

- HDD decision input validation
- Risk-aware rule-based decision engine
- Independent safety validator
- Gymnasium HDD mitigation simulation environments
- DQN, PPO and A2C research experiments
- Safety-filtered RL evaluation
- FastAPI backend integration
- React frontend dashboard

The component is integrated into the team repository.

Backend:
05_backend/app/api/v1/component4_decision.py

Frontend:
06_frontend/src/pages/SafeOptimization.jsx

Research source:
04_component_4_safe_rl/src/

Research scripts:
04_component_4_safe_rl/scripts/

Tests:
04_component_4_safe_rl/tests/

Experiment reports:
04_component_4_safe_rl/results/

## 3. Research Dataset

Source: Backblaze Hard Drive Stats, Q1 2026.

Selected HDD model:
ST12000NM0008

Initial extracted data:

- 1,664,558 HDD-day observations
- 18,650 unique HDDs
- 129 HDDs with recorded failure events

The source data was processed in a separate local
research workspace:

E:\HDD_Research

Raw Backblaze data and large processed datasets are
not currently included in this Git repository.

Dataset ingestion and preprocessing evidence must
be preserved separately for reproducibility.

## 4. Data Preprocessing

The supporting failure-prediction study included:

- HDD model-specific data extraction
- Missing-value and data-quality audits
- SMART attribute availability checks
- Removal of unusable observations
- Preservation of failure events
- Prevention of direct failure-day label leakage
- Seven-day future-failure target creation
- End-of-period censoring checks
- Chronological train/validation/test splitting

Selected SMART predictors:

- smart_5_raw
- smart_9_raw
- smart_187_raw
- smart_197_raw

SMART 198 was excluded from the selected feature set
because its values duplicated SMART 197 in the
available observations.

Clean pre-failure observations:
1,664,426

Seven-day label-eligible observations:
1,520,797

Positive labelled observations:
679

Important limitation:
The data is extremely imbalanced, and some failing
HDDs do not have eligible positive seven-day examples.

## 5. Chronological Dataset Splits

Training period:
2026-01-01 to 2026-02-08

Validation period:
2026-02-16 to 2026-03-01

Held-out test period:
2026-03-09 to 2026-03-24

Seven-day gaps separate the time periods.

Split sizes:

- Training: 716,096 rows
- Validation: 256,871 rows
- Test: 292,303 rows

Some HDD identities occur in multiple chronological
periods. The splits are not drive-independent.

The held-out test partition must not be treated as
evaluated until its final evaluation is completed.

## 6. Supporting Supervised Models

Three classification models were trained to study
seven-day HDD failure prediction:

### Logistic Regression

- Log-transformed SMART features
- StandardScaler trained on training data only
- Class-weighted classification

Validation PR-AUC: 0.036524
Validation ROC-AUC: 0.956506

### Random Forest

- 100 trees
- Maximum depth: 12
- Minimum leaf samples: 20

Validation PR-AUC: 0.030072
Validation ROC-AUC: 0.697521

### XGBoost

- 200 boosting trees
- Maximum depth: 4
- Imbalance weighting

Validation PR-AUC: 0.033067
Validation ROC-AUC: 0.789437

These are validation results, not final test results.

Because HDD failure is rare, classification accuracy
alone is not an appropriate primary metric.

Precision, recall, PR-AUC and operational alert
budgets are more informative.

The prediction outputs have not been calibrated
as reliable real-world failure probabilities.

These models are supporting research models.
They are not currently deployed through the
Component 4 decision endpoint.

## 7. Reinforcement Learning Models

Three RL algorithms were studied using
Stable-Baselines3 and PyTorch:

1. DQN — Deep Q-Network
2. PPO — Proximal Policy Optimization
3. A2C — Advantage Actor-Critic

Research environments:

- SafeHDDEnv: Initial synthetic environment
- SafeHDDEnvV2: Revised synthetic environment

The V2 environment includes:

- Seven-day to daily hazard conversion
- Expected failure-loss reward
- Configurable risk drift and risk noise
- Stratified initial-risk sampling
- Independent safety checks
- Simulated migration progress and costs

These environments are synthetic research simulators.
They have not been validated as real-world HDD
Digital Twins.

## 8. Mitigation Action Space

Seven mitigation proposals are represented:

- NO_OP
- RECHECK_SOON
- THROTTLE_BACKGROUND_IO
- MIGRATE_DATA
- REBALANCE_WORKLOAD
- ISOLATE_DRIVE
- RECOMMEND_REPLACEMENT

Not all actions are fully operationally supported.

For example, the safety validator blocks throttling
and rebalancing when the required workload context
cannot be verified.

All actions remain simulated or advisory.

## 9. Independent Safety Validator

Implemented safety checks include:

- Destination capacity requirements
- Healthy redundant-copy requirements
- Migration I/O utilization limit
- Healthy migration target
- Migration progress and completion
- Administrator approval requirements
- Safe fallback recommendations

The independent validator evaluates actions
separately from the decision policy.

The policy may propose an action that is blocked
by the validator.

The current safe fallback is RECHECK_SOON.

Administrator approval in the research simulator
is a synthetic input, not real access authorization.

## 10. V2 Research Results

The initial V2 experiments used:

- Stratified synthetic starting risks
- 30-day simulation episodes
- Stable-risk training configuration
- Matched evaluation seeds
- 10,000-step RL pilot budgets

Original V2 pilot results:

Policy                  Mean episode reward

DQN V2                  -14.32076
PPO V2                  -14.27776
A2C V2                  -14.27776
NO_OP                   -14.27776
SAFE_MIGRATION_FIRST    -10.04356

These results indicate that the short initial RL
pilots did not learn effective migration behaviour.

### Actionable-Start DQN

Training was repeated using initially safe and
simulated-approved migration scenarios.

Evaluation results showed:

- 1,620 migration proposals
- 1,539 blocked migration proposals
- 81 applied migration actions
- 8 completed migrations

This revealed poor generalization from actionable
training scenarios to unrestricted evaluation states.

### Safety-Filtered DQN

An external safety filter was evaluated with the
trained actionable-start DQN checkpoint.

Held-out evaluation:
200 simulation episodes.

Mean episode rewards:

RAW_DQN                 -42.68885
FILTERED_DQN            -13.59935
NO_OP                   -16.84177
SAFE_MIGRATION_FIRST    -12.87650

The filtered model had zero observed safety blocks
in this evaluation, but the filter was responsible
for enforcing the migration constraints.

The underlying DQN had not learned reliable safety.

### Paired Bootstrap Analysis

Filtered DQN minus NO_OP:

Mean improvement: +3.24242
95% bootstrap interval: [1.74906, 5.02087]

Filtered DQN minus safe migration-first:

Mean difference: -0.72285
95% bootstrap interval: [-1.05219, -0.44185]

These results apply only to the synthetic evaluation
environment and the tested model checkpoint.

No claim of real-world HDD effectiveness is made.

## 11. Main Research Contributions

Component 4 contributes:

1. Independent safety validation of mitigation actions.
2. Risk-aware HDD mitigation decision architecture.
3. Reproducible DQN, PPO and A2C experiments.
4. Risk-sensitive migration versus NO_OP analysis.
5. Diagnosis of unsafe RL action proposals.
6. Hybrid safety-filtered DQN evaluation.
7. Matched-seed evaluation and statistical analysis.
8. Explainable FastAPI and React dry-run integration.

## 12. Backend Integration

Endpoint:

POST /api/v1/component4/decide


### Trained DQN Inference Endpoint

POST /api/v1/component4/rl-decide

This endpoint loads the actual trained DQN checkpoint:

models/153_dqn_v2_actionable_10000.zip

The endpoint:

1. Accepts validated HDD risk, RUL and storage inputs.
2. Converts the inputs into the ten-feature RL observation.
3. Generates an action using the trained DQN model.
4. Evaluates the proposed action using the independent safety validator.
5. Applies the experimental safety filter when migration is blocked.
6. Returns the model name, proposed action, final recommendation,
   safety assessment and explanation.

The endpoint has been manually verified through FastAPI Swagger.

The React Safe Optimization page supports two decision modes:

- Trained DQN + Independent Safety Validator
- Rule-Based Safety Baseline

Both modes use the same manually entered HDD conditions.

The trained DQN was developed using synthetic simulation data.
Risk and RUL are supplied as inputs and are not currently
predicted automatically by this endpoint.

All recommendations remain advisory. No physical HDD operations occur.


The endpoint accepts:

- HDD identifier
- Seven-day failure-risk estimate
- Remaining useful life
- Free destination capacity
- Data migration size
- Healthy redundant-copy count
- Destination health
- Migration progress
- I/O utilization
- Administrator approval flag

The endpoint returns:

- Risk classification
- Proposed mitigation action
- Safety-checked recommendation
- Decision status
- Safety violations
- Explanation
- Execution mode: DRY_RUN_ONLY

The current endpoint uses the deterministic
risk-aware baseline policy.

The rule-based /decide endpoint does not load an RL policy. The separate /rl-decide endpoint loads the trained DQN checkpoint.

## 13. Manual Input Demonstration

The React dashboard is available at:

/safe-rl

The browser address uses the frontend development
server's displayed port.

Example input:

- Seven-day failure risk: 75%
- RUL: 10 days
- Available capacity: 1200 GB
- Data to migrate: 500 GB
- Healthy redundant copies: 2
- Destination healthy: true
- I/O utilization: 0.35
- Administrator approval: false

Expected result:

- Proposed action: MIGRATE_DATA
- Recommended action: RECHECK_SOON
- Status: PENDING_ADMIN_APPROVAL

Change available capacity to 100 GB.

Expected result:

- Status: BLOCKED_BY_SAFETY
- Violation: Insufficient destination capacity
- Fallback: RECHECK_SOON

The manual inputs exercise the decision system.
They do not measure the accuracy of an HDD
failure-prediction model.

## 14. Local Setup

Recommended tested environment:

- Windows
- Python 3.11
- Anaconda environment: hdd-pp1
- Node.js and npm

The Python environment used for research includes:

- NumPy
- Gymnasium 1.1.1
- Stable-Baselines3 2.7.0
- PyTorch 2.7.1+cpu

The FastAPI backend dependencies are listed in:

05_backend/requirements.txt

### Start Backend

From repository root:

cd 05_backend

python -m uvicorn app.main:app --reload --port 8000

API documentation:

http://127.0.0.1:8000/docs

### Start Frontend

Open a separate terminal at repository root:

cd 06_frontend

npm install

npm run dev

Open the displayed local URL and navigate to:

/safe-rl

### Run Component 4 Tests

From repository root:

python -m unittest discover -s 04_component_4_safe_rl/tests -p "test_*.py" -v

Latest verified result before this README update:

74 tests passed.

### Build Frontend

From repository root:

cd 06_frontend

npm run build

## 15. Dataset and Checkpoint Availability

The complete Backblaze dataset, external
preprocessing notebooks and trained model
checkpoints are not all included in Git.

The supporting preprocessing and supervised
modelling work was performed under:

E:\HDD_Research

Experiment scripts and JSON summaries are committed
under the Component 4 directory.

Trained RL checkpoint ZIPs are stored locally in
04_component_4_safe_rl/models/.

Any deployment of an RL model requires explicit
checkpoint delivery, version verification and
additional safety testing.

Do not assume research evidence files represent
a deployed, end-to-end prediction pipeline.

## 16. Integration Boundaries

Component 4 expects a seven-day risk estimate.

The team's other components may provide:

- HDD health or failure predictions
- Digital Twin degradation information
- Remaining useful life estimates
- Causal explanations

These outputs must be connected through agreed,
validated API contracts.

Prediction horizons, probability calibration,
units and missing-value handling must be
consistent before full integration.

## 17. Current Limitations

- Synthetic simulator dynamics
- Provisional reward and intervention-cost assumptions
- No real HDD operational execution
- No real administrator authorization integration
- No calibrated failure-risk inference in the UI
- Trained DQN inference requires the local model checkpoint and compatible Python dependencies; PPO and A2C are not currently served.
- No proven superiority to the safety-aware
  rule-based migration baseline
- Limited RL training seeds and sensitivity studies
- Final held-out supervised test evaluation incomplete

This is a research prototype intended for
safe experimentation and further validation.

## 18. PP1 Research Status

Implemented and tested:

- Decision input contracts
- Independent safety validation
- Risk-aware decision baseline
- Synthetic HDD mitigation simulators
- DQN, PPO and A2C pilots
- Safety-filtered DQN analysis
- FastAPI and React integration
- Automated regression tests
- Simulation performance comparisons

Remaining work is documented as research
limitations and future integration tasks.

The completed work must not be represented
as validated autonomous HDD self-healing.
