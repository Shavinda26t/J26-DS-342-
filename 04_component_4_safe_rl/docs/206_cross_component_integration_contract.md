
# Component 4 — Cross-Component Integration Contract

## Status
Trained DQN inference: IMPLEMENTED AND VERIFIED
Independent safety validation: IMPLEMENTED AND VERIFIED
React and FastAPI integration: IMPLEMENTED AND VERIFIED
Automatic C1/C2 prediction transfer: NOT YET CONNECTED

## Component 1 — Failure Risk Input

Component 4 requires:
- disk_id: string
- failure_probability: number between 0 and 1
- prediction_horizon_days: 7
- observed_at: timezone-aware timestamp

The current Component 1 /analyze endpoint exposes
fleet drift analysis, not a compatible seven-day
failure-probability prediction.

A drift score must not be substituted for a calibrated
failure probability.

Integration requires a confirmed C1 prediction endpoint
or another validated seven-day risk provider.

## Component 2 — Remaining Useful Life

Component 2 provides trained BiLSTM RUL inference through:

POST /api/v1/component2/inference/batch

The endpoint accepts HDD observation CSV data and
requires at least 30 observations per drive, including
the model's required feature columns.

Its response includes:
- serial_number
- status
- predicted_rul_days

For successful predictions, Component 4 can map:
predicted_rul_days -> rul_days

The serial number must match the HDD being assessed.

The held-out /twin endpoint is for drives present in
the digital twin's evaluation results and should not
be treated as a general live inference endpoint.

## Component 4 — Trained Mitigation Endpoint

POST /api/v1/component4/rl-decide

Required fields:
- disk_id
- observed_at
- failure_probability
- prediction_horizon_days = 7
- rul_days (nonnegative number)
- storage context:
  - free_capacity_gb
  - data_to_migrate_gb
  - healthy_redundant_copies
  - target_healthy
  - migration_progress_fraction
  - io_utilization
  - admin_approved

The endpoint loads:
153_dqn_v2_actionable_10000.zip

It returns:
- DQN proposed action
- safety-checked recommendation
- decision status
- independent safety assessment
- model checkpoint identifier
- explanation

## Integration Responsibilities

Component 1:
Provide an appropriately validated seven-day failure
risk for the selected HDD.

Component 2:
Provide RUL for the same HDD and observation period
using the trained model.

Team integration:
Match identifiers and timestamps, obtain current
storage context, and pass the combined inputs to C4.

Component 4:
Generate a trained DQN mitigation proposal and apply
independent safety validation.

## Limitations

- The DQN was trained in a synthetic HDD simulator.
- Automatic C1/C2 prediction transfer is pending.
- The current dashboard supports manual scenario inputs.
- Administrator approval in the dashboard is simulated.
- No physical HDD migration or isolation is performed.
