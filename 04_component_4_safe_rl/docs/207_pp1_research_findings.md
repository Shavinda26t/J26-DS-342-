
# Component 4 — PP1 Research Findings

## Research Title

Safe Reinforcement Learning for HDD Failure Mitigation

## 1. Research Objective

Develop a risk-aware HDD mitigation decision system using
reinforcement learning and independent safety constraints.

The system recommends mitigation actions based on HDD risk,
remaining useful life (RUL), and storage conditions.

## 2. Research Methodology

The work includes:

1. Supporting HDD SMART preprocessing and seven-day
   failure-label research using Backblaze data.
2. Supervised prediction experiments using:
   - Logistic Regression
   - Random Forest
   - XGBoost
3. Development of synthetic HDD mitigation environments.
4. Training and comparison of:
   - Deep Q-Network (DQN)
   - Proximal Policy Optimization (PPO)
   - Advantage Actor-Critic (A2C)
5. Evaluation against NO_OP and safety-aware
   migration-first baseline policies.
6. Independent mitigation safety validation.
7. Safety filtering and held-out evaluation.
8. Integration of a saved DQN checkpoint with
   FastAPI and React.

The supervised experiments support risk-prediction
research. They do not automatically provide risk
probabilities to the deployed Component 4 API.

## 3. Mitigation System

The system considers seven actions:

- NO_OP
- RECHECK_SOON
- THROTTLE_BACKGROUND_IO
- MIGRATE_DATA
- REBALANCE_WORKLOAD
- ISOLATE_DRIVE
- RECOMMEND_REPLACEMENT

Actions are subject to independent validation.

The validator checks relevant restrictions, including
administrator approval, redundancy, destination capacity,
destination health and I/O constraints.

A recommendation is advisory and does not execute a
physical storage operation.

## 4. DQN, PPO and A2C Pilot Comparison

All three V2 pilots were evaluated on the same
100 simulated evaluation seeds.

| Policy | Mean Episode Reward |
|---|---:|
| DQN V2 pilot | -14.32076 |
| PPO V2 pilot | -14.27776 |
| A2C V2 pilot | -14.27776 |
| NO_OP | -14.27776 |
| SAFE_MIGRATION_FIRST | -10.04356 |

Higher episode reward is better.

At this pilot stage, DQN, PPO and A2C did not
outperform the migration-first baseline.

PPO and A2C selected NO_OP throughout their
reported evaluations.

These results motivated additional DQN experiments
using actionable initial conditions and safety filtering.

## 5. Safety-Filtered DQN Held-Out Evaluation

The held-out evaluation used 200 matched simulated
episodes, starting at evaluation seed 70000.

| Policy | Mean Episode Reward |
|---|---:|
| RAW_DQN | -42.68885 |
| FILTERED_DQN | -13.59935 |
| NO_OP | -16.84177 |
| SAFE_MIGRATION_FIRST | -12.87650 |

Safety-filtered DQN improved simulated mean reward
over NO_OP by 3.24242.

It did not outperform SAFE_MIGRATION_FIRST.

In this evaluation:

- Raw DQN incurred 2838 safety blocks.
- The filtered policy intercepted 2838 migration proposals.
- Filtered DQN incurred zero simulator safety blocks.
- Filtered DQN completed 9 migrations.
- All compared policies recorded 88 failure episodes.

The filter rejects unsafe migration proposals before
they reach the simulated environment.

Zero safety blocks for the filtered policy does not
mean the neural network stopped proposing unsafe actions.

Mitigation changed simulated failure consequences,
not the underlying physical failure probability.

## 6. Paired Bootstrap Analysis

Using 200 paired evaluation episodes and
5000 bootstrap repetitions:

| Comparison | Mean Difference | Bootstrap 95% Interval |
|---|---:|---|
| FILTERED_DQN - NO_OP | +3.24242 | [1.74906, 5.02087] |
| FILTERED_DQN - SAFE_MIGRATION_FIRST | -0.72285 | [-1.05219, -0.44185] |
| FILTERED_DQN - RAW_DQN | +29.08950 | [25.17400, 32.89251] |

These intervals describe simulated evaluation variation
for one trained DQN checkpoint. They do not represent
uncertainty across multiple independent training runs
or real-world HDD effectiveness.

## 7. Risk-Sensitive Mitigation Finding

An additional controlled evaluation compared
migration-first against NO_OP over 600
matched simulated episodes.

| Initial Condition | Migration Advantage |
|---|---:|
| Very low risk | -0.24332 |
| Moderate risk | +2.27219 |
| High risk | +9.63039 |
| Actionable at reset | +13.12373 |
| All scenarios | +3.84963 |

This supports the simulator-specific finding that
migration benefits depend on initial risk and
technical eligibility.

The synthetic sampling distribution is not the
actual HDD fleet risk distribution.

## 8. Application Implementation

The deployed experimental inference checkpoint is:

153_dqn_v2_actionable_10000.zip

The trained DQN is served through:

POST /api/v1/component4/rl-decide

The React Safe Optimization page provides:

- Trained DQN inference
- Rule-based baseline comparison
- Editable HDD and storage inputs
- Independent safety assessment
- Administrator approval checks
- Blocked-action explanations
- Model checkpoint identification
- Advisory mitigation recommendations

The system runs inside the shared authenticated
research dashboard.

## 9. Verification

- 84 Component 4 automated tests passed.
- 14 backend automated tests passed.
- Shared frontend production build passed.
- Real DQN checkpoint loading was verified.
- FastAPI trained-model inference was verified.
- React trained-model inference was verified.
- Safety blocking and approval scenarios were tested.

Total automated tests verified: 98.

The application uses demonstration authentication
and must not be represented as production-secure.

## 10. Research Contributions

1. Risk-aware, multi-action HDD mitigation environment.
2. Independent safety validator for RL decisions.
3. Comparative DQN, PPO and A2C pilot experiments.
4. Risk-stratified evaluation of migration benefits.
5. Safety-filtered DQN evaluation on held-out seeds.
6. Paired bootstrap confidence interval analysis.
7. Reproducible trained DQN inference integration.
8. Explainable safety decisions in the team dashboard.

## 11. Limitations

- The RL environment uses synthetic, uncalibrated
  HDD failure dynamics.
- Only one DQN training seed was used for the
  reported held-out comparison.
- PPO and A2C did not demonstrate improvement
  over the simple baselines.
- The trained DQN did not outperform the
  safety-aware migration-first baseline.
- Seven-day failure risk and RUL are currently
  supplied to Component 4 as inputs.
- Automatic prediction transfer from C1 and C2
  has not yet been completed.
- The dashboard uses simulated administrator approval.
- There is no real HDD action execution.
- Results do not establish production reliability
  or real-world mitigation effectiveness.

## 12. Evidence Files

Primary experimental evidence is in:

- results/146_dqn_v2_pilot_results.json
- results/148_ppo_v2_pilot_results.json
- results/149_a2c_v2_pilot_results.json
- results/150_v2_training_opportunities.json
- results/151_v2_migration_by_risk_band.json
- results/153_dqn_v2_actionable_results.json
- results/154_dqn_v2_safety_block_audit.json
- results/155_dqn_v2_safety_filter_results.json
- results/156_dqn_v2_safety_filter_holdout.json
- results/157_v2_holdout_confidence_intervals.json

The cross-component handover agreement is:

docs/206_cross_component_integration_contract.md

## 13. PP1 Conclusion

Component 4 demonstrates a functioning research
prototype for safe RL-based HDD mitigation.

The trained DQN produces actual decisions, while an
independent safety layer prevents ineligible migration
recommendations from being passed through unchanged.

Safety filtering substantially improved simulated
DQN outcomes over the unfiltered policy and NO_OP.

However, a safety-aware rule-based migration baseline
remained stronger than the trained DQN in the
reported held-out evaluation.

The next research phase should investigate
improved RL policies, calibrated input risk,
realistic failure dynamics, and end-to-end
component integration.
