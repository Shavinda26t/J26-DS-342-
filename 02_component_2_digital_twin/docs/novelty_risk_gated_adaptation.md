# Novelty: Risk-Gated, Degradation-Aware Self-Adaptive HDD Digital Twin

## 1. The problem
The twin already adapts online: an RLS corrector updates the frozen BiLSTM's RUL using realised outcomes. Most online-learning systems treat **more adaptation as always better**. For HDDs that assumption is dangerous for three reasons:

1. **The failure looks like drift.** When a drive starts failing, its SMART values change. A drift-triggered learner adapts at exactly that moment and can learn to *explain the failure away* by raising the RUL, which delays the warning.
2. **Labels are delayed.** True RUL is known only after a drive fails, so feedback arrives late, in bursts, and can be noisy or wrong.
3. **Fixed acceptance rules have no guarantee.** A rule such as "accept if 2 % better" can be fooled by a short, lucky burst of feedback. The existing population gate was fooled this way (scenario S5 below).

## 2. The contribution
A controller that decides **whether** the twin may adapt, not only **how**. After every batch of realised outcomes it takes exactly one action:

| Action | Meaning |
|---|---|
| **WAIT** | Evidence is not sufficient yet. Production is unchanged and a challenger learns in the shadow. |
| **ADAPT** | The evidence test passed (or adaptation is already approved and still safe). The update reaches production. |
| **REJECT** | The shadow challenger is provably worse. It is discarded and reset. |
| **ROLLBACK** | Live adaptation became provably harmful. Production is restored to the last known-good checkpoint. |

The decision combines three factors.

**(a) Degradation-aware safety.** Each window is labelled degrading or stable by the project's own production detector (SMART 1 / SMART 187 severity ≥ 1). For a degrading drive, an update may only make the RUL **more conservative, never more optimistic**: adaptation can never postpone a failure warning. Degrading windows are also excluded from the drift score, so the failure signal is not mistaken for drift.

**(b) Anytime-valid acceptance (testing by betting / e-values).** For each batch, the paired loss gain `d` = (error of production) − (error of challenger) is bounded by `C` = the largest gap between the two models' predictions. `C` is computed **before** the labels are seen. The evidence is

  `E ← E · (1 + λ·d/C)`

where the bet λ is the aGRAPA plug-in estimated from *previous* batches only. Under H0 ("the update is not better"), `E` is a non-negative supermartingale, so **Ville's inequality gives P(ever accepting a non-beneficial update) ≤ α = 0.05**. The bound holds even though the twin checks after every batch, with no multiple-testing inflation. ADAPT fires at `E ≥ 1/α = 20`. Mirror-image e-processes drive REJECT and ROLLBACK with the same guarantee for harm.

**(c) Drift-aware betting.** A drift score compares the operating SMART signature (SMART 1 level and volatility, stable windows only) with the drive-to-drive spread of the reference population. The score uses features only, so it may raise the bet cap without breaking the guarantee.

**Data quality.** Observations with a data-quality score below 0.6 are not used as evidence.

## 3. Experimental protocol
- The frozen seq_30 BiLSTM predicts every window of the **validation** split (89 drives, used for seeding only) and the **held-out test** split (90 drives, 3,026 windows). The splits do not overlap.
- Test drives are replayed **in failure-date order with delayed labels**. Each drive is predicted first. Its labels are revealed only after it "fails" and are then used for learning (test-then-train).
- Baselines:
  - **frozen**: the raw BiLSTM.
  - **always_adapt**: RLS correction applied after every batch.
  - **current_twin**: the existing population adapter (validated once on the seed hold-out).
- Metrics:
  - **MAE** over all windows.
  - **Critical MAE**: windows with true RUL ≤ 14 d.
  - **Unsafe over-estimate rate**: critical windows where the prediction exceeds the truth by more than 14 d, i.e. the twin says "safe" for a drive about to fail.
- S1–S5 are controlled stress tests applied on top of the real predictions. Evaluation always uses the true labels.

Script: `scripts/evaluate_risk_gated_adaptation.py`. Outputs: `results/risk_gated_adaptation/`.

## 4. Results

| Scenario | Method | MAE (d) | Critical MAE (d) | Unsafe over-estimate % |
|---|---|---|---|---|
| **S0 real stream** | frozen | 10.61 | 11.97 | 33.8 |
| | always_adapt | **7.98** | **9.26** | 23.4 |
| | current_twin | 10.61 | 11.97 | 33.8 |
| | **risk_gated** | 8.35 | 9.34 | **22.7** |
| **S1 operating drift** (+10 d from drive 30) | frozen | 12.34 | 19.11 | 82.5 |
| | always_adapt | 8.09 | 9.84 | 23.6 |
| | **risk_gated** | 8.21 | 10.09 | 23.6 |
| **S4 biased feedback** (labels 20 d late) | always_adapt | 11.13 | 15.18 | 57.9 |
| | **risk_gated** | **10.91** | **13.88** | **48.9** |
| **S5 mismatched calibration** | current_twin | 12.01 | 17.46 | **61.7** |
| | always_adapt | **8.93** | **11.39** | **30.7** |
| | **risk_gated** | 9.84 | 11.95 | 35.4 |

**Null check (validation of the guarantee):** in 200 synthetic streams where the model is already calibrated, so no update can help, the gate **never adapted (0 % of runs, bound ≤ 5 %)**. Always-adapt applied its update in 100 % of runs.

**Ablation (degradation-awareness off):** with the guard switched off, critical-window error and the unsafe rate get worse in every scenario. In S0 the unsafe rate rises from 22.7 % to 23.9 %; in S4 it rises from 48.9 % to 58.4 %.

## 5. What to claim, and what not to claim
**Claim:**
- Adaptation with a **provable bound on false acceptance**, confirmed empirically (0 % of 200 null runs).
- **The twin can never delay a failure warning by adapting.** This is a structural guarantee, and the ablation confirms its value.
- About **21 % lower MAE** than the deployed frozen model (10.61 → 8.35 d), with the **lowest unsafe over-estimate rate** of all methods on the real stream.
- It refused a harmful calibration that the existing twin accepted (S5: 61.7 % unsafe for current_twin vs 35.4 % for the gate). In S5 always-adapt still did better (30.7 %), because it re-learns from the stream straight away.

**Do not claim:**
- *Best accuracy.* Always-adapt is slightly better on raw MAE (7.98 vs 8.35 d). This is the deliberate price of waiting for evidence: the gate spent its first 8 drives in WAIT.
- *Detection of consistently biased labels.* No label-driven method can detect them (S4). The degradation guard only limits the damage.
- *Value from the drift factor.* The drift detector works (4.4 % false alarms on real data, close to the theoretical 4.6 % for a 2σ rule; 46 % detection under injected drift), but drift-aware betting made **no measurable difference** to accuracy.
- *Field-tested REJECT/ROLLBACK.* These fired in the unit tests but were never triggered on the real-data stream.

**Limitations:**
- Only one quarter of data (Q1 2023), and stress scenarios S1–S5 are synthetic.
- The guarantee assumes batches (drives) give conditionally valid evidence; windows *within* a drive are correlated, which is why a whole drive is used as one unit of evidence.
- The base BiLSTM is weak (R² ≈ 0.04), and the gate cannot fix that.
- Design choices (stable/degrading split, bet window) were refined during development on the same test stream, so these numbers are not a fully blind evaluation.

## 6. Where it lives
| Part | File |
|---|---|
| Controller | `05_backend/app/ml/risk_gate.py` |
| Live integration | `05_backend/app/services/twin_sync_service.py` (`learn_outcome`, `_build_state`) |
| API | `GET /api/v1/component2/twin/risk-gate`; `POST /twin/{serial}/learn` returns `risk_gate_decision` |
| UI | Live Twin page → "Risk-gated adaptation (novelty)" panel |
| Tests | `05_backend/tests/test_risk_gate.py`, `test_twin_online_feedback.py` |
| Evaluation | `02_component_2_digital_twin/scripts/evaluate_risk_gated_adaptation.py` |
