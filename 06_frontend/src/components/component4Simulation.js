/**
 * Component 4 experimental, deterministic WHAT-IF projection.
 * Mirrors migration progress, capacity consumption and failure-consequence
 * equations in SafeHDDEnvV2, while HOLDING externally supplied risk fixed.
 * NOT an HDD health forecast, a full Gymnasium rollout, or physical execution.
 */
const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));
const round = (value) => Math.round((value + Number.EPSILON) * 10000) / 10000;

export function simulateMitigation(form, decision) {
  if (!form || !decision) throw new Error('A validated decision is required');

  const risk = clamp(Number(form.risk_percent) / 100, 0, 1);
  const initialProgress = clamp(Number(form.migration_progress_fraction), 0, 1);
  const initialCapacity = Math.max(0, Number(form.free_capacity_gb));
  const dataSize = Math.max(0, Number(form.data_to_migrate_gb));
  const hazard = 1 - Math.pow(1 - risk, 1 / 7);
  const failureConsequence = (progress, isolated = false) =>
    isolated ? 3 : 40 * (1 - 0.9 * progress);
  const expectedLoss = (progress, isolated = false) => hazard * failureConsequence(progress, isolated);
  const originalExpectedLoss = expectedLoss(initialProgress);

  const timeline = [{
    stage: 0,
    label: 'Before',
    progress: initialProgress,
    capacity: initialCapacity,
    movedGb: 0,
    consequence: failureConsequence(initialProgress),
    expectedLoss: originalExpectedLoss,
    noOpLoss: originalExpectedLoss,
    isolated: false,
  }];

  const isEligible = decision.decision_status === 'ELIGIBLE_DRY_RUN';
  const action = decision.recommended_action;
  let reason = '';

  if (!isEligible) {
    reason = `Action not applied: ${decision.decision_status || 'not eligible'}. The safety-checked decision is ${action || 'unknown'}.`;
  } else if (action === 'MIGRATE_DATA') {
    let progress = initialProgress;
    let capacity = initialCapacity;
    let stage = 0;

    while (progress < 1 - 1e-9 && stage < 5) {
      const remainingGb = dataSize * (1 - progress);
      if (remainingGb <= 0 || capacity + 1e-8 < remainingGb * 1.1) {
        reason = 'Projection stopped because the 10% destination-capacity reserve cannot be maintained.';
        break;
      }
      const nextProgress = Math.min(1, progress + 0.25);
      const movedGb = dataSize * (nextProgress - progress);
      capacity = Math.max(0, capacity - movedGb);
      progress = nextProgress;
      stage += 1;
      timeline.push({
        stage,
        label: `Stage ${stage}`,
        progress,
        capacity,
        movedGb,
        consequence: failureConsequence(progress),
        expectedLoss: expectedLoss(progress),
        noOpLoss: originalExpectedLoss,
        isolated: false,
      });
    }
    if (!reason) {
      reason = 'Projected migration progresses in 25-percentage-point stages using the V2 simulator assumptions. HDD failure risk is held constant.';
    }
  } else if (action === 'ISOLATE_DRIVE') {
    timeline.push({
      stage: 1, label: 'Isolated', progress: initialProgress,
      capacity: initialCapacity, movedGb: 0,
      consequence: 3, expectedLoss: expectedLoss(initialProgress, true),
      noOpLoss: originalExpectedLoss, isolated: true,
    });
    reason = 'Modeled isolation changes the synthetic failure-consequence parameter to 3 units; it does not repair the HDD.';
  } else {
    reason = `No physical state transition is modeled for ${action}. The recommendation remains advisory.`;
  }

  return {
    risk, dailyHazard: hazard, action, eligible: isEligible,
    blocked: !isEligible,
    timeline: timeline.map(p => ({
      ...p,
      progressPercent: round(p.progress * 100),
      capacity: round(p.capacity),
      movedGb: round(p.movedGb),
      consequence: round(p.consequence),
      expectedLoss: round(p.expectedLoss),
      noOpLoss: round(p.noOpLoss),
    })),
    reason,
    disclaimer: 'SIMULATION ONLY — synthetic, uncalibrated assumptions. No HDD command is run. This is a what-if projection, not a measured improvement or failure forecast.',
  };
}
