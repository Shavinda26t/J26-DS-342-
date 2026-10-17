
"""
Step 104: Audit SafeHDDEnv V2 initial conditions.

Checks:
1. Synthetic seven-day failure-risk distribution.
2. First simulated day's failure hazard.
3. Migration technical eligibility and approval.
4. Most common migration safety blockers.

No RL training and no real HDD operations.
"""

from pathlib import Path
from collections import Counter
from statistics import mean
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contracts import MitigationAction
from hdd_rl_env_v2 import SafeHDDEnvV2


SEED_START = 1000
EPISODES = 200

initial_risks = []
first_day_hazards = []
first_day_failures = 0

technically_eligible = 0
approved_and_eligible = 0
approval_pending = 0

violations = Counter()

for seed in range(SEED_START, SEED_START + EPISODES):

    env = SafeHDDEnvV2(episode_days=30)

    try:
        env.reset(seed=seed)

        initial_risks.append(float(env.risk))

        assessment = env.validator.assess(
            env._get_drive_state(),
            MitigationAction.MIGRATE_DATA
        )

        if assessment.technically_eligible:
            technically_eligible += 1

            if assessment.approval_pending:
                approval_pending += 1
            else:
                approved_and_eligible += 1

        for violation in assessment.violations:
            violations[violation] += 1

        # Take NO_OP for one day to observe the actual
        # first-step hazard after synthetic degradation.
        action_id = env.ACTIONS.index(
            MitigationAction.NO_OP
        )

        _, _, _, _, info = env.step(action_id)

        first_day_hazards.append(
            float(info["daily_failure_hazard"])
        )

        if info["failed"]:
            first_day_failures += 1

    finally:
        env.close()


report = {
    "experiment": "C4_V2_INITIAL_CONDITIONS_AUDIT",
    "simulation_only": True,
    "episodes": EPISODES,
    "seed_start": SEED_START,
    "initial_seven_day_risk": {
        "minimum": round(min(initial_risks), 6),
        "mean": round(mean(initial_risks), 6),
        "maximum": round(max(initial_risks), 6)
    },
    "first_day_hazard": {
        "minimum": round(min(first_day_hazards), 6),
        "mean": round(mean(first_day_hazards), 6),
        "maximum": round(max(first_day_hazards), 6)
    },
    "first_day_failures": first_day_failures,
    "migration_technically_eligible": technically_eligible,
    "migration_approved_and_eligible": approved_and_eligible,
    "migration_approval_pending": approval_pending,
    "migration_violation_counts": dict(violations),
    "limitations": [
        "All risks and failure dynamics are synthetic.",
        "Approval flags are simulated, not real authorization.",
        "Eligibility is checked only at episode reset.",
        "This audit does not measure migration effectiveness."
    ]
}

output = (
    ROOT / "results"
    / "104_v2_initial_conditions_audit.json"
)

output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)


print("=" * 60)
print("STEP 104 - V2 INITIAL CONDITIONS AUDIT")
print("=" * 60)

print("Episodes:", EPISODES)

print(
    "Initial seven-day risk (mean):",
    report["initial_seven_day_risk"]["mean"]
)

print(
    "First-day failure hazard (mean):",
    report["first_day_hazard"]["mean"]
)

print(
    "First-day failures:",
    first_day_failures
)

print(
    "Migration technically eligible:",
    technically_eligible
)

print(
    "Migration approved and eligible:",
    approved_and_eligible
)

print(
    "Migration approval pending:",
    approval_pending
)

print("\nMigration safety violations:")

for reason, count in violations.most_common():
    print(f"  {reason}: {count}")

print("\nAudit saved:", output)
print("STEP 104 COMPLETE")
