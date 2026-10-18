
"""
Step 150 - Simulator V2 Training Opportunity Audit

Measure how frequently migration is permitted
across the three synthetic initial-risk bands.

No RL training.
No physical HDD operations.
"""

import json
from collections import Counter
from datetime import datetime, timezone

from v2_training_config import (
    ROOT,
    ENVIRONMENT_CONFIG,
    make_v2_env,
)

from contracts import MitigationAction


TOTAL_SCENARIOS = 1000
SEED_START = 40000


def risk_band(risk):

    if risk < 0.012:
        return "VERY_LOW"

    if risk < 0.15:
        return "MODERATE"

    return "HIGH"


band_results = {
    band: {
        "scenarios": 0,
        "technically_eligible": 0,
        "approved_and_eligible": 0,
        "approval_pending": 0,
        "technical_blockers": Counter(),
    }
    for band in ["VERY_LOW", "MODERATE", "HIGH"]
}


env = make_v2_env()

try:

    for seed in range(
        SEED_START,
        SEED_START + TOTAL_SCENARIOS
    ):

        env.reset(seed=seed)

        band = risk_band(env.risk)

        stats = band_results[band]

        stats["scenarios"] += 1

        assessment = env.validator.assess(
            env._get_drive_state(),
            MitigationAction.MIGRATE_DATA,
        )

        if assessment.technically_eligible:

            stats["technically_eligible"] += 1

            if assessment.approval_pending:
                stats["approval_pending"] += 1

            else:
                stats["approved_and_eligible"] += 1

        for reason in assessment.violations:
            stats["technical_blockers"][reason] += 1

finally:
    env.close()


print("=" * 62)
print("STEP 150 - V2 TRAINING OPPORTUNITY AUDIT")
print("=" * 62)

print("Scenarios:", TOTAL_SCENARIOS)

for band, stats in band_results.items():

    total = stats["scenarios"]
    eligible = stats["technically_eligible"]
    approved = stats["approved_and_eligible"]
    pending = stats["approval_pending"]

    print("\nRisk band:", band)
    print("Scenarios:", total)

    print(
        "Technically eligible:",
        eligible
    )

    print(
        "Approved and eligible:",
        approved
    )

    print(
        "Approved opportunity rate:",
        round(100 * approved / total, 2)
        if total else 0,
        "%"
    )

    print("Approval pending:", pending)

    print("Technical blockers:")

    for reason, count in (
        stats["technical_blockers"].most_common()
    ):
        print(" ", reason, ":", count)


report = {
    "experiment": "C4_V2_TRAINING_OPPORTUNITY_AUDIT",
    "simulation_only": True,
    "scenarios": TOTAL_SCENARIOS,
    "seed_start": SEED_START,
    "environment_configuration": ENVIRONMENT_CONFIG,
    "risk_bands": {
        band: {
            **{
                key: value
                for key, value in stats.items()
                if key != "technical_blockers"
            },
            "technical_blockers": dict(
                stats["technical_blockers"]
            )
        }
        for band, stats in band_results.items()
    },
    "limitations": [
        "Initial states only; later steps are not audited",
        "Synthetic approval is not real authorization",
        "Risk bands are not real fleet prevalence",
        "Eligibility does not guarantee successful migration",
    ],
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
}

output = (
    ROOT / "results"
    / "150_v2_training_opportunities.json"
)

output.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    output,
    "w",
    encoding="utf-8"
) as file:
    json.dump(report, file, indent=4)

print("\nReport saved:", output)
print("STEP 150 COMPLETE")
