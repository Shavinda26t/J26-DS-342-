
"""
Step 151 - V2 Migration Benefit by Risk Band

Compare:
1. NO_OP
2. SAFE_MIGRATION_FIRST

Use matched synthetic initial states and random seeds.

This is a diagnostic study of our simulator,
not a calibrated real-world HDD evaluation.
"""

import json
from collections import defaultdict
from statistics import mean, stdev

from v2_training_config import (
    ROOT,
    ENVIRONMENT_CONFIG,
    make_v2_env,
)

from contracts import MitigationAction


EPISODES = 600
SEED_START = 50000


def classify_risk(risk):

    if risk < 0.012:
        return "VERY_LOW"

    if risk < 0.15:
        return "MODERATE"

    return "HIGH"


def run_episode(seed, policy):

    env = make_v2_env()

    try:
        observation, _ = env.reset(seed=seed)

        initial_risk = float(env.risk)

        initial_assessment = env.validator.assess(
            env._get_drive_state(),
            MitigationAction.MIGRATE_DATA,
        )

        initially_actionable = bool(
            initial_assessment.technically_eligible
            and not initial_assessment.approval_pending
        )

        total_reward = 0.0
        blocks = 0
        migrations = 0
        failed = False

        done = False

        while not done:

            if policy == "SAFE_MIGRATION_FIRST":

                assessment = env.validator.assess(
                    env._get_drive_state(),
                    MitigationAction.MIGRATE_DATA,
                )

                if (
                    env.migration_progress < 1.0
                    and assessment.technically_eligible
                    and not assessment.approval_pending
                ):
                    action = MitigationAction.MIGRATE_DATA

                else:
                    action = MitigationAction.NO_OP

            elif policy == "NO_OP":

                action = MitigationAction.NO_OP

            else:
                raise ValueError("Unknown policy")

            action_id = env.ACTIONS.index(action)

            observation, reward, terminated, truncated, info = (
                env.step(action_id)
            )

            total_reward += float(reward)

            blocks += int(info["blocked_by_safety"])

            migrations += int(
                info["applied_action"] == "MIGRATE_DATA"
            )

            failed = failed or bool(info["failed"])

            done = terminated or truncated

        return {
            "initial_risk": initial_risk,
            "risk_band": classify_risk(initial_risk),
            "initially_actionable": initially_actionable,
            "episode_reward": total_reward,
            "failure": failed,
            "safety_blocks": blocks,
            "migration_actions": migrations,
            "migration_complete": (
                env.migration_progress >= 1.0
            ),
        }

    finally:
        env.close()


groups = defaultdict(list)

for offset in range(EPISODES):

    seed = SEED_START + offset

    no_op = run_episode(seed, "NO_OP")

    migration = run_episode(
        seed,
        "SAFE_MIGRATION_FIRST",
    )

    # Both policies must start in the same scenario.
    assert (
        no_op["initial_risk"]
        == migration["initial_risk"]
    )

    difference = (
        migration["episode_reward"]
        - no_op["episode_reward"]
    )

    row = {
        "seed": seed,
        "difference": difference,
        "no_op_reward": no_op["episode_reward"],
        "migration_reward": migration["episode_reward"],
        "initially_actionable": (
            migration["initially_actionable"]
        ),
        "migration_complete": (
            migration["migration_complete"]
        ),
        "migration_actions": (
            migration["migration_actions"]
        ),
        "migration_blocks": (
            migration["safety_blocks"]
        ),
    }

    groups["ALL"].append(row)

    groups[migration["risk_band"]].append(row)

    if migration["initially_actionable"]:
        groups["ACTIONABLE_AT_RESET"].append(row)

    else:
        groups["NOT_ACTIONABLE_AT_RESET"].append(row)


def summarize(rows):

    if not rows:
        return {"episodes": 0}

    differences = [
        row["difference"]
        for row in rows
    ]

    n = len(differences)

    # Exploratory normal-approximation uncertainty
    # interval over paired simulation returns.
    standard_error = (
        stdev(differences) / (n ** 0.5)
        if n > 1 else 0.0
    )

    avg = mean(differences)

    return {
        "episodes": n,
        "mean_no_op_reward": round(
            mean(row["no_op_reward"] for row in rows), 5
        ),
        "mean_migration_reward": round(
            mean(row["migration_reward"] for row in rows), 5
        ),
        "mean_reward_advantage": round(avg, 5),
        "approximate_95pct_interval": [
            round(avg - 1.96 * standard_error, 5),
            round(avg + 1.96 * standard_error, 5),
        ],
        "positive_advantage_episodes": sum(
            row["difference"] > 0 for row in rows
        ),
        "initially_actionable": sum(
            row["initially_actionable"] for row in rows
        ),
        "completed_migrations": sum(
            row["migration_complete"] for row in rows
        ),
        "safety_blocks": sum(
            row["migration_blocks"] for row in rows
        ),
    }


summary = {
    group: summarize(rows)
    for group, rows in groups.items()
}


print("=" * 62)
print("STEP 151 - MIGRATION BENEFITS BY INITIAL RISK BAND")
print("=" * 62)

for band in [
    "VERY_LOW",
    "MODERATE",
    "HIGH",
    "ACTIONABLE_AT_RESET",
    "NOT_ACTIONABLE_AT_RESET",
    "ALL",
]:

    result = summary.get(band, {"episodes": 0})

    print("\nGroup:", band)
    print("Episodes:", result["episodes"])

    if result["episodes"] == 0:
        continue

    print(
        "NO_OP mean reward:",
        result["mean_no_op_reward"]
    )

    print(
        "Migration mean reward:",
        result["mean_migration_reward"]
    )

    print(
        "Migration advantage:",
        result["mean_reward_advantage"]
    )

    print(
        "Approx. 95% interval:",
        result["approximate_95pct_interval"]
    )

    print(
        "Completed migrations:",
        result["completed_migrations"]
    )


report = {
    "experiment": "C4_V2_MIGRATION_BENEFIT_BY_RISK_BAND",
    "simulation_only": True,
    "seed_start": SEED_START,
    "episodes_per_policy": EPISODES,
    "matched_seeds": True,
    "environment_configuration": ENVIRONMENT_CONFIG,
    "summary": summary,
    "limitations": [
        "Synthetic and uncalibrated environment",
        "Equal risk-band sampling is not fleet prevalence",
        "Paired interval is exploratory",
        "Approved states use simulated authorization",
        "Migration changes failure consequences only",
        "No real HDD operations",
    ],
}


output = (
    ROOT / "results"
    / "151_v2_migration_by_risk_band.json"
)

output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\nReport saved:", output)
print("STEP 151 COMPLETE")
