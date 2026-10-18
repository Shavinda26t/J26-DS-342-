
"""
Step 131 - Controlled V2 migration experiment.

Compare NO_OP with MIGRATE_THEN_NO_OP using matched
random seeds and technically safe synthetic conditions.

This is a diagnostic experiment, not a fleet estimate.
No physical HDD operations occur.
"""

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contracts import MitigationAction
from hdd_rl_env_v2 import SafeHDDEnvV2



EPISODES = 500
SEED_START = 20000

SCENARIOS = {
    "RISK_1_0_PERCENT": 0.010,
    "RISK_1_2_PERCENT": 0.012,
    "RISK_1_4_PERCENT": 0.014,
    "RISK_1_6_PERCENT": 0.016,
}


def run_episode(seed, initial_risk, policy):
    env = SafeHDDEnvV2(
        episode_days=30,
        risk_drift_per_day=0.0,
        risk_noise_std=0.0,
    )

    try:
        env.reset(seed=seed)

        # Controlled synthetic starting conditions.
        # These intentionally isolate the reward trade-off.
        env.risk = initial_risk
        env.rul_days = 45.0
        env.free_capacity_gb = 1000.0
        env.data_to_migrate_gb = 200.0
        env.redundant_copies = 2
        env.target_healthy = True
        env.migration_progress = 0.0
        env.io_utilization = 0.40
        env.admin_approved = True
        env.isolated = False

        total_reward = 0.0
        failures = 0
        blocks = 0
        migrations = 0

        done = False

        while not done:

            if (
                policy == "MIGRATE_THEN_NO_OP"
                and env.migration_progress < 1.0
            ):
                action = MitigationAction.MIGRATE_DATA
            else:
                action = MitigationAction.NO_OP

            action_id = env.ACTIONS.index(action)

            _, reward, terminated, truncated, info = (
                env.step(action_id)
            )

            total_reward += reward
            failures += int(info["failed"])
            blocks += int(info["blocked_by_safety"])

            if info["applied_action"] == "MIGRATE_DATA":
                migrations += 1

            done = terminated or truncated

        return {
            "reward": total_reward,
            "failures": failures,
            "blocks": blocks,
            "migrations": migrations,
            "final_migration_progress": env.migration_progress,
        }

    finally:
        env.close()


results = []

print("=" * 65)
print("STEP 139 - HELD-OUT BREAK-EVEN VALIDATION")
print("=" * 65)

for scenario, initial_risk in SCENARIOS.items():

    policy_results = {}

    for policy in ["NO_OP", "MIGRATE_THEN_NO_OP"]:

        episodes = [
            run_episode(
                seed=SEED_START + i,
                initial_risk=initial_risk,
                policy=policy,
            )
            for i in range(EPISODES)
        ]

        policy_results[policy] = {
            "mean_episode_reward": round(
                mean(x["reward"] for x in episodes), 4
            ),
            "failure_episodes": sum(
                x["failures"] for x in episodes
            ),
            "safety_blocks": sum(
                x["blocks"] for x in episodes
            ),
            "applied_migrations": sum(
                x["migrations"] for x in episodes
            ),
            "episodes_with_complete_migration": sum(
                x["final_migration_progress"] >= 1.0
                for x in episodes
            ),
        }

    no_op = policy_results["NO_OP"]
    migration = policy_results["MIGRATE_THEN_NO_OP"]

    difference = round(
        migration["mean_episode_reward"]
        - no_op["mean_episode_reward"],
        4,
    )

    print(f"\nScenario: {scenario}")
    print("Initial 7-day risk:", initial_risk)
    print("NO_OP mean reward:", no_op["mean_episode_reward"])
    print(
        "MIGRATE_THEN_NO_OP mean reward:",
        migration["mean_episode_reward"],
    )
    print("Migration minus NO_OP:", difference)
    print("NO_OP failure episodes:", no_op["failure_episodes"])
    print(
        "Migration failure episodes:",
        migration["failure_episodes"],
    )
    print("Migration safety blocks:", migration["safety_blocks"])
    print(
        "Episodes completing migration:",
        migration["episodes_with_complete_migration"],
    )

    results.append({
        "scenario": scenario,
        "initial_seven_day_risk": initial_risk,
        "policies": policy_results,
        "reward_difference_migration_minus_no_op": difference,
    })


report = {
    "experiment": "C4_V2_HELDOUT_BREAK_EVEN_VALIDATION",
    "simulation_only": True,
    "episodes_per_policy_per_scenario": EPISODES,
    "seed_start": SEED_START,
    "episode_days": 30,
    "matched_seeds": True,
    "controlled_storage_conditions": True,
    "results": results,
    "limitations": [
        "Synthetic risks and rewards are not calibrated.",
        "Migration eligibility is intentionally controlled.",
        "Simulated administrator approval is not real authorization.",
        "Migration changes consequences, not physical failure hazard.",
        "This experiment is not representative of fleet prevalence.",
    ],
}

output = (
    ROOT / "results"
    / "139_v2_breakeven_validation.json"
)

output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("\nReport:", output)
print("STEP 139 COMPLETE")
