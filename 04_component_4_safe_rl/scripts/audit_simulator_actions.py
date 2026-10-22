
"""
Component 4 - RL Simulator Action Effect Audit

Purpose:
Compare all seven mitigation actions from the same
initial synthetic HDD state and random seed.

No RL training or physical HDD operations.
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hdd_rl_env import SafeHDDEnv

SEED = 2026
results = []

for action_id, action in enumerate(SafeHDDEnv.ACTIONS):

    env = SafeHDDEnv(episode_days=30)

    try:
        env.reset(seed=SEED)

        # Identical synthetic starting conditions
        env.risk = 0.75
        env.rul_days = 10.0
        env.free_capacity_gb = 1000.0
        env.data_to_migrate_gb = 200.0
        env.redundant_copies = 2
        env.target_healthy = True
        env.migration_progress = 0.0
        env.io_utilization = 0.40
        env.admin_approved = True
        env.isolated = False

        _, reward, terminated, truncated, info = (
            env.step(action_id)
        )

        results.append({
            "proposed_action": action.value,
            "applied_action": info["applied_action"],
            "blocked": info["blocked_by_safety"],
            "violations": info["safety_violations"],
            "reward": round(reward, 5),
            "risk_after": round(env.risk, 6),
            "migration_progress_after": round(
                env.migration_progress, 4
            ),
            "free_capacity_after_gb": round(
                env.free_capacity_gb, 2
            ),
            "isolated_after": env.isolated,
            "failure_hazard": round(
                info["daily_failure_hazard"], 6
            ),
            "failure_occurred": info["failed"],
            "failure_consequence_if_failed": round(
                info["failure_consequence_if_failed"], 4
            )
        })

    finally:
        env.close()

# Save research evidence
OUTPUT = ROOT / "results" / "87_simulator_action_audit.json"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

report = {
    "experiment": "C4_Simulator_Action_Audit",
    "seed": SEED,
    "mode": "SYNTHETIC_DRY_RUN",
    "training_performed": False,
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "results": results
}

with open(OUTPUT, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

print("=" * 65)
print("STEP 87 - SAFE HDD RL ACTION EFFECT AUDIT")
print("=" * 65)

for item in results:
    print("\nAction:", item["proposed_action"])
    print("Applied:", item["applied_action"])
    print("Blocked:", item["blocked"])
    print("Reward:", item["reward"])
    print("Migration progress:",
          item["migration_progress_after"])
    print("Failure consequence:",
          item["failure_consequence_if_failed"])
    print("Violations:", item["violations"])

print("\nAUDIT COMPLETED")
print("Report saved:", OUTPUT)
