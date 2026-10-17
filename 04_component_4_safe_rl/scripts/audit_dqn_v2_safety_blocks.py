
"""
Step 154 - Actionable-Start DQN Safety Audit

Evaluate the saved Step 153 DQN model.

Identify which proposed actions are blocked
and why the independent safety validator blocks them.

Read-only model evaluation.
No retraining or physical HDD operations.
"""

import json
from collections import Counter
from pathlib import Path

from stable_baselines3 import DQN

# Import configuration before contracts.
from v2_training_config import (
    ROOT,
    EVALUATION_SEED_START,
    EVALUATION_EPISODES,
    make_v2_env,
)


MODEL_FILE = (
    ROOT / "models"
    / "153_dqn_v2_actionable_10000.zip"
)

OUTPUT_FILE = (
    ROOT / "results"
    / "154_dqn_v2_safety_block_audit.json"
)

if not MODEL_FILE.exists():
    raise FileNotFoundError(MODEL_FILE)

if OUTPUT_FILE.exists():
    raise FileExistsError(
        "Step 154 report already exists. "
        "Preserve previous research evidence."
    )

model = DQN.load(
    str(MODEL_FILE),
    device="cpu",
)

proposed_counts = Counter()
blocked_by_action = Counter()
technical_reasons = Counter()
blocked_by_risk_band = Counter()

total_blocks = 0
approval_blocks = 0
applied_migrations = 0
completed_migrations = 0

examples = []


def classify_risk(risk):
    if risk < 0.012:
        return "VERY_LOW"
    if risk < 0.15:
        return "MODERATE"
    return "HIGH"


for offset in range(EVALUATION_EPISODES):

    seed = EVALUATION_SEED_START + offset
    env = make_v2_env()

    try:
        observation, _ = env.reset(seed=seed)

        band = classify_risk(env.risk)
        done = False

        while not done:

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            observation, reward, terminated, truncated, info = (
                env.step(int(action))
            )

            proposed = info["proposed_action"]
            proposed_counts[proposed] += 1

            if info["applied_action"] == "MIGRATE_DATA":
                applied_migrations += 1

            if info["blocked_by_safety"]:

                total_blocks += 1
                blocked_by_action[proposed] += 1
                blocked_by_risk_band[band] += 1

                if info["approval_pending"]:
                    approval_blocks += 1

                for reason in info["safety_violations"]:
                    technical_reasons[str(reason)] += 1

                if len(examples) < 5:
                    examples.append({
                        "seed": seed,
                        "day": info["day"],
                        "risk_band": band,
                        "proposed_action": proposed,
                        "applied_action": info["applied_action"],
                        "approval_pending": info["approval_pending"],
                        "violations": info["safety_violations"],
                    })

            done = terminated or truncated

        if env.migration_progress >= 1.0:
            completed_migrations += 1

    finally:
        env.close()


report = {
    "experiment": "C4_DQN_V2_SAFETY_BLOCK_AUDIT_154",
    "simulation_only": True,
    "model": MODEL_FILE.name,
    "evaluation_seed_start": EVALUATION_SEED_START,
    "evaluation_episodes": EVALUATION_EPISODES,
    "total_proposals": sum(proposed_counts.values()),
    "proposed_actions": dict(proposed_counts),
    "total_safety_blocks": total_blocks,
    "blocked_by_proposed_action": dict(blocked_by_action),
    "approval_pending_blocks": approval_blocks,
    "technical_violation_counts": dict(technical_reasons),
    "blocked_by_initial_risk_band": dict(blocked_by_risk_band),
    "applied_migration_actions": applied_migrations,
    "completed_migrations": completed_migrations,
    "example_blocked_decisions": examples,
    "limitations": [
        "Synthetic evaluation only",
        "Single trained DQN policy",
        "Technical violation counts can overlap",
        "Risk bands use initial episode risk",
        "No real HDD operations performed",
    ],
}

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(report, file, indent=4)


print("=" * 62)
print("STEP 154 - DQN V2 SAFETY BLOCK AUDIT")
print("=" * 62)

print("Total proposals:", report["total_proposals"])
print("Total safety blocks:", total_blocks)
print("Approval-pending blocks:", approval_blocks)

print("\nBlocked proposed actions:")
for action, count in blocked_by_action.most_common():
    print(f"  {action}: {count}")

print("\nTechnical safety violations:")
for reason, count in technical_reasons.most_common():
    print(f"  {reason}: {count}")

print("\nBlocks by initial risk band:")
for band, count in blocked_by_risk_band.most_common():
    print(f"  {band}: {count}")

print("\nApplied migration actions:", applied_migrations)
print("Completed migrations:", completed_migrations)

print("\nReport:", OUTPUT_FILE)
print("STEP 154 COMPLETE")
