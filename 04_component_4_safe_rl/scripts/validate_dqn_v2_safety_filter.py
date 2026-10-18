
"""
Step 155 - Safety-Filtered DQN Evaluation

Compare RAW_DQN, FILTERED_DQN, NO_OP,
and SAFE_MIGRATION_FIRST on identical seeds.

The filtering layer checks migration eligibility
before execution, but does not bypass or replace
the independent environment safety validator.

Synthetic research only. No real HDD operations.
"""

import json
from collections import Counter
from statistics import mean

from stable_baselines3 import DQN


from v2_training_config import (
    ROOT,
    make_v2_env,
)

# Fresh seeds not used in previous pilot comparisons.
EVALUATION_SEED_START = 70000
EVALUATION_EPISODES = 200


from contracts import MitigationAction


MODEL_FILE = (
    ROOT / "models" / "153_dqn_v2_actionable_10000.zip"
)

RESULT_FILE = (
    ROOT / "results"
   / "156_dqn_v2_safety_filter_holdout.json"
)

if not MODEL_FILE.exists():
    raise FileNotFoundError(MODEL_FILE)

if RESULT_FILE.exists():
    raise FileExistsError(
        "Step 155 results already exist. "
        "Do not overwrite research evidence."
    )

model = DQN.load(str(MODEL_FILE), device="cpu")


def migration_permitted(env):

    if env.migration_progress >= 1.0:
        return False

    assessment = env.validator.assess(
        env._get_drive_state(),
        MitigationAction.MIGRATE_DATA,
    )

    return bool(
        assessment.technically_eligible
        and not assessment.approval_pending
    )


def choose_action(policy, env, observation):

    noop = env.ACTIONS.index(
        MitigationAction.NO_OP
    )

    migrate = env.ACTIONS.index(
        MitigationAction.MIGRATE_DATA
    )

    if policy == "NO_OP":
        return noop, False

    if policy == "SAFE_MIGRATION_FIRST":

        if migration_permitted(env):
            return migrate, False

        return noop, False

    if policy in ("RAW_DQN", "FILTERED_DQN"):

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        action = int(action)

        if policy == "FILTERED_DQN":

            if (
                action == migrate
                and not migration_permitted(env)
            ):
                return noop, True

        return action, False

    raise ValueError("Unknown policy")


def evaluate(policy):

    rewards = []
    blocks = 0
    failures = 0
    completed = 0
    filtered_proposals = 0

    proposed_actions = Counter()
    applied_actions = Counter()

    for offset in range(EVALUATION_EPISODES):

        seed = EVALUATION_SEED_START + offset

        env = make_v2_env()

        try:
            observation, _ = env.reset(seed=seed)

            total_reward = 0.0
            done = False

            while not done:

                action, filtered = choose_action(
                    policy,
                    env,
                    observation,
                )

                filtered_proposals += int(filtered)

                observation, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += float(reward)

                blocks += int(
                    info["blocked_by_safety"]
                )

                failures += int(
                    info["failed"]
                )

                proposed_actions[
                    info["proposed_action"]
                ] += 1

                applied_actions[
                    info["applied_action"]
                ] += 1

                done = terminated or truncated

            rewards.append(total_reward)

            completed += int(
                env.migration_progress >= 1.0
            )

        finally:
            env.close()

    return {
        "policy": policy,
        "evaluation_episodes": EVALUATION_EPISODES,
        "mean_episode_reward": round(
            mean(rewards), 5
        ),
        "failure_episodes": failures,
        "safety_blocks": blocks,
        "completed_migrations": completed,
        "filtered_migration_proposals": filtered_proposals,
        "proposed_actions": dict(proposed_actions),
        "applied_actions": dict(applied_actions),
        "episode_returns": rewards,
    }


print("=" * 64)
print("STEP 156 - SAFETY-FILTERED DQN EVALUATION")
print("=" * 64)

results = []

for policy in (
    "RAW_DQN",
    "FILTERED_DQN",
    "NO_OP",
    "SAFE_MIGRATION_FIRST",
):

    result = evaluate(policy)

    results.append(result)

    print("\nPolicy:", policy)
    print("Mean reward:", result["mean_episode_reward"])
    print("Failure episodes:", result["failure_episodes"])
    print("Safety blocks:", result["safety_blocks"])
    print("Completed migrations:", result["completed_migrations"])

    print(
        "Filtered migration proposals:",
        result["filtered_migration_proposals"]
    )

    print("Proposed actions:", result["proposed_actions"])


report = {
    "experiment": "C4_DQN_V2_SAFETY_FILTER_HOLDOUT_156",
    "simulation_only": True,
    "model_file": MODEL_FILE.name,
    "evaluation_seed_start": EVALUATION_SEED_START,
    "evaluation_episodes": EVALUATION_EPISODES,
    "filter": (
        "Replace migration with NO_OP when migration "
        "is not eligible, not approved, or already complete"
    ),
    "independent_validator_preserved": True,
    "results": results,
    "limitations": [
        "Synthetic uncalibrated simulation",
        "Single trained DQN checkpoint",
        "Filtering is an external rule, not learned safety",
        "Same held-out seeds used in prior pilot evaluations",
        "No real HDD operations or deployment claims",
    ],
}

RESULT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(report, file, indent=4)

print("\nReport saved:", RESULT_FILE)
print("STEP 156 COMPLETE")
