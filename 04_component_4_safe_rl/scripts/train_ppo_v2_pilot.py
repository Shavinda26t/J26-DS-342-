
"""
Step 148 - PPO V2 Pilot

Compare PPO with NO_OP and SAFE_MIGRATION_FIRST.

Synthetic research simulation only.
No real HDD operations.
"""

from collections import Counter
from datetime import datetime, timezone
from statistics import mean
import json

import gymnasium
import stable_baselines3
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Import configuration first to register Component 4 src.
from v2_training_config import (
    ROOT,
    TRAIN_SEED,
    PILOT_TRAINING_STEPS,
    EVALUATION_SEED_START,
    EVALUATION_EPISODES,
    ENVIRONMENT_CONFIG,
    make_v2_env,
)

from contracts import MitigationAction


MODEL_FILE = (
    ROOT / "models" / "148_ppo_v2_pilot_10000.zip"
)

RESULT_FILE = (
    ROOT / "results" / "148_ppo_v2_pilot_results.json"
)

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

if MODEL_FILE.exists() or RESULT_FILE.exists():
    raise FileExistsError(
        "Step 148 artifacts already exist. "
        "Do not overwrite previous evidence."
    )


print("=" * 60)
print("STEP 148 - PPO V2 PILOT TRAINING")
print("=" * 60)

print("Environment:", ENVIRONMENT_CONFIG)
print("Requested training steps:", PILOT_TRAINING_STEPS)
print("Training seed:", TRAIN_SEED)
print("Evaluation episodes:", EVALUATION_EPISODES)


# --------------------------------------------------
# 1. PPO model and training
# --------------------------------------------------

training_env = Monitor(make_v2_env())

model = PPO(
    "MlpPolicy",
    training_env,
    learning_rate=0.0003,
    n_steps=256,
    batch_size=64,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    seed=TRAIN_SEED,
    device="cpu",
    verbose=1,
)

try:
    model.learn(
        total_timesteps=PILOT_TRAINING_STEPS,
        progress_bar=False,
    )

    actual_steps = int(model.num_timesteps)

    model.save(str(MODEL_FILE))

finally:
    training_env.close()

model = PPO.load(
    str(MODEL_FILE),
    device="cpu",
)

print("\nPPO checkpoint reloaded.")
print("Actual training steps:", actual_steps)


# --------------------------------------------------
# 2. Policy selection
# --------------------------------------------------

def choose_action(policy, env, observation):

    if policy == "PPO_V2":
        action, _ = model.predict(
            observation,
            deterministic=True,
        )
        return int(action)

    if policy == "NO_OP":
        return env.ACTIONS.index(
            MitigationAction.NO_OP
        )

    if policy == "SAFE_MIGRATION_FIRST":

        if env.migration_progress < 1.0:

            assessment = env.validator.assess(
                env._get_drive_state(),
                MitigationAction.MIGRATE_DATA,
            )

            if (
                assessment.technically_eligible
                and not assessment.approval_pending
            ):
                return env.ACTIONS.index(
                    MitigationAction.MIGRATE_DATA
                )

        return env.ACTIONS.index(
            MitigationAction.NO_OP
        )

    raise ValueError("Unknown policy")


# --------------------------------------------------
# 3. Matched-seed evaluation
# --------------------------------------------------

def evaluate(policy):

    returns = []
    failure_episodes = 0
    safety_blocks = 0
    completed_migrations = 0

    proposed_actions = Counter()
    applied_actions = Counter()

    for offset in range(EVALUATION_EPISODES):

        seed = EVALUATION_SEED_START + offset
        env = make_v2_env()

        try:
            observation, _ = env.reset(seed=seed)
            episode_reward = 0.0
            done = False

            while not done:

                action = choose_action(
                    policy,
                    env,
                    observation,
                )

                observation, reward, terminated, truncated, info = (
                    env.step(action)
                )

                episode_reward += float(reward)

                proposed_actions[
                    info["proposed_action"]
                ] += 1

                applied_actions[
                    info["applied_action"]
                ] += 1

                safety_blocks += int(
                    info["blocked_by_safety"]
                )

                failure_episodes += int(
                    info["failed"]
                )

                done = terminated or truncated

            completed_migrations += int(
                env.migration_progress >= 1.0
            )

            returns.append(episode_reward)

        finally:
            env.close()

    return {
        "policy": policy,
        "evaluation_episodes": EVALUATION_EPISODES,
        "mean_episode_reward": round(
            mean(returns), 5
        ),
        "failure_episodes": failure_episodes,
        "safety_blocks": safety_blocks,
        "completed_migrations": completed_migrations,
        "proposed_actions": dict(proposed_actions),
        "applied_actions": dict(applied_actions),
        "episode_returns": returns,
    }


# --------------------------------------------------
# 4. Evaluate the three policies
# --------------------------------------------------

results = []

for policy in [
    "PPO_V2",
    "NO_OP",
    "SAFE_MIGRATION_FIRST",
]:

    print("\nEvaluating:", policy, flush=True)

    result = evaluate(policy)
    results.append(result)

    print("Mean reward:", result["mean_episode_reward"])
    print("Failure episodes:", result["failure_episodes"])
    print("Safety blocks:", result["safety_blocks"])
    print("Completed migrations:", result["completed_migrations"])
    print("Proposed actions:", result["proposed_actions"])


# --------------------------------------------------
# 5. Save experiment evidence
# --------------------------------------------------

report = {
    "experiment": "C4_PPO_V2_PILOT_148",
    "status": "EXPLORATORY_PILOT",
    "simulation_only": True,
    "algorithm": "PPO",
    "requested_training_steps": PILOT_TRAINING_STEPS,
    "actual_training_steps": actual_steps,
    "training_seed": TRAIN_SEED,
    "evaluation_seed_start": EVALUATION_SEED_START,
    "evaluation_episodes": EVALUATION_EPISODES,
    "environment": "SafeHDDEnvV2",
    "environment_configuration": ENVIRONMENT_CONFIG,
    "hyperparameters": {
        "learning_rate": 0.0003,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
    },
    "versions": {
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
    },
    "results": results,
    "limitations": [
        "Synthetic and uncalibrated HDD failure dynamics",
        "Stratified risks do not represent fleet prevalence",
        "Single training seed and short pilot",
        "No real HDD operations",
        "No final performance claim",
    ],
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat(),
}

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(report, file, indent=4)


print("\n" + "=" * 60)
print("STEP 148 - PPO V2 PILOT COMPLETED")
print("=" * 60)

print("Model:", MODEL_FILE)
print("Report:", RESULT_FILE)
print("NOT A FINAL SCIENTIFIC PERFORMANCE CLAIM")
