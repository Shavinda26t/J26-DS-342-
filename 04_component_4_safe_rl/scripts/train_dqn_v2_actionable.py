
"""
Step 146 - DQN Pilot on SafeHDDEnvV2

Train DQN with stratified synthetic HDD risk sampling.

Evaluate:
1. DQN_V2
2. NO_OP
3. SAFE_MIGRATION_FIRST

This is a research pilot only.
No real HDD operations are performed.
"""

from pathlib import Path
from collections import Counter
from statistics import mean
from datetime import datetime, timezone
import json

import gymnasium
import stable_baselines3
import torch

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor


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
from actionable_start_wrapper import ActionableStartWrapper


# ---------------------------------------------------------
# 1. Experiment configuration
# ---------------------------------------------------------

MODEL_FILE = (
    ROOT / "models" / "153_dqn_v2_actionable_10000.zip"
)

RESULT_FILE = (
    ROOT / "results" / "153_dqn_v2_actionable_results.json"
)

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

if MODEL_FILE.exists() or RESULT_FILE.exists():
    raise FileExistsError(
        "Experiment 146 artifacts already exist. "
        "Do not overwrite research evidence."
    )


# ---------------------------------------------------------
# 2. Train DQN
# ---------------------------------------------------------

print("=" * 65)
print("STEP 153 - DQN V2 PILOT TRAINING")
print("=" * 65)

print("Algorithm: DQN")
print("Environment: SafeHDDEnvV2")
print("Training steps:", PILOT_TRAINING_STEPS)
print("Training seed:", TRAIN_SEED)
print("Evaluation episodes:", EVALUATION_EPISODES)


train_env = Monitor(
    ActionableStartWrapper(
        make_v2_env(),
        max_attempts=100,
    )
)


model = DQN(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=0.0003,
    buffer_size=10000,
    learning_starts=500,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=500,
    exploration_fraction=0.30,
    exploration_final_eps=0.10,
    seed=TRAIN_SEED,
    device="cpu",
    verbose=1,
)

print("\nTraining DQN...", flush=True)

try:
    model.learn(
        total_timesteps=PILOT_TRAINING_STEPS,
        progress_bar=False,
    )

    model.save(str(MODEL_FILE))

finally:
    train_env.close()

print("\nModel saved:", MODEL_FILE)

# Verify the saved policy can be reloaded.
model = DQN.load(str(MODEL_FILE), device="cpu")

print("DQN model reloaded successfully.")


# ---------------------------------------------------------
# 3. Choose actions for each evaluation policy
# ---------------------------------------------------------

def select_action(policy_name, env, observation):

    if policy_name == "DQN_V2":
        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        return int(action)

    if policy_name == "NO_OP":
        return env.ACTIONS.index(
            MitigationAction.NO_OP
        )

    if policy_name == "SAFE_MIGRATION_FIRST":

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

    raise ValueError("Unknown evaluation policy")


# ---------------------------------------------------------
# 4. Evaluate using held-out simulation seeds
# ---------------------------------------------------------

def evaluate(policy_name):

    episode_rewards = []
    failure_episodes = 0
    safety_blocks = 0
    completed_migrations = 0

    proposed_actions = Counter()
    applied_actions = Counter()

    for i in range(EVALUATION_EPISODES):

        seed = EVALUATION_SEED_START + i
        env = make_v2_env()

        try:
            observation, _ = env.reset(seed=seed)

            episode_reward = 0.0
            done = False

            while not done:

                action = select_action(
                    policy_name,
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

                if info["failed"]:
                    failure_episodes += 1

                done = terminated or truncated

            if env.migration_progress >= 1.0:
                completed_migrations += 1

            episode_rewards.append(episode_reward)

        finally:
            env.close()

    return {
        "policy": policy_name,
        "evaluation_episodes": EVALUATION_EPISODES,
        "mean_episode_reward": round(
            mean(episode_rewards), 5
        ),
        "failure_episodes": failure_episodes,
        "safety_blocks": safety_blocks,
        "completed_migrations": completed_migrations,
        "proposed_actions": dict(proposed_actions),
        "applied_actions": dict(applied_actions),
    }


# ---------------------------------------------------------
# 5. Compare DQN against two baselines
# ---------------------------------------------------------

results = []

for policy in [
    "DQN_V2",
    "NO_OP",
    "SAFE_MIGRATION_FIRST",
]:

    print(f"\nEvaluating {policy}...", flush=True)

    result = evaluate(policy)
    results.append(result)

    print("Mean reward:", result["mean_episode_reward"])
    print("Failure episodes:", result["failure_episodes"])
    print("Safety blocks:", result["safety_blocks"])
    print("Completed migrations:", result["completed_migrations"])
    print("Proposed actions:", result["proposed_actions"])


# ---------------------------------------------------------
# 6. Save reproducible research evidence
# ---------------------------------------------------------

report = {
    "experiment": "C4_DQN_V2_ACTIONABLE_153",
    "status": "EXPLORATORY_PILOT",
    "simulation_only": True,
    "algorithm": "DQN",
    "training_seed": TRAIN_SEED,
    "training_steps": PILOT_TRAINING_STEPS,
    "evaluation_seed_start": EVALUATION_SEED_START,
    "evaluation_episodes": EVALUATION_EPISODES,
    "environment": "SafeHDDEnvV2",
    "environment_configuration": ENVIRONMENT_CONFIG,
    
    "training_start_distribution": "ACTIONABLE_ONLY",
    "training_wrapper": "ActionableStartWrapper",
    "training_wrapper_max_attempts": 100,
    "evaluation_start_distribution": "ORIGINAL_UNFILTERED_STRATIFIED",

    "hyperparameters": {
        "learning_rate": 0.0003,
        "buffer_size": 10000,
        "learning_starts": 500,
        "batch_size": 64,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 500,
        "exploration_fraction": 0.30,
        "exploration_final_eps": 0.10,
    },
    "versions": {
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
    },
    "results": results,
    "limitations": [
        "Synthetic, uncalibrated HDD failure dynamics",
        "Equal risk-band sampling is not fleet prevalence",
        "Only one RL training seed",
        "Short pilot training duration",
        "Evaluation uses 100 matched initial seeds",
        "Mitigation reduces consequences, not physical failures",
        "No evidence of production safety or effectiveness",
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


print("\n" + "=" * 65)
print("STEP 153 - DQN V2 PILOT COMPLETED")
print("=" * 65)

print("Model:", MODEL_FILE)
print("Results:", RESULT_FILE)
print("NOT A FINAL SCIENTIFIC PERFORMANCE CLAIM")
