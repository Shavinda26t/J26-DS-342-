
"""
Component 4 - DQN Pilot Training

Purpose:
Verify that DQN can train, save, reload, and interact
with our synthetic Safe HDD RL environment.

This is an exploratory pilot, not final policy evaluation.

No physical HDD operations are performed.
"""

from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from statistics import mean
import json
import sys

import gymnasium
import stable_baselines3
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contracts import MitigationAction
from hdd_rl_env import SafeHDDEnv


# ---------------------------------------------------------
# 1. Reproducible experiment configuration
# ---------------------------------------------------------

SEED = 42
TRAINING_STEPS = 5000
EPISODE_DAYS = 30
EVALUATION_SEEDS = list(range(9000, 9030))

MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "91_dqn_pilot_5000.zip"
RESULT_FILE = RESULT_DIR / "91_dqn_pilot_results.json"

# Avoid silently overwriting previous experimental evidence.
if MODEL_FILE.exists() or RESULT_FILE.exists():
    raise FileExistsError(
        "Pilot artifacts already exist. Preserve them "
        "before starting another run."
    )


# ---------------------------------------------------------
# 2. Initialize the synthetic training environment
# ---------------------------------------------------------

print("=" * 65)
print("STEP 91 - DQN PILOT TRAINING")
print("=" * 65)

print("Environment: SafeHDDEnv")
print("Algorithm: DQN")
print("Training steps:", TRAINING_STEPS)
print("Random seed:", SEED)
print("Device: CPU")

train_env = Monitor(
    SafeHDDEnv(episode_days=EPISODE_DAYS)
)


# ---------------------------------------------------------
# 3. Configure the DQN agent
# ---------------------------------------------------------

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
    seed=SEED,
    device="cpu",
    verbose=1
)


# ---------------------------------------------------------
# 4. Train the DQN policy
# ---------------------------------------------------------

print("\nStarting DQN training...", flush=True)

try:
    model.learn(
        total_timesteps=TRAINING_STEPS,
        progress_bar=False
    )

    model.save(str(MODEL_FILE))

finally:
    train_env.close()

print("\nTraining completed.")
print("Model saved:", MODEL_FILE)


# ---------------------------------------------------------
# 5. Reload the trained model
# ---------------------------------------------------------

model = DQN.load(
    str(MODEL_FILE),
    device="cpu"
)

print("Saved DQN model reloaded successfully.")


# ---------------------------------------------------------
# 6. Evaluate on separate simulation seeds
# ---------------------------------------------------------

def evaluate(policy_name):

    episode_rewards = []
    failure_episodes = 0
    safety_blocks = 0
    action_counts = Counter()

    for seed in EVALUATION_SEEDS:

        env = SafeHDDEnv(
            episode_days=EPISODE_DAYS
        )

        try:
            observation, _ = env.reset(seed=seed)

            total_reward = 0.0
            done = False

            while not done:

                if policy_name == "DQN_PILOT":
                    action, _ = model.predict(
                        observation,
                        deterministic=True
                    )
                    action = int(action)

                elif policy_name == "NO_OP":
                    action = env.ACTIONS.index(
                        MitigationAction.NO_OP
                    )

                else:
                    raise ValueError("Unknown policy")

                observation, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += float(reward)

                action_counts[
                    info["proposed_action"]
                ] += 1

                if info["blocked_by_safety"]:
                    safety_blocks += 1

                if info["failed"]:
                    failure_episodes += 1

                done = terminated or truncated

            episode_rewards.append(total_reward)

        finally:
            env.close()

    return {
        "policy": policy_name,
        "evaluation_episodes": len(EVALUATION_SEEDS),
        "mean_episode_reward": round(
            mean(episode_rewards), 5
        ),
        "failure_episodes": failure_episodes,
        "safety_blocks": safety_blocks,
        "proposed_action_counts": dict(action_counts),
    }


print("\nEvaluating DQN pilot...", flush=True)
dqn_result = evaluate("DQN_PILOT")

print("Evaluating NO_OP baseline...", flush=True)
noop_result = evaluate("NO_OP")


# ---------------------------------------------------------
# 7. Save reproducible experiment evidence
# ---------------------------------------------------------

report = {
    "experiment": "C4_DQN_PILOT_91",
    "status": "EXPLORATORY_PILOT",
    "simulation_only": True,
    "training_steps": TRAINING_STEPS,
    "training_seed": SEED,
    "evaluation_seeds": EVALUATION_SEEDS,
    "episode_days": EPISODE_DAYS,
    "algorithm": "DQN",
    "network": "MlpPolicy",
    "device": "cpu",
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
    "results": [
        dqn_result,
        noop_result
    ],
    "limitations": [
        "Synthetic environment with provisional dynamics",
        "Single training seed",
        "Short training run",
        "Only 30 evaluation episodes per policy",
        "Pilot evaluation seeds are separate from training",
        "No real-world operational validity established",
        "No final model selection performed"
    ],
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat()
}

with open(RESULT_FILE, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)


# ---------------------------------------------------------
# 8. Display results
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("DQN PILOT - PRELIMINARY EVALUATION")
print("=" * 65)

for result in [dqn_result, noop_result]:

    print("\nPolicy:", result["policy"])

    print(
        "Mean episode reward:",
        result["mean_episode_reward"]
    )

    print(
        "Failure episodes:",
        result["failure_episodes"]
    )

    print(
        "Safety blocks:",
        result["safety_blocks"]
    )

    print(
        "Proposed actions:",
        result["proposed_action_counts"]
    )

print("\nSTATUS: DQN PILOT COMPLETED")
print("Model:", MODEL_FILE)
print("Evidence:", RESULT_FILE)
print("\nNOT A FINAL SCIENTIFIC PERFORMANCE CLAIM")
