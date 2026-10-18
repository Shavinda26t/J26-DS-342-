
"""
Component 4 - PPO Pilot Training

Research objective:
Verify PPO training, checkpointing, and policy evaluation
on the synthetic SafeHDDEnv environment.

Pilot only - not final model comparison.
No physical HDD operations.
"""

from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from statistics import mean
import json
import sys

import torch
import gymnasium
import stable_baselines3

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hdd_rl_env import SafeHDDEnv
from contracts import MitigationAction

# -------------------------------------------------------
# 1. Reproducible configuration
# -------------------------------------------------------

SEED = 42
TRAIN_STEPS = 5000
EPISODE_DAYS = 30
EVAL_SEEDS = list(range(9000, 9030))

MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "92_ppo_pilot_5000.zip"
RESULT_PATH = RESULT_DIR / "92_ppo_pilot_results.json"

if MODEL_PATH.exists() or RESULT_PATH.exists():
    raise FileExistsError(
        "PPO pilot artifacts already exist. "
        "Preserve them before another experiment."
    )

print("=" * 65)
print("STEP 92 - PPO PILOT TRAINING")
print("=" * 65)

# -------------------------------------------------------
# 2. Build training environment
# -------------------------------------------------------

env = Monitor(
    SafeHDDEnv(episode_days=EPISODE_DAYS)
)

# -------------------------------------------------------
# 3. Configure PPO
# -------------------------------------------------------

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0003,
    n_steps=256,
    batch_size=64,
    n_epochs=5,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    seed=SEED,
    device="cpu",
    verbose=1
)

# -------------------------------------------------------
# 4. Train and save
# -------------------------------------------------------

print("\nStarting PPO training...", flush=True)

try:
    model.learn(
        total_timesteps=TRAIN_STEPS,
        progress_bar=False
    )

    actual_steps = int(model.num_timesteps)
    model.save(str(MODEL_PATH))

finally:
    env.close()

print("\nPPO training completed.")
print("Actual training steps:", actual_steps)
print("Model saved:", MODEL_PATH)

# Reload checkpoint
model = PPO.load(
    str(MODEL_PATH),
    device="cpu"
)

print("Checkpoint reloaded successfully.")

# -------------------------------------------------------
# 5. Evaluate matched-seed policies
# -------------------------------------------------------

def evaluate(policy):

    rewards = []
    failures = 0
    safety_blocks = 0
    actions = Counter()

    for seed in EVAL_SEEDS:

        test_env = SafeHDDEnv(
            episode_days=EPISODE_DAYS
        )

        try:
            obs, _ = test_env.reset(seed=seed)
            done = False
            total_reward = 0.0

            while not done:

                if policy == "PPO_PILOT":
                    action, _ = model.predict(
                        obs,
                        deterministic=True
                    )
                    action = int(action)

                elif policy == "NO_OP":
                    action = test_env.ACTIONS.index(
                        MitigationAction.NO_OP
                    )

                else:
                    raise ValueError("Unknown policy")

                obs, reward, terminated, truncated, info = (
                    test_env.step(action)
                )

                total_reward += float(reward)

                actions[info["proposed_action"]] += 1

                safety_blocks += int(
                    info["blocked_by_safety"]
                )

                failures += int(info["failed"])

                done = terminated or truncated

            rewards.append(total_reward)

        finally:
            test_env.close()

    return {
        "policy": policy,
        "evaluation_episodes": len(EVAL_SEEDS),
        "mean_episode_reward": round(mean(rewards), 5),
        "failure_episodes": failures,
        "safety_blocks": safety_blocks,
        "proposed_actions": dict(actions),
        "episode_returns": rewards
    }


print("\nEvaluating PPO...", flush=True)
ppo_result = evaluate("PPO_PILOT")

print("Evaluating NO_OP...", flush=True)
noop_result = evaluate("NO_OP")

# -------------------------------------------------------
# 6. Save complete experimental report
# -------------------------------------------------------

report = {
    "experiment": "C4_PPO_PILOT_92",
    "status": "EXPLORATORY_PILOT",
    "environment": "SafeHDDEnv",
    "simulation_only": True,
    "algorithm": "PPO",
    "requested_training_steps": TRAIN_STEPS,
    "actual_training_steps": actual_steps,
    "training_seed": SEED,
    "evaluation_seeds": EVAL_SEEDS,
    "episode_days": EPISODE_DAYS,
    "hyperparameters": {
        "learning_rate": 0.0003,
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01
    },
    "versions": {
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__
    },
    "results": [ppo_result, noop_result],
    "limitations": [
        "Synthetic uncalibrated environment",
        "Single training seed",
        "Short pilot training",
        "No final policy selection",
        "Only 30 evaluation episodes",
        "No real storage actions performed"
    ],
    "created_utc": datetime.now(timezone.utc).isoformat()
}

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)

# -------------------------------------------------------
# 7. Display comparison
# -------------------------------------------------------

print("\n" + "=" * 65)
print("STEP 92 - PPO PILOT EVALUATION")
print("=" * 65)

for item in [ppo_result, noop_result]:

    print("\nPolicy:", item["policy"])
    print("Mean reward:", item["mean_episode_reward"])
    print("Failure episodes:", item["failure_episodes"])
    print("Safety blocks:", item["safety_blocks"])
    print("Proposed actions:", item["proposed_actions"])

print("\nPPO PILOT COMPLETED")
print("Model:", MODEL_PATH)
print("Evidence:", RESULT_PATH)
print("NOT A FINAL SCIENTIFIC PERFORMANCE CLAIM")
