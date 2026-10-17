
"""
Component 4 - A2C Pilot Training

Research purpose:
Verify A2C training, checkpointing and evaluation
in the SafeHDDEnv synthetic simulator.

Pilot only. No real HDD actions.
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

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hdd_rl_env import SafeHDDEnv
from contracts import MitigationAction

# -------------------------------------------------------
# 1. Reproducible experiment configuration
# -------------------------------------------------------

SEED = 42
TRAIN_STEPS = 5000
EPISODE_DAYS = 30
EVAL_SEEDS = list(range(9000, 9030))

MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "93_a2c_pilot_5000.zip"
RESULT_PATH = RESULT_DIR / "93_a2c_pilot_results.json"

if MODEL_PATH.exists() or RESULT_PATH.exists():
    raise FileExistsError(
        "A2C pilot artifacts already exist. "
        "Preserve previous experimental results."
    )

print("=" * 65)
print("STEP 93 - A2C PILOT TRAINING")
print("=" * 65)

# -------------------------------------------------------
# 2. Initialize environment and A2C algorithm
# -------------------------------------------------------

env = Monitor(
    SafeHDDEnv(episode_days=EPISODE_DAYS)
)

PARAMETERS = {
    "learning_rate": 0.0003,
    "n_steps": 32,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5
}

model = A2C(
    policy="MlpPolicy",
    env=env,
    seed=SEED,
    device="cpu",
    verbose=1,
    **PARAMETERS
)

# -------------------------------------------------------
# 3. Train A2C
# -------------------------------------------------------

print("\nStarting A2C training...", flush=True)

try:
    model.learn(
        total_timesteps=TRAIN_STEPS,
        progress_bar=False
    )

    actual_steps = int(model.num_timesteps)
    model.save(str(MODEL_PATH))

finally:
    env.close()

print("\nA2C training completed.")
print("Actual training steps:", actual_steps)

# Reload trained checkpoint
model = A2C.load(
    str(MODEL_PATH),
    device="cpu"
)

print("Checkpoint reloaded successfully.")

# -------------------------------------------------------
# 4. Matched-seed evaluation
# -------------------------------------------------------

def evaluate(policy):

    episode_returns = []
    failure_episodes = 0
    safety_blocks = 0
    action_counts = Counter()

    for seed in EVAL_SEEDS:

        test_env = SafeHDDEnv(
            episode_days=EPISODE_DAYS
        )

        try:
            obs, _ = test_env.reset(seed=seed)

            done = False
            total_reward = 0.0

            while not done:

                if policy == "A2C_PILOT":

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

                action_counts[
                    info["proposed_action"]
                ] += 1

                safety_blocks += int(
                    info["blocked_by_safety"]
                )

                failure_episodes += int(
                    info["failed"]
                )

                done = terminated or truncated

            episode_returns.append(total_reward)

        finally:
            test_env.close()

    return {
        "policy": policy,
        "evaluation_episodes": len(EVAL_SEEDS),
        "mean_episode_reward": round(
            mean(episode_returns), 5
        ),
        "failure_episodes": failure_episodes,
        "safety_blocks": safety_blocks,
        "proposed_actions": dict(action_counts),
        "episode_returns": episode_returns
    }


print("\nEvaluating A2C...", flush=True)
a2c_result = evaluate("A2C_PILOT")

print("Evaluating NO_OP...", flush=True)
noop_result = evaluate("NO_OP")

# -------------------------------------------------------
# 5. Preserve research evidence
# -------------------------------------------------------

report = {
    "experiment": "C4_A2C_PILOT_93",
    "status": "EXPLORATORY_PILOT",
    "simulation_only": True,
    "algorithm": "A2C",
    "requested_training_steps": TRAIN_STEPS,
    "actual_training_steps": actual_steps,
    "training_seed": SEED,
    "evaluation_seeds": EVAL_SEEDS,
    "episode_days": EPISODE_DAYS,
    "hyperparameters": PARAMETERS,
    "versions": {
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__
    },
    "results": [
        a2c_result,
        noop_result
    ],
    "limitations": [
        "Synthetic environment is not calibrated",
        "Single training seed",
        "Short pilot run",
        "Thirty evaluation episodes",
        "No final policy selection",
        "No real storage actions performed"
    ],
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat()
}

with open(RESULT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4)

# -------------------------------------------------------
# 6. Display evaluation
# -------------------------------------------------------

print("\n" + "=" * 65)
print("STEP 93 - A2C PILOT EVALUATION")
print("=" * 65)

for item in [a2c_result, noop_result]:

    print("\nPolicy:", item["policy"])
    print("Mean reward:", item["mean_episode_reward"])
    print("Failure episodes:", item["failure_episodes"])
    print("Safety blocks:", item["safety_blocks"])
    print("Proposed actions:", item["proposed_actions"])

print("\nA2C PILOT COMPLETED")
print("Model:", MODEL_PATH)
print("Evidence:", RESULT_PATH)
print("NOT A FINAL SCIENTIFIC PERFORMANCE CLAIM")
