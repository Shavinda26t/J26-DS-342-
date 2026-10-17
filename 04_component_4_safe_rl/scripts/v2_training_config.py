
"""
Component 4 - Simulator V2 RL Experiment Configuration

Shared configuration for DQN, PPO and A2C.

Purpose:
- Use identical synthetic environment settings.
- Separate training and evaluation random seeds.
- Make experimental comparisons reproducible.

IMPORTANT:
This is a synthetic research simulator, not a
calibrated HDD failure model or production digital twin.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hdd_rl_env_v2 import SafeHDDEnvV2


EXPERIMENT_NAME = "C4_V2_STRATIFIED_RL_PILOT"

TRAIN_SEED = 42

EVALUATION_SEED_START = 30000
EVALUATION_EPISODES = 100

PILOT_TRAINING_STEPS = 10000


# Identical experimental environment for all agents.
ENVIRONMENT_CONFIG = {
    "episode_days": 30,

    # Sample across three synthetic risk bands.
    "initial_risk_mode": "stratified",

    # Stable-risk experiment:
    # isolate mitigation decisions from daily risk drift.
    "risk_drift_per_day": 0.0,
    "risk_noise_std": 0.0,
}


def make_v2_env():
    """Create a fresh, identically configured V2 environment."""

    return SafeHDDEnvV2(
        **ENVIRONMENT_CONFIG
    )


if __name__ == "__main__":

    from stable_baselines3.common.env_checker import (
        check_env
    )

    env = make_v2_env()

    try:
        check_env(env, warn=True)

        counts = {
            "VERY_LOW": 0,
            "MODERATE": 0,
            "HIGH": 0,
        }

        for seed in range(30000, 30300):

            env.reset(seed=seed)

            if env.risk < 0.012:
                counts["VERY_LOW"] += 1

            elif env.risk < 0.15:
                counts["MODERATE"] += 1

            else:
                counts["HIGH"] += 1

        assert sum(counts.values()) == 300
        assert all(count > 0 for count in counts.values())

        print("=" * 60)
        print("STEP 145 - V2 RL TRAINING CONFIGURATION")
        print("=" * 60)

        print("Experiment:", EXPERIMENT_NAME)
        print("Training seed:", TRAIN_SEED)
        print("Pilot training steps:", PILOT_TRAINING_STEPS)
        print("Evaluation episodes:", EVALUATION_EPISODES)
        print("Evaluation seed start:", EVALUATION_SEED_START)

        print("\nEnvironment configuration:")

        for key, value in ENVIRONMENT_CONFIG.items():
            print(f"  {key}: {value}")

        print("\nInitial risk band samples (300 resets):")

        for band, count in counts.items():
            print(f"  {band}: {count}")

        print("\nGymnasium compatibility: PASSED")
        print("STEP 145 COMPLETE")

    finally:
        env.close()
