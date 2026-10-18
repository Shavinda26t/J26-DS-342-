
"""
Step 152 - Actionable-Start Training Wrapper

Purpose:
Investigate whether increased exposure to safe,
approved migration opportunities helps RL learning.

This changes the TRAINING distribution only.

The independent safety validator remains active.
The original evaluation distribution remains unchanged.

Synthetic research experiment only.
No real HDD operations.
"""

from collections import Counter

import gymnasium as gym

# Import configuration first so Component 4 src
# is registered on the Python import path.
from v2_training_config import make_v2_env

from contracts import MitigationAction


class ActionableStartWrapper(gym.Wrapper):
    """
    Reset until migration is technically eligible
    and simulated administrator approval is present.

    This is rejection sampling, not an override
    of storage safety constraints.
    """

    def __init__(self, env, max_attempts=100):

        super().__init__(env)

        if (
            type(max_attempts) is not int
            or max_attempts <= 0
        ):
            raise ValueError(
                "max_attempts must be a positive integer"
            )

        self.max_attempts = max_attempts

    def reset(self, *, seed=None, options=None):

        for attempt in range(self.max_attempts):

            # Seed only the first reset.
            # Subsequent attempts advance the same RNG.
            observation, info = self.env.reset(
                seed=seed if attempt == 0 else None,
                options=options,
            )

            base_env = self.env.unwrapped

            assessment = base_env.validator.assess(
                base_env._get_drive_state(),
                MitigationAction.MIGRATE_DATA,
            )

            actionable = (
                assessment.technically_eligible
                and not assessment.approval_pending
            )

            if actionable:

                info = dict(info)

                info["actionable_training_start"] = True
                info["sampling_attempts"] = attempt + 1

                return observation, info

        raise RuntimeError(
            "Could not sample an actionable starting "
            "state within max_attempts."
        )


if __name__ == "__main__":

    print("=" * 60)
    print("STEP 152 - ACTIONABLE TRAINING WRAPPER AUDIT")
    print("=" * 60)

    env = ActionableStartWrapper(make_v2_env())

    counts = Counter()
    attempts = []

    try:

        for seed in range(60000, 60300):

            observation, info = env.reset(seed=seed)

            assert info["actionable_training_start"]
            assert env.observation_space.contains(
                observation
            )

            attempts.append(
                info["sampling_attempts"]
            )

            risk = env.unwrapped.risk

            if risk < 0.012:
                counts["VERY_LOW"] += 1

            elif risk < 0.15:
                counts["MODERATE"] += 1

            else:
                counts["HIGH"] += 1

        print("Successful resets:", len(attempts))
        print(
            "Average sampling attempts:",
            round(sum(attempts) / len(attempts), 2)
        )

        print("Maximum attempts:", max(attempts))

        print("\nInitial risk bands:")

        for band in [
            "VERY_LOW",
            "MODERATE",
            "HIGH",
        ]:
            print(f"  {band}: {counts[band]}")

        print("\nAll sampled starting states:")
        print("  Technically eligible: YES")
        print("  Simulated approval: YES")
        print("  Safety validator bypassed: NO")

        print("\nSTEP 152 COMPLETE")

    finally:
        env.close()

