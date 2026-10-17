
"""
Component 4 - Safe RL Policy Baseline Evaluation

Purpose:
Compare three fixed mitigation strategies before
training DQN, PPO and A2C.

Environment: Synthetic HDD simulation
Evaluation: 200 matched-seed episodes per policy

IMPORTANT:
- No real storage operations.
- No reinforcement learning training yet.
- Simulator dynamics are provisional assumptions.
"""

from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from statistics import mean, pstdev
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from contracts import MitigationAction
from decision_engine import HDDDecisionEngine
from hdd_rl_env_v2 import SafeHDDEnvV2 as SafeHDDEnv


# ---------------------------------------------------------
# 1. Experiment configuration
# ---------------------------------------------------------

EPISODES = 200
EPISODE_DAYS = 30
SEED_START = 1000

POLICIES = [
    "NO_OP",
    "RULE_BASED",
    "MIGRATION_FIRST"
]

engine = HDDDecisionEngine()


# ---------------------------------------------------------
# 2. Define baseline policies
# ---------------------------------------------------------

def choose_action(policy, env):

    if policy == "NO_OP":
        return MitigationAction.NO_OP

    if policy == "RULE_BASED":
        state = env._get_drive_state()
        result = engine.decide(state)

        # Propose the baseline action.
        # The environment independently applies its shield.
        return result.proposed_action

    if policy == "MIGRATION_FIRST":

        if env.migration_progress >= 1.0:
            return MitigationAction.ISOLATE_DRIVE

        return MitigationAction.MIGRATE_DATA

    raise ValueError("Unknown policy")


# ---------------------------------------------------------
# 3. Evaluate one policy across matched seeds
# ---------------------------------------------------------

def evaluate_policy(policy):

    returns = []
    episode_lengths = []

    failures = 0
    safety_blocks = 0
    total_steps = 0

    applied_actions = Counter()

    for episode in range(EPISODES):

        seed = SEED_START + episode

        env = SafeHDDEnv(
            episode_days=EPISODE_DAYS
        )

        try:
            obs, info = env.reset(seed=seed)

            episode_reward = 0.0
            done = False
            steps = 0

            while not done:

                selected = choose_action(
                    policy, env
                )

                action_id = env.ACTIONS.index(
                    selected
                )

                obs, reward, terminated, truncated, info = (
                    env.step(action_id)
                )

                episode_reward += reward
                steps += 1
                total_steps += 1

                applied_actions[
                    info["applied_action"]
                ] += 1

                if info["blocked_by_safety"]:
                    safety_blocks += 1

                if info["failed"]:
                    failures += 1

                done = terminated or truncated

            returns.append(episode_reward)
            episode_lengths.append(steps)

        finally:
            env.close()

    return {
        "policy": policy,
        "episodes": EPISODES,
        "mean_episode_reward": round(
            mean(returns), 4
        ),
        "std_episode_reward": round(
            pstdev(returns), 4
        ),
        "mean_episode_length": round(
            mean(episode_lengths), 3
        ),
        "episodes_ending_in_failure": failures,
        "failure_episode_rate_pct": round(
            100 * failures / EPISODES, 3
        ),
        "total_decision_steps": total_steps,
        "safety_blocks": safety_blocks,
        "applied_actions": dict(applied_actions)
    }


# ---------------------------------------------------------
# 4. Execute baseline comparison
# ---------------------------------------------------------

print("=" * 65)
print("SAFE HDD RL - BASELINE POLICY EVALUATION")
print("=" * 65)

results = []

for policy in POLICIES:

    print(
        f"\nEvaluating {policy}...",
        flush=True
    )

    result = evaluate_policy(policy)

    results.append(result)

    print(
        "Mean reward:",
        result["mean_episode_reward"]
    )

    print(
        "Failure episodes:",
        result["episodes_ending_in_failure"]
    )

    print(
        "Safety blocks:",
        result["safety_blocks"]
    )


# ---------------------------------------------------------
# 5. Save reproducible research results
# ---------------------------------------------------------

REPORT = {
    "experiment": "C4_V2_BASELINE_POLICY_EVALUATION",
   "environment": "SafeHDDEnvV2",
    "simulation_only": True,
    "episodes_per_policy": EPISODES,
    "episode_days": EPISODE_DAYS,
    "seed_start": SEED_START,
    "matched_seeds": True,
    "rl_training_performed": False,
    "results": results,
    "limitations": [
        "Synthetic failure hazard is not calibrated.",
        "Intervention effects are provisional assumptions.",
        "Some mitigation actions remain unavailable.",
        "Failure probability is independent of mitigation.",
        "Results are not evidence of production performance."
    ],
    "created_utc": datetime.now(
        timezone.utc
    ).isoformat()
}

OUTPUT = (
    ROOT / "results"
    / "103_v2_policy_baseline_evaluation.json"
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as file:
    json.dump(REPORT, file, indent=4)


# ---------------------------------------------------------
# 6. Display comparison
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("BASELINE POLICY COMPARISON")
print("=" * 75)

for result in results:

    print("\nPolicy:", result["policy"])

    print(
        "Mean episode reward:",
        result["mean_episode_reward"]
    )

    print(
        "Reward std:",
        result["std_episode_reward"]
    )

    print(
        "Failure episodes:",
        result["episodes_ending_in_failure"]
    )

    print(
        "Safety blocks:",
        result["safety_blocks"]
    )

    print(
        "Applied actions:",
        result["applied_actions"]
    )

print("\nSTATUS: BASELINE EVALUATION COMPLETED")
print("Report:", OUTPUT)
