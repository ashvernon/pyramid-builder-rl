from __future__ import annotations

from typing import Dict
import numpy as np


def run_eval(env, agent, n_episodes: int = 50, fixed_goal_set: bool = True) -> Dict[str, float]:
    successes = 0
    max_layers = []
    stepss = []

    base_seed = 12345 if fixed_goal_set else None

    for i in range(int(n_episodes)):
        if base_seed is not None:
            obs, goal = env.reset(seed=base_seed + i)
        else:
            obs, goal = env.reset()

        done = False
        steps = 0
        max_layer = 0

        while not done and steps < env.t_max:
            a = agent.act(obs, goal, eps=0.0)
            obs, goal, done, info = env.step(a)
            steps += 1
            max_layer = max(max_layer, int(info.get("max_layer", 0)))

        successes += int(info.get("success", 0))
        max_layers.append(max_layer)
        stepss.append(steps)

    return {
        "eval_success_rate": successes / max(1, int(n_episodes)),
        "eval_max_layer_mean": float(np.mean(max_layers)) if max_layers else 0.0,
        "eval_steps_mean": float(np.mean(stepss)) if stepss else 0.0,
    }
