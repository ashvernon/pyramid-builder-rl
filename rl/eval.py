from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple
import numpy as np

from sim.goals import Goal, capability_probe_goal


def _run_episode(
    env,
    agent,
    *,
    goal: Optional[Goal],
    seed: Optional[int],
    random_policy_ceiling: Optional[int],
) -> Tuple[int, int, int, Optional[int]]:
    """Run a single evaluation episode for a provided goal (or env default)."""

    obs, g = env.reset(seed=seed, goal=goal)
    done = False
    steps = 0
    max_layer = 0
    first_exceed = None

    while not done and steps < env.t_max:
        a = agent.act(obs, g, eps=0.0)
        obs, g, done, info = env.step(a)
        steps += 1
        max_layer = max(max_layer, int(info.get("max_layer", 0)))

        if (
            first_exceed is None
            and random_policy_ceiling is not None
            and max_layer > random_policy_ceiling
        ):
            first_exceed = steps

    success = int(info.get("success", 0))
    return success, max_layer, steps, first_exceed


def _run_capability_probes(
    env,
    agent,
    probe_tiers: Sequence[int],
    episodes_per_probe: int,
    random_policy_ceiling: Optional[int],
) -> Dict[str, object]:
    """Evaluate a deterministic set of capability probes (never used in training)."""

    pass_rates = {}
    tier_rate_pairs = []
    probe_seed_base = 98765

    for idx, tier in enumerate(probe_tiers):
        probe_goal = capability_probe_goal(env.n_layers, tier)
        successes = 0

        for ep in range(int(episodes_per_probe)):
            seed = probe_seed_base + idx * int(episodes_per_probe) + ep
            s, _, _, _ = _run_episode(
                env,
                agent,
                goal=probe_goal,
                seed=seed,
                random_policy_ceiling=random_policy_ceiling,
            )
            successes += int(s)

        rate = successes / max(1, int(episodes_per_probe))
        pass_rates[str(int(tier))] = float(rate)
        tier_rate_pairs.append((int(tier), float(rate)))

    tier_rate_pairs.sort(key=lambda x: x[0])
    max_tier = max((t for t, r in tier_rate_pairs if r >= 0.5), default=0)

    if len(tier_rate_pairs) >= 2:
        tiers = np.asarray([t for t, _ in tier_rate_pairs], dtype=np.float32)
        rates = np.asarray([r for _, r in tier_rate_pairs], dtype=np.float32)
        # numpy 2.x removed ``np.trapz`` in favor of ``np.trapezoid``; keep a
        # small compatibility shim so either is used depending on availability.
        trapz_fn = getattr(np, "trapz", None)
        if trapz_fn is None:
            trapz_fn = np.trapezoid

        auc = float(trapz_fn(rates, tiers) / max(1e-9, tiers[-1] - tiers[0]))
    elif tier_rate_pairs:
        auc = float(tier_rate_pairs[0][1])
    else:
        auc = 0.0

    return {
        "cap_probe_pass_rates": pass_rates,
        "cap_probe_max_tier": float(max_tier),
        "cap_probe_auc": float(auc),
    }


def run_eval(
    env,
    agent,
    n_episodes: int = 50,
    fixed_goal_set: bool = True,
    capability_probe_tiers: Optional[Sequence[int]] = None,
    capability_probe_episodes: int = 3,
    random_policy_ceiling: Optional[int] = None,
) -> Dict[str, object]:
    """Run evaluation with standard metrics plus capability probes."""

    successes = 0
    max_layers = []
    stepss = []

    first_exceed_overall: Optional[int] = None
    cumulative_steps = 0
    base_seed = 12345 if fixed_goal_set else None

    for i in range(int(n_episodes)):
        seed = base_seed + i if base_seed is not None else None
        success, max_layer, steps, first_exceed = _run_episode(
            env,
            agent,
            goal=None,
            seed=seed,
            random_policy_ceiling=random_policy_ceiling,
        )

        successes += int(success)
        max_layers.append(max_layer)
        stepss.append(steps)

        if first_exceed is not None and first_exceed_overall is None:
            first_exceed_overall = cumulative_steps + int(first_exceed)

        cumulative_steps += steps

    metrics: Dict[str, object] = {
        "eval_success_rate": successes / max(1, int(n_episodes)),
        "eval_max_layer_mean": float(np.mean(max_layers)) if max_layers else 0.0,
        "eval_steps_mean": float(np.mean(stepss)) if stepss else 0.0,
        "first_step_exceeding_random_ceiling": int(first_exceed_overall)
        if first_exceed_overall is not None
        else -1,
    }

    probe_tiers = list(capability_probe_tiers or [])
    if probe_tiers:
        metrics.update(
            _run_capability_probes(
                env,
                agent,
                probe_tiers=probe_tiers,
                episodes_per_probe=int(capability_probe_episodes),
                random_policy_ceiling=random_policy_ceiling,
            )
        )

    return metrics
