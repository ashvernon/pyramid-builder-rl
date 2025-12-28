from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# ─────────────────────────────────────────────────────────────
# Goal kinds
# ─────────────────────────────────────────────────────────────
GOAL_KIND_MULTI = 4


@dataclass
class Goal:
    kind: int
    # target is a fixed-length vector encoded as a tuple:
    #   [shape (n_layers), ramp_required (1)]
    target: tuple


# ─────────────────────────────────────────────────────────────
# Public API expected by env.py
# ─────────────────────────────────────────────────────────────

def goal_dim(n_layers: int) -> int:
    """
    Goal vector includes:
    - pyramid shape only (n_layers)
    - ramp requirement (1) in *layer units*
    """
    return n_layers + 1


def goal_to_vec(goal: Goal, n_layers: int) -> np.ndarray:
    """
    Convert Goal → vector (fixed size).
    """
    v = np.asarray(goal.target, dtype=np.float32)
    if v.shape[0] != n_layers + 1:
        raise ValueError(f"Goal target has dim {v.shape[0]} but expected {n_layers + 1}")
    return v


def achieved_goal_vec(state) -> np.ndarray:
    """
    Achieved goal includes:
    - current pyramid shape (state.p)
    - current ramp height (layer units)
    """
    shape = np.asarray(state.p, dtype=np.float32)
    ramp = np.asarray([float(state.ramp_height)], dtype=np.float32)
    return np.concatenate([shape, ramp], axis=0)


# ─────────────────────────────────────────────────────────────
# Goal sampling (HARD, tiered)
# ─────────────────────────────────────────────────────────────

def sample_goal(state, rng: np.random.Generator) -> Goal:
    """
    Tiered capability goal:
    - Complete layers 0..tier-1
    - Partial progress into tier
    - Requires ramp to be high enough for that tier (learnable now)

    Notes:
    - ramp_required follows the same scaling you previously enforced in is_success:
        ramp_height >= 0.6 * achieved_layer
      Here we approximate achieved_layer by the sampled tier.
    """
    n = state.n_layers

    progressed = [i for i, p in enumerate(state.p) if p > 0.2]
    cur_max = max(progressed) if progressed else 0

    tier = int(
        rng.integers(
            low=min(cur_max + 2, n // 2),
            high=n
        )
    )

    target_shape = np.zeros(n, dtype=np.float32)
    for i in range(tier):
        target_shape[i] = rng.uniform(0.9, 1.0)

    if tier < n:
        target_shape[tier] = rng.uniform(0.2, 0.6)

    # NEW: ramp requirement (layer units)
    ramp_required = 0.6 * float(tier)

    target = np.concatenate([target_shape, [ramp_required]]).astype(np.float32)

    return Goal(
        kind=GOAL_KIND_MULTI,
        target=tuple(target.tolist())
    )


# ─────────────────────────────────────────────────────────────
# Success condition (capability-based)
# ─────────────────────────────────────────────────────────────

def is_success(state, goal: Goal, tol_shape: float = 0.05, tol_ramp: float = 0.05) -> bool:
    """
    Success requires:
    - Shape target reached (within tol_shape)
    - Ramp height meets the goal's ramp requirement (within tol_ramp)

    This replaces the previous hidden constraint:
        ramp_height >= 0.6 * achieved_layer
    by making the ramp requirement part of the goal vector, so HER can learn it.
    """
    cur_shape = np.asarray(state.p, dtype=np.float32)
    tgt = np.asarray(goal.target, dtype=np.float32)

    if tgt.shape[0] != state.n_layers + 1:
        # Defensive: if someone passes an old-style goal, fail loudly.
        raise ValueError(
            f"Goal target has dim {tgt.shape[0]} but expected {state.n_layers + 1}. "
            "Did you rebuild goals after changing goal_dim?"
        )

    target_shape = tgt[:-1]
    target_ramp = float(tgt[-1])

    shape_ok = bool(np.all(cur_shape >= target_shape - tol_shape))
    ramp_ok = bool(float(state.ramp_height) >= target_ramp - tol_ramp)

    return bool(shape_ok and ramp_ok)
