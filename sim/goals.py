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
    #   [foundation (1), ramp_height (1), ramp_integrity (1), locks (n_layers), capstone (1)]
    target: tuple


# ─────────────────────────────────────────────────────────────
# Public API expected by env.py
# ─────────────────────────────────────────────────────────────

def goal_dim(n_layers: int) -> int:
    """
    Goal vector includes:
    - foundation strength target (1)
    - ramp height target (1)
    - ramp integrity target (1)
    - layer lock targets (n_layers)
    - capstone placement flag (1)
    """
    return n_layers + 4


def goal_to_vec(goal: Goal, n_layers: int) -> np.ndarray:
    """
    Convert Goal → vector (fixed size).
    """
    v = np.asarray(goal.target, dtype=np.float32)
    if v.shape[0] != n_layers + 4:
        raise ValueError(f"Goal target has dim {v.shape[0]} but expected {n_layers + 4}")
    return v


def vector_to_goal(target: np.ndarray, n_layers: int) -> Goal:
    """Convert a goal vector back to a Goal dataclass.

    This is useful for replay-sampled achieved states that should become
    explicit goals (e.g., for diversity sampling).
    """

    arr = np.asarray(target, dtype=np.float32)
    if arr.shape[0] != n_layers + 4:
        raise ValueError(f"Goal vector has dim {arr.shape[0]} but expected {n_layers + 4}")
    return Goal(kind=GOAL_KIND_MULTI, target=tuple(arr.tolist()))


def achieved_goal_vec(state) -> np.ndarray:
    """
    Achieved goal includes:
    - locked foundation strength (or current stability)
    - current ramp height (layer units)
    - current ramp integrity
    - stabilized/locked progress per layer
    - capstone placement flag
    """
    foundation = np.asarray([float(state.foundation_strength)], dtype=np.float32)
    ramp = np.asarray([float(state.ramp_height), float(state.ramp_integrity)], dtype=np.float32)
    locks = np.asarray(
        [1.0 if locked else float(state.layer_stability[i]) for i, locked in enumerate(state.layer_locked)],
        dtype=np.float32,
    )
    capstone = np.asarray([1.0 if state.capstone_placed else 0.0], dtype=np.float32)
    return np.concatenate([foundation, ramp, locks, capstone], axis=0)


# ─────────────────────────────────────────────────────────────
# Goal sampling (HARD, tiered)
# ─────────────────────────────────────────────────────────────

def sample_goal(state, rng: np.random.Generator) -> Goal:
    """
    Tiered hierarchical goal:
    - Lock foundation with sufficient strength
    - Build and maintain a ramp with integrity
    - Stabilize layers sequentially (locks)
    - Optionally require capstone placement for deepest tiers
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

    target_layers = np.zeros(n, dtype=np.float32)
    for i in range(tier):
        target_layers[i] = rng.uniform(0.9, 1.0)

    if tier < n:
        target_layers[tier] = rng.uniform(0.3, 0.7)

    foundation_target = rng.uniform(0.88, 0.95)
    ramp_height_target = 0.6 * float(tier)
    ramp_integrity_target = rng.uniform(0.65, 0.9)

    capstone_target = 1.0 if tier >= n - 1 else 0.0
    target = np.concatenate(
        [
            [foundation_target, ramp_height_target, ramp_integrity_target],
            target_layers,
            [capstone_target],
        ]
    ).astype(np.float32)

    return Goal(
        kind=GOAL_KIND_MULTI,
        target=tuple(target.tolist())
    )


def capability_probe_goal(n_layers: int, tier: int, ramp_scale: float = 0.6) -> Goal:
    """Deterministic, capability-tiered goal used only for evaluation probes."""

    tier = int(tier)
    target_shape = np.zeros(n_layers, dtype=np.float32)
    up_to = min(tier, n_layers)
    target_shape[:up_to] = 1.0
    if tier < n_layers:
        target_shape[tier] = 0.7  # partial stabilize to create a soft boundary

    ramp_required = ramp_scale * float(tier)
    foundation_target = 0.92
    ramp_integrity_target = 0.8
    capstone_target = 1.0 if tier >= n_layers - 1 else 0.0

    target = np.concatenate(
        [
            [foundation_target, ramp_required, ramp_integrity_target],
            target_shape,
            [capstone_target],
        ]
    ).astype(np.float32)

    return Goal(kind=GOAL_KIND_MULTI, target=tuple(target.tolist()))


# ─────────────────────────────────────────────────────────────
# Success condition (capability-based)
# ─────────────────────────────────────────────────────────────

def is_success(state, goal: Goal, tol_shape: float = 0.05, tol_ramp: float = 0.05) -> bool:
    """
    Success requires:
    - Locked foundation strength meets the goal
    - Ramp height and integrity meet the goal
    - Layer locks meet the goal
    - Capstone placement if required
    """
    if getattr(state, "irreversible_error", False):
        return False

    tgt = np.asarray(goal.target, dtype=np.float32)

    if tgt.shape[0] != state.n_layers + 4:
        # Defensive: if someone passes an old-style goal, fail loudly.
        raise ValueError(
            f"Goal target has dim {tgt.shape[0]} but expected {state.n_layers + 4}. "
            "Did you rebuild goals after changing goal_dim?"
        )

    foundation_target = float(tgt[0])
    ramp_height_target = float(tgt[1])
    ramp_integrity_target = float(tgt[2])
    target_shape = tgt[3:-1]
    capstone_target = float(tgt[-1])

    locks = np.asarray(
        [1.0 if locked else float(state.layer_stability[i]) for i, locked in enumerate(state.layer_locked)],
        dtype=np.float32,
    )
    foundation_ok = bool(float(state.foundation_strength) >= foundation_target - tol_shape)
    ramp_height_ok = bool(float(state.ramp_height) >= ramp_height_target - tol_ramp)
    ramp_integrity_ok = bool(float(state.ramp_integrity) >= ramp_integrity_target - tol_ramp)
    shape_ok = bool(np.all(locks >= target_shape - tol_shape))
    capstone_ok = bool(capstone_target <= 0.0 or state.capstone_placed)

    return bool(foundation_ok and ramp_height_ok and ramp_integrity_ok and shape_ok and capstone_ok)
