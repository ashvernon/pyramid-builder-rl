from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Batch:
    obs: np.ndarray
    goal: np.ndarray
    action: np.ndarray
    next_obs: np.ndarray
    done: np.ndarray
    reward: np.ndarray


class ReplayBuffer:
    """
    Replay buffer with HER and distance-based rewards.

    Assumes goal/achieved vectors are:
      [shape (n_layers), ramp_required_or_height (1)]
    where the last dimension encodes ramp in "layer units".

    Reward is negative distance-to-goal with a small success bonus.
    Shape distance is L1 mean over layers.
    Ramp distance only penalizes if achieved_ramp < goal_ramp.
    """

    def __init__(self, capacity: int, obs_dim: int, goal_dim: int, her_k: int = 4):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.her_k = her_k

        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._goal = np.zeros((capacity, goal_dim), dtype=np.float32)
        self._action = np.zeros((capacity,), dtype=np.int64)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._done = np.zeros((capacity,), dtype=np.float32)
        self._achieved_next = np.zeros((capacity, goal_dim), dtype=np.float32)

        self._idx = 0
        self._size = 0
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return self._size

    def add(self, obs, goal, action, next_obs, done, achieved_next):
        i = self._idx
        self._obs[i] = obs
        self._goal[i] = goal
        self._action[i] = int(action)
        self._next_obs[i] = next_obs
        self._done[i] = float(done)
        self._achieved_next[i] = achieved_next

        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = self.rng.integers(0, self._size, size=batch_size)

        obs = self._obs[idx].copy()
        goal = self._goal[idx].copy()
        action = self._action[idx].copy()
        next_obs = self._next_obs[idx].copy()
        done = self._done[idx].copy()

        # HER relabel: 50%
        relabel = self.rng.random(batch_size) < 0.5
        goal[relabel] = self._achieved_next[idx][relabel]

        achieved = self._achieved_next[idx]

        # ---- distance-based reward with ramp-aware last dimension ----
        if self.goal_dim < 2:
            # Fallback: no room for ramp dimension; keep old behavior
            dist = np.mean(np.abs(achieved - goal), axis=1)
        else:
            achieved_shape = achieved[:, :-1]
            goal_shape = goal[:, :-1]

            achieved_ramp = achieved[:, -1]
            goal_ramp = goal[:, -1]

            # Shape distance: mean absolute difference across layers
            shape_dist = np.mean(np.abs(achieved_shape - goal_shape), axis=1)

            # Ramp distance: only penalize being BELOW required ramp
            ramp_shortfall = np.maximum(0.0, goal_ramp - achieved_ramp)

            # Combine. Ramp is "hard-ish" but we don't want it to dominate too early.
            # Tune weight if needed; 0.5 is a reasonable start.
            dist = shape_dist + 0.5 * ramp_shortfall

        reward = -dist
        reward += (dist < 0.05).astype(np.float32)

        return Batch(
            obs=obs,
            goal=goal,
            action=action,
            next_obs=next_obs,
            done=done,
            reward=reward.astype(np.float32),
        )
