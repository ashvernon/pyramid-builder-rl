from __future__ import annotations

from typing import Tuple, Optional, Dict, Any
import numpy as np

from sim.state import PyramidState
from sim.dynamics import step_state
from sim.goals import Goal, sample_goal, goal_to_vec, achieved_goal_vec, goal_dim, is_success


class PyramidEnv:
    """Headless-first environment (vector obs, goal-conditioned)."""

    def __init__(self, sim_cfg: dict, render: bool = False, seed: int = 1):
        self.cfg = sim_cfg
        self.render_enabled = render
        self.rng = np.random.default_rng(seed)

        self.n_layers = int(sim_cfg["n_layers"])
        self.t_max = int(sim_cfg["t_max"])

        self._alloc_presets = self._make_alloc_presets(sim_cfg["workers_total"])
        self._ramp_cmds = [0, 1, 2, 3]  # none, extend, switch, dismantle
        self._switch_types = [1, 2, 3]  # straight, zigzag, spiral

        self.obs_dim = 54
        self.goal_dim = goal_dim(self.n_layers)
        self.n_actions = len(self._alloc_presets) * len(self._ramp_cmds)

        self.state: Optional[PyramidState] = None
        self.goal: Optional[Goal] = None

        if self.render_enabled:
            from sim.renderer import Renderer
            self.renderer = Renderer(self.n_layers)
        else:
            self.renderer = None

    def _make_alloc_presets(self, W: int):
        presets = [
            (int(0.50*W), int(0.30*W), W - int(0.50*W) - int(0.30*W)),
            (int(0.30*W), int(0.50*W), W - int(0.30*W) - int(0.50*W)),
            (int(0.20*W), int(0.30*W), W - int(0.20*W) - int(0.30*W)),
            (int(0.40*W), int(0.40*W), W - int(0.40*W) - int(0.40*W)),
            (int(0.20*W), int(0.40*W), W - int(0.20*W) - int(0.40*W)),
            (int(0.33*W), int(0.33*W), W - int(0.33*W) - int(0.33*W)),
        ]
        return presets

    def sample_action(self) -> int:
        return int(self.rng.integers(0, self.n_actions))

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        p0 = [0.0 for _ in range(self.n_layers)]
        self.state = PyramidState(
            t=0,
            n_layers=self.n_layers,
            p=p0,
            current_layer=0,
            ramp_type=0,
            ramp_height=0.0,
            ramp_length=1.0,
            quarry_stock=int(self.rng.integers(200, 800)),
            in_transit=0,
            site_stock=int(self.rng.integers(0, 200)),
            workers_total=int(self.cfg["workers_total"]),
            workers_quarry=int(0.4 * self.cfg["workers_total"]),
            workers_haul=int(0.4 * self.cfg["workers_total"]),
            workers_place=int(0.2 * self.cfg["workers_total"]),
            friction=float(self.cfg["friction"]),
            heat=float(self.cfg.get("heat", 0.2)),
            fatigue=0.0,
            done=False,
        )
        self.goal = sample_goal(self.state, self.rng)
        obs = self._obs_vec(self.state)
        g = goal_to_vec(self.goal, self.n_layers)
        return obs, g

    def step(self, action_id: int) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        assert self.state is not None and self.goal is not None

        alloc_i = action_id // len(self._ramp_cmds)
        ramp_i = action_id % len(self._ramp_cmds)

        wq, wh, wp = self._alloc_presets[alloc_i]
        ramp_cmd = self._ramp_cmds[ramp_i]
        ramp_type = int(self.state.ramp_type)

        if ramp_cmd == 2:
            ramp_type = int(self.rng.choice(self._switch_types))

        a = {"wq": wq, "wh": wh, "wp": wp, "ramp_cmd": ramp_cmd, "ramp_type": ramp_type}
        next_state = step_state(self.state, a, {**self.cfg, "t_max": self.t_max})

        success = is_success(next_state, self.goal)
        done = bool(next_state.done)

        obs = self._obs_vec(next_state)
        g = goal_to_vec(self.goal, self.n_layers)

        info = {
            "success": int(success),
            "achieved_goal": achieved_goal_vec(next_state),
            "max_layer": int(next_state.current_layer),
            "ramp_height": float(next_state.ramp_height),
            "site_stock": int(next_state.site_stock),
            "quarry_stock": int(next_state.quarry_stock),
            "action_id": int(action_id),
        }

        self.state = next_state
        return obs, g, done, info

    def render(self, info: Optional[dict] = None):
        if not self.render_enabled or self.renderer is None or self.state is None or self.goal is None:
            return
        self.renderer.render(self.state, self.goal, info=info)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def _obs_vec(self, s: PyramidState) -> np.ndarray:
        n = s.n_layers
        t_norm = s.t / max(1.0, float(self.t_max))
        heat_norm = float(s.heat)

        layer_idx_norm = s.current_layer / max(1.0, float(n - 1))
        p_cur = s.p[s.current_layer]
        p_prev = s.p[s.current_layer - 1] if s.current_layer > 0 else 0.0
        p_next = s.p[s.current_layer + 1] if s.current_layer < n - 1 else 0.0

        p_vec = np.asarray(s.p, dtype=np.float32)

        ramp_onehot = np.zeros(4, dtype=np.float32)
        ramp_onehot[s.ramp_type] = 1.0
        ramp_height_norm = s.ramp_height / max(1.0, float(n))
        ramp_eff = float(max(0.2, 1.0 - 0.2 * ramp_height_norm))

        q_max = float(self.cfg["q_max"])
        tr_max = float(self.cfg["transit_max"])
        s_max = float(self.cfg["site_max"])

        quarry_stock_norm = s.quarry_stock / max(1.0, q_max)
        in_transit_norm = s.in_transit / max(1.0, tr_max)
        site_stock_norm = s.site_stock / max(1.0, s_max)

        W = float(s.workers_total)
        wq = s.workers_quarry / max(1.0, W)
        wh = s.workers_haul / max(1.0, W)
        wp = s.workers_place / max(1.0, W)

        friction_norm = float(s.friction)
        fatigue_norm = float(s.fatigue)
        damage_norm = 0.0
        idle_penalty_norm = 0.0

        vec = np.concatenate([
            np.asarray([t_norm, heat_norm], dtype=np.float32),
            np.asarray([layer_idx_norm, p_cur, p_prev, p_next], dtype=np.float32),
            p_vec.astype(np.float32),
            ramp_onehot,
            np.asarray([ramp_height_norm, ramp_eff], dtype=np.float32),
            np.asarray([quarry_stock_norm, in_transit_norm, site_stock_norm, wq, wh, wp], dtype=np.float32),
            np.asarray([friction_norm, fatigue_norm, damage_norm, idle_penalty_norm], dtype=np.float32),
        ], axis=0)

        assert vec.shape[0] == self.obs_dim
        return vec
