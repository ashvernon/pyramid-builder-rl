from dataclasses import replace
from unittest.mock import patch
import unittest

from sim.env import PyramidEnv
from sim.goals import GOAL_KIND_MULTI, Goal


class IrreversibleSuccessInvariantTest(unittest.TestCase):
    def test_irreversible_error_blocks_success(self) -> None:
        sim_cfg = {
            "n_layers": 3,
            "t_max": 5,
            "workers_total": 10,
            "friction": 0.3,
            "heat": 0.2,
            "q_max": 100,
            "transit_max": 100,
            "site_max": 100,
        }

        goal_target = (0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0)
        goal = Goal(kind=GOAL_KIND_MULTI, target=goal_target)

        env = PyramidEnv(sim_cfg, render=False, seed=0)
        env.reset(goal=goal)

        base_state = env.state
        assert base_state is not None

        successful_state = replace(
            base_state,
            foundation_strength=0.6,
            ramp_height=0.6,
            ramp_integrity=0.6,
            layer_stability=[1.0 for _ in range(base_state.n_layers)],
            layer_locked=[True for _ in range(base_state.n_layers)],
            capstone_placed=True,
            done=True,
        )

        setattr(successful_state, "irreversible_error", True)
        setattr(successful_state, "phase_reached", 3)
        setattr(successful_state, "phase_failure_reason", "test")

        with patch("sim.env.step_state", return_value=successful_state):
            _, _, done, info = env.step(0)

        self.assertTrue(done)
        self.assertEqual(info["success"], 0)
        self.assertTrue(info["irreversible_error"])
        self.assertEqual(info["phase_reached"], 3)
        self.assertEqual(info.get("phase_failure_reason"), "test")


if __name__ == "__main__":
    unittest.main()
