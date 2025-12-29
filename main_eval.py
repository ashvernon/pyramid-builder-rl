from __future__ import annotations

import argparse
import yaml
from pathlib import Path

from sim.env import PyramidEnv
from rl.learner import GoalDQNAgent
from rl.eval import run_eval
from utils.rng import set_global_seed


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict, patch: dict) -> dict:
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--base", default="config.yaml")
    ap.add_argument("--ckpt", default="", help="Path to checkpoint (optional)")
    args = ap.parse_args()

    cfg = deep_update(load_yaml(args.base), load_yaml(args.exp))
    sim_cfg = cfg["sim"]
    rl_cfg = cfg["rl"]

    capability_probe_tiers = rl_cfg.get("capability_probe_tiers", [12, 16, 20])
    capability_probe_episodes = int(rl_cfg.get("capability_probe_episodes", 3))
    random_policy_ceiling = int(rl_cfg.get("random_policy_ceiling", 11))

    set_global_seed(int(rl_cfg["seed"]))
    env = PyramidEnv(sim_cfg, render=False, seed=int(rl_cfg["seed"]))

    agent = GoalDQNAgent(
        obs_dim=env.obs_dim, goal_dim=env.goal_dim, n_actions=env.n_actions,
        width=int(rl_cfg["width"]), depth=int(rl_cfg["depth"]),
        lr=float(rl_cfg["lr"]), gamma=float(rl_cfg["gamma"]),
        target_update_every=int(rl_cfg["target_update_every"]),
        target_tau=float(rl_cfg["target_tau"]),
    )
    if args.ckpt:
        agent.load(Path(args.ckpt))

    metrics = run_eval(
        env,
        agent,
        n_episodes=100,
        fixed_goal_set=True,
        capability_probe_tiers=capability_probe_tiers,
        capability_probe_episodes=capability_probe_episodes,
        random_policy_ceiling=random_policy_ceiling,
    )
    print(metrics)


if __name__ == "__main__":
    main()
