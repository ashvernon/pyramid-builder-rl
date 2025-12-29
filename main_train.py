from __future__ import annotations

import argparse
from pathlib import Path
import yaml

from sim.env import PyramidEnv
from rl.replay import ReplayBuffer
from rl.learner import GoalDQNAgent
from rl.eval import run_eval
from utils.rng import set_global_seed
from utils.logging import JsonlLogger, ensure_dir


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict, patch: dict) -> dict:
    # Recursively merge dict patch into base (in-place).
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="Path to experiments/*.yaml")
    ap.add_argument("--base", default="config.yaml", help="Base config")
    ap.add_argument("--out", default="logs", help="Log output dir")
    args = ap.parse_args()

    base = load_yaml(args.base)
    exp = load_yaml(args.exp)
    cfg = deep_update(base, exp)

    exp_name = exp.get("name", Path(args.exp).stem)

    out_dir = Path(args.out) / exp_name
    ensure_dir(out_dir)

    # ✅ Make checkpoint dir absolute (anchored to this file), so saves always land in your repo
    repo_root = Path(__file__).resolve().parent
    checkpoint_dir = (repo_root / "checkpoints" / exp_name).resolve()
    ensure_dir(checkpoint_dir)
    print(f"[CKPT] checkpoint_dir = {checkpoint_dir}")

    def save_ckpt(tag: str) -> None:
        p = checkpoint_dir / f"{tag}.pt"
        agent.save(p)
        print(f"[CKPT] saved {tag} -> {p}")

    sim_cfg = cfg["sim"]
    rl_cfg = cfg["rl"]

    set_global_seed(int(rl_cfg["seed"]))

    env = PyramidEnv(sim_cfg, render=False, seed=int(rl_cfg["seed"]))
    obs_dim, goal_dim = env.obs_dim, env.goal_dim
    n_actions = env.n_actions

    agent = GoalDQNAgent(
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        n_actions=n_actions,
        width=int(rl_cfg["width"]),
        depth=int(rl_cfg["depth"]),
        lr=float(rl_cfg["lr"]),
        gamma=float(rl_cfg["gamma"]),
        target_update_every=int(rl_cfg["target_update_every"]),
        target_tau=float(rl_cfg["target_tau"]),
    )

    replay = ReplayBuffer(
        capacity=int(rl_cfg["replay_size"]),
        obs_dim=obs_dim,
        goal_dim=goal_dim,
        her_k=int(rl_cfg["her_k"]),
    )

    logger = JsonlLogger(out_dir / "train.jsonl")

    total_steps = int(rl_cfg["total_steps"])
    warmup_steps = int(rl_cfg["warmup_steps"])
    batch_size = int(rl_cfg["batch_size"])

    eps_start = float(rl_cfg["eps_start"])
    eps_end = float(rl_cfg["eps_end"])
    eps_decay = int(rl_cfg["eps_decay_steps"])

    def epsilon(step: int) -> float:
        if step >= eps_decay:
            return eps_end
        frac = step / max(1, eps_decay)
        return eps_start + frac * (eps_end - eps_start)

    step = 0
    episode = 0

    # ✅ Track best model by eval_success_rate
    best_success = -1.0

    while step < total_steps:
        obs, goal = env.reset()
        done = False
        ep_steps = 0
        ep_success = 0

        # episode-level ramp tracking (final can be 0 if dismantled late)
        ep_max_ramp_height = 0.0
        ep_mean_ramp_height_sum = 0.0

        # episode-level ramp command stats
        ep_ramp_cmd_extend = 0
        ep_ramp_cmd_dismantle = 0
        ep_ramp_cmd_switch = 0
        ep_ramp_cmd_none = 0

        while not done and step < total_steps:
            eps = epsilon(step)
            if step < warmup_steps:
                action = env.sample_action()
            else:
                action = agent.act(obs, goal, eps=eps)

            next_obs, next_goal, done, info = env.step(action)
            achieved_next = info["achieved_goal"]

            replay.add(
                obs=obs,
                goal=goal,
                action=action,
                next_obs=next_obs,
                done=done,
                achieved_next=achieved_next,
            )

            # ramp tracking + ramp-cmd decode from action_id
            ramp_h = float(info.get("ramp_height", 0.0))
            ep_max_ramp_height = max(ep_max_ramp_height, ramp_h)
            ep_mean_ramp_height_sum += ramp_h

            # Decode ramp_cmd from action_id based on env design (ramp_cmds length = 4)
            # 0 none, 1 extend, 2 switch, 3 dismantle
            ramp_cmd = int(info.get("action_id", action)) % 4
            if ramp_cmd == 0:
                ep_ramp_cmd_none += 1
            elif ramp_cmd == 1:
                ep_ramp_cmd_extend += 1
            elif ramp_cmd == 2:
                ep_ramp_cmd_switch += 1
            elif ramp_cmd == 3:
                ep_ramp_cmd_dismantle += 1

            obs, goal = next_obs, next_goal
            ep_steps += 1
            step += 1
            ep_success = max(ep_success, int(info.get("success", 0)))

            if step >= warmup_steps and len(replay) >= batch_size:
                batch = replay.sample(batch_size=batch_size)
                losses = agent.update(batch)
                if step % 100 == 0:
                    logger.log({
                        "type": "train_step",
                        "step": step,
                        "episode": episode,
                        "epsilon": eps,
                        **losses,
                    })

            if step % int(rl_cfg["eval_every"]) == 0 and step >= warmup_steps:
                metrics = run_eval(env, agent, n_episodes=30, fixed_goal_set=True)
                logger.log({
                    "type": "eval",
                    "step": step,
                    **metrics,
                })

                # ✅ always save latest
                save_ckpt("latest")

                # ✅ save best by eval_success_rate
                sr = float(metrics.get("eval_success_rate", -1.0))
                if sr > best_success:
                    best_success = sr
                    save_ckpt("best")

        # log goal ramp requirement (last dim of goal), plus episode ramp stats
        goal_ramp_required = (
            float(goal[-1])
            if getattr(goal, "shape", None) is not None and goal.shape[0] >= 1
            else None
        )
        ep_mean_ramp_height = (ep_mean_ramp_height_sum / max(1, ep_steps)) if ep_steps > 0 else 0.0

        logger.log({
            "type": "episode",
            "episode": episode,
            "step": step,
            "ep_steps": ep_steps,
            "success": ep_success,
            "max_layer": info.get("max_layer", None),
            "final_ramp_height": float(info.get("ramp_height", 0.0)),
            "goal_ramp_required": goal_ramp_required,
            "max_ramp_height": float(ep_max_ramp_height),
            "mean_ramp_height": float(ep_mean_ramp_height),
            "ramp_cmd_counts": {
                "none": int(ep_ramp_cmd_none),
                "extend": int(ep_ramp_cmd_extend),
                "switch": int(ep_ramp_cmd_switch),
                "dismantle": int(ep_ramp_cmd_dismantle),
            },
        })
        episode += 1

    # ✅ always save final at end of training
    save_ckpt("final")


if __name__ == "__main__":
    main()
