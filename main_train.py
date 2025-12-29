from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import yaml
import numpy as np

from sim.state import PyramidState

from sim.env import PyramidEnv
from sim.goals import vector_to_goal
from rl.replay import ReplayBuffer
from rl.learner import GoalDQNAgent
from rl.eval import run_eval
from utils.rng import set_global_seed
from utils.logging import JsonlLogger, ensure_dir, flush, log_event


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


def episode_phase_outcome(
    state: Optional[PyramidState],
    goal_vec: np.ndarray,
    success: int,
    tol_shape: float = 0.05,
    tol_ramp: float = 0.05,
) -> tuple[int, Optional[str], bool]:
    """
    Decode hierarchical construction phases for episode logging.

    Phases are cumulative counts of completed milestones:
    1. Foundation stability meets goal target
    2. Ramp height + integrity meet goal target
    3. Target layers stabilized/locked
    4. Capstone placed (only if required by goal)
    """

    if state is None:
        return 0, None, False

    g = np.asarray(goal_vec, dtype=np.float32)
    n = int(state.n_layers)
    if g.shape[0] != n + 4:
        raise ValueError(f"Goal vector has dim {g.shape[0]} but expected {n + 4}")

    foundation_target = float(g[0])
    ramp_height_target = float(g[1])
    ramp_integrity_target = float(g[2])
    target_shape = g[3:-1]
    capstone_target = float(g[-1])

    locks = np.asarray(
        [1.0 if locked else float(state.layer_stability[i]) for i, locked in enumerate(state.layer_locked)],
        dtype=np.float32,
    )

    foundation_ready = float(state.foundation_strength) >= foundation_target - tol_shape
    ramp_ready = bool(
        float(state.ramp_height) >= ramp_height_target - tol_ramp
        and float(state.ramp_integrity) >= ramp_integrity_target - tol_ramp
    )
    shape_ready = bool(np.all(locks >= target_shape - tol_shape))
    capstone_required = capstone_target > 0.0

    phase_reached = 0
    if foundation_ready:
        phase_reached = 1
    if foundation_ready and ramp_ready:
        phase_reached = 2
    if foundation_ready and ramp_ready and shape_ready:
        phase_reached = 3
    if capstone_required and foundation_ready and ramp_ready and shape_ready and state.capstone_placed:
        phase_reached = 4

    failure_reason: Optional[str] = None
    if not success:
        if not foundation_ready:
            failure_reason = "foundation_unstable"
        elif not ramp_ready:
            failure_reason = "ramp_inadequate"
        elif not shape_ready:
            failure_reason = "shape_incomplete"
        elif capstone_required and not state.capstone_placed:
            failure_reason = "capstone_unreachable"

    irreversible_error = bool(
        (not foundation_ready and float(state.foundation_strength) < 0.5 * foundation_target)
        or (float(state.ramp_integrity) < 0.25)
    )

    return phase_reached, failure_reason, irreversible_error


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

    goal_replay_fraction = float(rl_cfg.get("goal_replay_fraction", 0.15))
    goal_diversity_rng = np.random.default_rng(int(rl_cfg.get("seed", 1)) + 7)

    eps_start = float(rl_cfg["eps_start"])
    eps_end = float(rl_cfg["eps_end"])
    eps_decay = int(rl_cfg["eps_decay_steps"])

    capability_probe_tiers = rl_cfg.get("capability_probe_tiers", [12, 16, 20])
    capability_probe_episodes = int(rl_cfg.get("capability_probe_episodes", 3))
    random_policy_ceiling = int(rl_cfg.get("random_policy_ceiling", 11))
    ramp_commit_height = float(rl_cfg.get("ramp_commit_height", 8.0))
    ramp_commit_early_frac = float(rl_cfg.get("ramp_commit_early_frac", 0.3))

    def epsilon(step: int) -> float:
        if step >= eps_decay:
            return eps_end
        frac = step / max(1, eps_decay)
        return eps_start + frac * (eps_end - eps_start)

    step = 0
    episode = 0

    # ✅ Track best model by eval_success_rate
    best_success = -1.0
    milestone_max_layer_logged = False
    milestone_early_ramp_logged = False

    while step < total_steps:
        override_goal = None
        if len(replay) > 0 and goal_diversity_rng.random() < goal_replay_fraction:
            override_goal = vector_to_goal(replay.sample_achieved_goal(), env.n_layers)
        obs, goal = env.reset(goal=override_goal)
        done = False
        ep_steps = 0
        ep_success = 0

        # episode-level ramp tracking (final can be 0 if dismantled late)
        ep_max_ramp_height = 0.0
        ep_mean_ramp_height_sum = 0.0

        ep_max_layer = 0

        # episode-level ramp command stats
        ep_ramp_cmd_extend = 0
        ep_ramp_cmd_dismantle = 0
        ep_ramp_cmd_switch = 0
        ep_ramp_cmd_none = 0

        early_commit_step_limit = int(env.t_max * ramp_commit_early_frac)
        early_ramp_commit_step = None

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
            ep_max_layer = max(ep_max_layer, int(info.get("max_layer", 0)))

            if (
                not milestone_max_layer_logged
                and ep_max_layer > random_policy_ceiling
            ):
                milestone_max_layer_logged = True
                logger.log(
                    {
                        "type": "milestone",
                        "kind": "max_layer_exceeds_random_ceiling",
                        "episode": episode,
                        "step": step + 1,
                        "max_layer": ep_max_layer,
                    }
                )

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

            if (
                early_ramp_commit_step is None
                and ep_steps <= early_commit_step_limit
                and ramp_h >= ramp_commit_height
            ):
                early_ramp_commit_step = step

            if step >= warmup_steps and len(replay) >= batch_size:
                batch = replay.sample(batch_size=batch_size)
                losses = agent.update(batch)
                if step % 100 == 0:
                    log_event(
                        "train_step",
                        {
                            "step": step,
                            "episode": episode,
                            "epsilon": float(eps),
                            **losses,
                        },
                    )

            if step % int(rl_cfg["eval_every"]) == 0 and step >= warmup_steps:
                metrics = run_eval(
                    env,
                    agent,
                    n_episodes=30,
                    fixed_goal_set=True,
                    capability_probe_tiers=capability_probe_tiers,
                    capability_probe_episodes=capability_probe_episodes,
                    random_policy_ceiling=random_policy_ceiling,
                )
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

        if early_ramp_commit_step is not None and not milestone_early_ramp_logged:
            milestone_early_ramp_logged = True
            logger.log(
                {
                    "type": "milestone",
                    "kind": "early_ramp_commitment",
                    "episode": episode,
                    "step": int(early_ramp_commit_step),
                    "ramp_height": float(ep_max_ramp_height),
                }
            )

        phase_reached, failure_reason, irreversible_error = episode_phase_outcome(
            env.state,
            goal,
            ep_success,
        )

        log_event(
            "episode",
            {
                "episode": episode,
                "step": step,
                "ep_steps": ep_steps,
                "success": ep_success,
                "phase_reached": int(phase_reached),
                "phase_failure_reason": failure_reason,
                "irreversible_error": bool(irreversible_error),
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
            },
        )
        episode += 1

        if step > 0 and step % 5000 == 0:
            flush()

    # ✅ always save final at end of training
    save_ckpt("final")

    flush()


if __name__ == "__main__":
    main()
