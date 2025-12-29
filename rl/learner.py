from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rl.networks import ResMLP


class GoalDQNAgent:
    """Goal-conditioned DQN agent (discrete actions)."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        n_actions: int,
        width: int,
        depth: int,
        lr: float,
        gamma: float,
        target_update_every: int = 200,
        target_tau: float = 1.0,
        device: str | None = None,
    ):
        self.obs_dim = int(obs_dim)
        self.goal_dim = int(goal_dim)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.target_update_every = int(target_update_every)
        self.target_tau = float(target_tau)
        self.step_count = 0

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        inp = self.obs_dim + self.goal_dim
        self.q = ResMLP(inp, width=int(width), depth=int(depth), output_dim=self.n_actions).to(self.device)
        self.q_targ = ResMLP(inp, width=int(width), depth=int(depth), output_dim=self.n_actions).to(self.device)
        self.q_targ.load_state_dict(self.q.state_dict())
        self.q_targ.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=float(lr))

        # Optional: track best eval so training can save "best"
        self.best_eval_success: float = -1.0

    def act(self, obs: np.ndarray, goal: np.ndarray, eps: float) -> int:
        if np.random.random() < eps:
            return int(np.random.randint(0, self.n_actions))
        x = np.concatenate([obs, goal], axis=0).astype(np.float32)
        xt = torch.from_numpy(x).to(self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q(xt).squeeze(0)
        return int(torch.argmax(qvals).item())

    def update(self, batch) -> Dict[str, float]:
        self.step_count += 1

        obs = torch.from_numpy(batch.obs).to(self.device)
        goal = torch.from_numpy(batch.goal).to(self.device)
        action = torch.from_numpy(batch.action).to(self.device)
        next_obs = torch.from_numpy(batch.next_obs).to(self.device)
        done = torch.from_numpy(batch.done).to(self.device)
        reward = torch.from_numpy(batch.reward).to(self.device)

        x = torch.cat([obs, goal], dim=1)
        q_all = self.q(x)
        q = q_all.gather(1, action.view(-1, 1)).squeeze(1)

        with torch.no_grad():
            x2 = torch.cat([next_obs, goal], dim=1)
            q2 = self.q_targ(x2).max(dim=1).values
            target = reward + (1.0 - done) * self.gamma * q2

        loss = nn.functional.smooth_l1_loss(q, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        self.opt.step()

        if self.step_count % self.target_update_every == 0:
            self._update_target()

        return {"q_loss": float(loss.item())}

    def _update_target(self):
        if self.target_tau >= 1.0:
            self.q_targ.load_state_dict(self.q.state_dict())
            return
        with torch.no_grad():
            for p, pt in zip(self.q.parameters(), self.q_targ.parameters()):
                pt.data.mul_(1.0 - self.target_tau).add_(self.target_tau * p.data)

    # -----------------------
    # Checkpointing
    # -----------------------

    def state_dict(self) -> Dict[str, Any]:
        """Everything needed to resume training."""
        return {
            "meta": {
                "obs_dim": self.obs_dim,
                "goal_dim": self.goal_dim,
                "n_actions": self.n_actions,
                "gamma": self.gamma,
                "target_update_every": self.target_update_every,
                "target_tau": self.target_tau,
                "device": self.device,
            },
            "step_count": int(self.step_count),
            "best_eval_success": float(self.best_eval_success),
            "q": self.q.state_dict(),
            "q_targ": self.q_targ.state_dict(),
            "opt": self.opt.state_dict(),
        }

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path: Path, *, strict: bool = True, load_opt: bool = True):
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device)

        self.q.load_state_dict(ckpt["q"], strict=strict)
        self.q_targ.load_state_dict(ckpt["q_targ"], strict=strict)

        self.step_count = int(ckpt.get("step_count", 0))
        self.best_eval_success = float(ckpt.get("best_eval_success", -1.0))

        if load_opt and "opt" in ckpt:
            try:
                self.opt.load_state_dict(ckpt["opt"])
            except Exception:
                # Optimizer state can fail to load if hyperparams/param groups changed
                pass

        # Ensure correct modes
        self.q.train()
        self.q_targ.eval()

    def maybe_save(
        self,
        ckpt_dir: Path,
        *,
        step: int,
        save_every: int = 10_000,
        eval_success_rate: Optional[float] = None,
        prefix: str = "agent",
    ) -> Dict[str, str]:
        """
        Convenience helper:
        - saves periodic checkpoints every `save_every` steps
        - saves best checkpoint if eval_success_rate improves
        """
        ckpt_dir = Path(ckpt_dir)
        saved: Dict[str, str] = {}

        if save_every > 0 and (step % save_every == 0):
            p = ckpt_dir / f"{prefix}_step_{step}.pt"
            self.save(p)
            saved["periodic"] = str(p)

        if eval_success_rate is not None and eval_success_rate > self.best_eval_success:
            self.best_eval_success = float(eval_success_rate)
            p = ckpt_dir / f"{prefix}_best.pt"
            self.save(p)
            saved["best"] = str(p)

        return saved
