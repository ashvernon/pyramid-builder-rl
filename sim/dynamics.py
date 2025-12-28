from __future__ import annotations
import math
from dataclasses import replace

from sim.state import PyramidState


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def ramp_efficiency(ramp_type: int, ramp_height_norm: float, base: list[float], k: list[float]) -> float:
    eff = base[ramp_type] - k[ramp_type] * ramp_height_norm
    return clamp(eff, 0.2, 1.0)


def step_state(state: PyramidState, action: dict, cfg: dict) -> PyramidState:
    s = replace(state)
    s.p = list(state.p)

    n = s.n_layers
    s.t += 1

    ramp_cmd = int(action.get("ramp_cmd", 0))
    ramp_type = int(action.get("ramp_type", s.ramp_type))

    # NEW: gentler dismantle so ramp can accumulate under exploration
    # Old: 0.25 (one dismantle wiped ~dozens of extend steps)
    dismantle_rate = float(cfg.get("ramp_dismantle_rate", 0.02))

    if ramp_cmd == 3:
        s.ramp_type = 0
        s.ramp_height = clamp(s.ramp_height - dismantle_rate, 0.0, float(n))
    elif ramp_cmd == 2:
        s.ramp_type = ramp_type
        s.fatigue = clamp(s.fatigue + 0.03, 0.0, 1.0)
    elif ramp_cmd == 1:
        s.ramp_type = s.ramp_type if s.ramp_type != 0 else 1
        s.ramp_height = clamp(
            s.ramp_height + cfg["ramp_build_rate"] * (s.workers_place / max(1, s.workers_total)),
            0.0,
            float(n),
        )

    s.workers_quarry = int(action["wq"])
    s.workers_haul = int(action["wh"])
    s.workers_place = int(action["wp"])

    heat = float(s.heat)
    fatigue = float(s.fatigue)
    effort = 1.0 - 0.5 * fatigue
    effort = clamp(effort, 0.3, 1.0)

    ramp_h_norm = s.ramp_height / max(1.0, float(n))
    eff = ramp_efficiency(s.ramp_type, ramp_h_norm, cfg["ramp_type_base"], cfg["ramp_type_k"])

    gap = max(0.0, float(s.current_layer) - float(s.ramp_height))
    gap_factor = math.exp(-gap / 4.0)

    produced = int(cfg["alpha_q"] * s.workers_quarry * effort)
    s.quarry_stock += produced

    haul_cap = int(cfg["alpha_h"] * s.workers_haul * eff * gap_factor)
    moved = min(haul_cap, s.quarry_stock)
    s.quarry_stock -= moved
    s.site_stock += moved

    place_cap = int(cfg["alpha_p"] * s.workers_place * eff * gap_factor)
    usable = min(place_cap, s.site_stock)
    s.site_stock -= usable

    cost = 200 + 8 * s.current_layer
    delta = usable / max(1.0, float(cost))
    s.p[s.current_layer] = min(1.0, s.p[s.current_layer] + delta)

    if s.p[s.current_layer] >= 1.0 and s.current_layer < n - 1:
        s.current_layer += 1

    intensity = (s.workers_quarry + s.workers_haul + s.workers_place) / max(1.0, float(s.workers_total))
    s.fatigue = clamp(s.fatigue + 0.0025 * intensity * (0.5 + heat), 0.0, 1.0)
    s.fatigue = clamp(s.fatigue - 0.0008, 0.0, 1.0)

    if s.t >= cfg["t_max"] or (s.current_layer == n - 1 and s.p[n - 1] >= 1.0):
        s.done = True

    return s
