from __future__ import annotations
from dataclasses import dataclass


@dataclass
class EpisodeStats:
    success: int
    steps: int
    max_layer: int
    final_ramp_height: float
    starvation_ticks: int
