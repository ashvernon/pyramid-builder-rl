from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class PyramidState:
    """State for the pyramid construction simulation.

    Keep this dataclass logic-free; transitions belong in sim/dynamics.py.
    """

    t: int

    # Pyramid progress
    n_layers: int
    p: List[float]             # progress per layer, each in [0,1]
    current_layer: int

    # Ramp
    ramp_type: int             # 0 none, 1 straight, 2 zigzag, 3 spiral
    ramp_height: float         # in layer units (0..n_layers)
    ramp_length: float         # abstracted

    # Logistics buffers
    quarry_stock: int
    in_transit: int
    site_stock: int

    # Workforce
    workers_total: int
    workers_quarry: int
    workers_haul: int
    workers_place: int

    # Constraints
    friction: float            # 0..1
    heat: float                # 0..1
    fatigue: float             # 0..1

    done: bool = False
