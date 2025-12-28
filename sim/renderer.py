from __future__ import annotations

import pygame
from typing import Optional

from sim.state import PyramidState
from sim.goals import Goal


class Renderer:
    """Simple Pygame renderer for pyramid + ramp + HUD."""

    def __init__(self, n_layers: int):
        pygame.init()
        self.n_layers = n_layers
        self.w, self.h = 1100, 700
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Pyramid Goal RL (Play)")
        self.font = pygame.font.SysFont(None, 22)
        self.clock = pygame.time.Clock()

    def render(self, state: PyramidState, goal: Goal, info: Optional[dict] = None):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                raise SystemExit

        self.screen.fill((20, 20, 24))
        self._draw_pyramid(state)
        self._draw_ramp(state)
        self._draw_hud(state, goal, info or {})
        pygame.display.flip()
        self.clock.tick(60)

    def _draw_pyramid(self, state: PyramidState):
        base_w = 650
        layer_h = 12
        x0 = 120
        y0 = 620

        for i in range(self.n_layers):
            w = int(base_w * (1.0 - 0.6 * (i / max(1, self.n_layers - 1))))
            y = y0 - i * layer_h
            x = x0 + (base_w - w) // 2
            pygame.draw.rect(self.screen, (60, 60, 70), (x, y, w, layer_h), 1)
            p = max(0.0, min(1.0, state.p[i]))
            fill_w = int(w * p)
            pygame.draw.rect(self.screen, (170, 150, 90), (x, y, fill_w, layer_h))

        i = state.current_layer
        w = int(base_w * (1.0 - 0.6 * (i / max(1, self.n_layers - 1))))
        y = y0 - i * layer_h
        x = x0 + (base_w - w) // 2
        pygame.draw.rect(self.screen, (220, 210, 140), (x, y, w, layer_h), 2)

    def _draw_ramp(self, state: PyramidState):
        x1, y1 = 140, 640
        max_y = 640
        min_y = 140
        y2 = int(max_y - (max_y - min_y) * (state.ramp_height / max(1.0, float(state.n_layers))))
        x2 = 320
        pygame.draw.line(self.screen, (120, 180, 220), (x1, y1), (x2, y2), 4)

        labels = ["none", "straight", "zigzag", "spiral"]
        txt = self.font.render(f"Ramp: {labels[state.ramp_type]}  height={state.ramp_height:.1f}", True, (220, 220, 220))
        self.screen.blit(txt, (20, 20))

    def _draw_hud(self, state: PyramidState, goal: Goal, info: dict):
        lines = [
            f"t={state.t}  layer={state.current_layer}/{state.n_layers-1}  fatigue={state.fatigue:.2f}",
            f"quarry={state.quarry_stock}  site={state.site_stock}  wq/wh/wp={state.workers_quarry}/{state.workers_haul}/{state.workers_place}",
            f"goal_kind={goal.kind}  success={info.get('success', 0)}",
        ]
        y = 48
        for ln in lines:
            surf = self.font.render(ln, True, (230, 230, 230))
            self.screen.blit(surf, (20, y))
            y += 22

    def close(self):
        pygame.quit()
