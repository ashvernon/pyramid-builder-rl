from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return x + h


class ResMLP(nn.Module):
    def __init__(self, input_dim: int, width: int, depth: int, output_dim: int):
        super().__init__()
        self.inp = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(int(depth))])
        self.out = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return self.out(h)
