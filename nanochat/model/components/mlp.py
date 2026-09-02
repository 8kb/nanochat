import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.model.components.linear import Linear


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.c_fc = Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = Linear(4 * n_embd, n_embd, bias=False)

    @torch.no_grad()
    def init_weights(self):
        s = 3**0.5 * self.n_embd**-0.5
        torch.nn.init.uniform_(self.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
        torch.nn.init.zeros_(self.c_proj.weight)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x
