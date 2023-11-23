import torch
import torch.nn as nn
from utils.weight_init import weight_init

class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)