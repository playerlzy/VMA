import torch
import torch.nn as nn
from utils.weight_init import weight_init

class MLPLayer(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int) -> None:
        super(MLPLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
    
class PosEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.mlp = MLPLayer(input_dim, hidden_dim, hidden_dim)
        self.to_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.apply(weight_init)

    def forward(self, input, attr=None):
        x = self.mlp(input)
        if attr is not None:
            x = x + attr
        return self.to_out(x)