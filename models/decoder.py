import math
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.common_layers import MLPLayer
from utils.weight_init import weight_init


class SimpleDecoder(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int) -> None:
        super(SimpleDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes

        self.multimodal_proj = nn.Linear(in_features=hidden_dim, out_features=num_modes * hidden_dim)
        self.to_loc_pos = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_future_steps * output_dim),
        )
        self.to_scale_pos = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_future_steps * output_dim),
        )
        self.to_loc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_future_steps),
        )
        self.to_conc_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_future_steps),
        )
        self.to_pi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weight_init)

    def forward(self, x_a) -> Dict[str, torch.Tensor]:
        
        x_a = self.multimodal_proj(x_a).reshape(-1, self.num_modes, self.hidden_dim) #[N,6,D]
        loc_pos = torch.cumsum(
            self.to_loc_pos(x_a).reshape(-1, self.num_modes, self.num_future_steps, self.output_dim),
            dim=-2
        )
        scale_pos = torch.cumsum(
            F.elu_(self.to_scale_pos(x_a).reshape(-1, self.num_modes, self.num_future_steps, self.output_dim), alpha=1.0) + 1.0,
            dim=-2
        ) + 0.1
        loc_head = torch.cumsum(
            torch.tanh(self.to_loc_head(x_a).reshape(-1, self.num_modes, self.num_future_steps, 1)) * math.pi,
            dim=-2
        )
        conc_head = 1.0 / (torch.cumsum(F.elu_(self.to_conc_head(x_a).reshape(-1, self.num_modes, self.num_future_steps, 1)) + 1.0,
                                                    dim=-2) + 0.02)

        pi = self.to_pi(x_a).squeeze(-1)

    
        return {
            'loc_pos': loc_pos,
            'scale_pos': scale_pos,
            'loc_head': loc_head,
            'conc_head': conc_head,
            'pi': pi,
        }
