# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax

from utils import weight_init


class AttentionLayer(MessagePassing):

    def __init__(self) -> None:
        super(AttentionLayer, self).__init__(aggr='add', node_dim=0)
        self.num_heads = 8
        self.head_dim = 16

    def forward(self,
                x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                edge_index: torch.Tensor) -> torch.Tensor:
        x = self._attn_block(x, edge_index)
        return x

    def message(self,
                q_i: torch.Tensor,
                k_j: torch.Tensor,
                v_j: torch.Tensor,
                index: torch.Tensor,
                ptr: Optional[torch.Tensor]) -> torch.Tensor:
        sim = (q_i * k_j).sum(dim=-1)
        attn = softmax(sim, index, ptr)
        return v_j * attn.unsqueeze(-1)

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, self.num_heads * self.head_dim)
        return inputs

    def _attn_block(self,
                    x: torch.Tensor,
                    edge_index: torch.Tensor) -> torch.Tensor:
        q = x.view(-1, self.num_heads, self.head_dim)
        k = x.view(-1, self.num_heads, self.head_dim)
        v = x.view(-1, self.num_heads, self.head_dim)
        agg = self.propagate(edge_index=edge_index, x=x, q=q, k=k, v=v)
        return agg

test = AttentionLayer()
x = torch.ones(3, 128)
edge_index = torch.tensor([[0, 2], [1, 0]], dtype=torch.long)
print(test(x, edge_index))