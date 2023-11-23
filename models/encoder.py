from torch import Tensor

import torch.nn as nn

from .layers.sub_graph import SubGraph
from .layers.common_layers import PosEmbedding

from .layers.attention import Attention
from utils.weight_init import weight_init

class VMAEncoder(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,                 
                 m2m_layers: int,
                 a2a_layers: int,
                 num_map_types: int,
                 num_mark_types: int,
                 num_is_inter: int,
                 num_lane_edge: int,
                 num_agent_types: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.m2m_layers = m2m_layers
        self.a2a_layers = a2a_layers
        self.map_polyline_encoder = SubGraph('lane', in_channels=6, hidden_unit=hidden_dim)
        self.agent_polyline_encoder = SubGraph('agent', in_channels=6, hidden_unit=hidden_dim)

        self.map_type_embed = nn.Embedding(num_map_types, hidden_dim)
        self.left_mark_embed = nn.Embedding(num_mark_types, hidden_dim)
        self.right_mask_embed = nn.Embedding(num_mark_types, hidden_dim)
        self.is_inter_embed = nn.Embedding(num_is_inter, hidden_dim)
        self.edge_type_embed = nn.Embedding(num_lane_edge, hidden_dim)
        self.agent_type_embed = nn.Embedding(num_agent_types, hidden_dim)

        self.m2m_rel_pos_embed = PosEmbedding(4, hidden_dim)
        self.a2m_rel_pos_embed = PosEmbedding(4, hidden_dim)
        self.a2a_rel_pos_embed = PosEmbedding(4, hidden_dim)

        self.m2m_attention = nn.ModuleList(
            [Attention(
                hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, 
                dropout=dropout, bipartite=False
             )
             for _ in range(self.m2m_layers)]
        )
        self.a2m_attention = nn.ModuleList(
            [Attention(
                hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, 
                dropout=dropout, bipartite=True
             )
             for _ in range(self.a2a_layers)]
        )
        self.a2a_attention = nn.ModuleList(
            [Attention(
                hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, 
                dropout=dropout, bipartite=False
             )
             for _ in range(self.a2a_layers)]
        )
        self.apply(weight_init)

    def forward(self, data) -> Tensor:
        #map data
        map_feature = data["map_feature"]
        map_mask = data["lane_mask"]
        map_adj = data["m2m_edge"]
        m2m_feature = data["m2m_feature"]
        map_type = data["lane_type"]
        left_mark_type = data["lane_left_type"]
        right_mark_type = data["lane_right_type"]
        is_inter = data["is_intersection"]

        map_res = self.map_encoder(
            map_feature,
            map_mask,
            map_adj,
            m2m_feature,
            map_type,
            left_mark_type,
            right_mark_type,
            is_inter
        )

        #agent_data
        agent_feature = data["agent_feature"]
        agent_type = data["object_type"]
        agent_mask = data["padding_mask"][:, :, :self.num_historical_steps]
        a2a_edge = data["a2a_edge"]
        a2a_feature = data["a2a_feature"]
        a2m_edge = data["a2m_edge"]
        a2m_feature = data["a2m_feature"]

        agent_res = self.agent_encoder(
            agent_feature, 
            agent_type, 
            agent_mask, 
            a2a_edge,
            a2a_feature,
            a2m_edge,
            a2m_feature,
            map_res,
            map_mask
        )

        return agent_res

    def map_encoder(self, 
                    map_feature: Tensor, 
                    map_mask: Tensor, 
                    map_adj: Tensor,
                    m2m_featrue: Tensor,
                    map_type: Tensor,
                    left_mark_type: Tensor,
                    right_mark_type: Tensor,
                    is_inter: Tensor) -> Tensor:
        
        #local
        map_attr = [self.map_type_embed(map_type.long()),
                    self.left_mark_embed(left_mark_type.long()),
                    self.right_mask_embed(right_mark_type.long()),
                    self.is_inter_embed(is_inter.long())]
        polygon_feature = self.map_polyline_encoder(map_feature, ~map_mask, map_attr)
        
        #global
        edge_feature = self.edge_type_embed(map_adj.long())
        m2m_feature = self.m2m_rel_pos_embed(m2m_featrue, edge_feature)
        for i in range(self.m2m_layers):
            polygon_feature = self.m2m_attention[i](
                polygon_feature, polygon_feature, m2m_feature,
                map_mask, map_mask, map_adj
            )
        return polygon_feature

    def agent_encoder(self, 
                      agent_feature, 
                      agent_type,
                      agent_mask, 
                      a2a_edge,
                      a2a_feature,
                      a2m_edge,
                      a2m_feature,
                      map_res,
                      map_mask) -> Tensor:
        #local
        agent_attr = [self.agent_type_embed(agent_type.long())]
        agent_feature = self.agent_polyline_encoder(agent_feature, ~agent_mask, agent_attr)

        #global
        a2m_feature = self.a2m_rel_pos_embed(a2m_feature)
        a2a_feature = self.a2a_rel_pos_embed(a2a_feature)
        for i in range(self.a2a_layers):
            agent_feature = self.a2a_attention[i](
                agent_feature, agent_feature, a2a_feature, 
                agent_mask[..., -1], agent_mask[..., -1], a2a_edge
            )
            agent_feature = self.a2m_attention[i](
                map_res, agent_feature, a2m_feature, map_mask, 
                agent_mask[..., -1], a2m_edge
            )

        return agent_feature