# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


import torch
import torch.nn as nn
from .common_layers import MLP

class PointNetPolylineEncoder(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_dim, 
                 output_dim) -> None:
        super().__init__()
        self.pre_mlp = MLP(in_channels, hidden_dim)
        self.mlps = nn.Sequential(
            MLP(2 * hidden_dim, hidden_dim), 
            MLP(hidden_dim, hidden_dim)
        )
        self.out_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, polylines, polylines_mask, polylines_attr=None):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):

        Returns:
        """
        batch_size, num_polylines,  num_points_each_polylines, C = polylines.shape

        # pre-mlp
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])  # (N, C)
        polylines_feature = polylines.new_zeros(batch_size, num_polylines,  num_points_each_polylines, polylines_feature_valid.shape[-1])
        polylines_feature[polylines_mask] = polylines_feature_valid

        # get global feature
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)

        # mlp
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        feature_buffers[polylines_mask] = polylines_feature_valid

        # max-pooling
        feature_buffers = feature_buffers.max(dim=2)[0]  # (batch_size, num_polylines, C)
        
        # out-mlp 
        valid_mask = (polylines_mask.sum(dim=-1) > 0)
        feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])  # (N, C)
        feature_buffers = feature_buffers.new_zeros(batch_size, num_polylines, feature_buffers_valid.shape[-1])
        feature_buffers[valid_mask] = feature_buffers_valid
        return feature_buffers