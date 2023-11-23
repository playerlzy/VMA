import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.weight_init import weight_init

class Attention(nn.Module):
    """
    multi-head attention with relative position feature 
    """
    def __init__(self,
                 hidden_dim,
                 num_heads,
                 head_dim,
                 dropout,
                 bipartite) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_k = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.to_v = nn.Linear(hidden_dim, head_dim * num_heads)

        self.to_k_r = nn.Linear(hidden_dim, head_dim * num_heads, bias=False)
        self.to_v_r = nn.Linear(hidden_dim, head_dim * num_heads)

        self.to_s = nn.Linear(hidden_dim, head_dim * num_heads)
        self.to_g = nn.Linear(head_dim * num_heads + hidden_dim, head_dim * num_heads)
        self.to_out = nn.Linear(head_dim * num_heads, hidden_dim)
        self.attn_drop = nn.Dropout(dropout)

        self.ff_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        if bipartite:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim)
        else:
            self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
            self.attn_prenorm_x_dst = self.attn_prenorm_x_src

        self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
        self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim)

        self.attn_prenorm_r = nn.LayerNorm(hidden_dim)
        self.attn_postnorm = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)
        self.ff_postnorm = nn.LayerNorm(hidden_dim)
        self.apply(weight_init)

    def forward(self,
                x_src,
                x_dst,
                r,
                src_mask,
                dst_mask,
                edge_mask):

        x_src = self.attn_prenorm_x_src(x_src)
        x_dst = self.attn_prenorm_x_dst(x_dst)
        
        r = self.attn_prenorm_r(r)
        attn = x_dst + self.attn_postnorm(self._attn_block(
            x_src, x_dst, r, src_mask, dst_mask, edge_mask)
        )
        attn = attn + self.ff_postnorm(self._ff_block(self.ff_prenorm(attn)))
        return attn
    
    def _attn_block(self, x_src, x_dst, r, src_mask, dst_mask, edge_mask):

        len_q = x_dst.shape[1]
        len_k = x_src.shape[1]
        q = self.to_q(x_dst).reshape(-1, len_q, self.num_heads, self.head_dim)
        k = self.to_k(x_src).reshape(-1, len_k, self.num_heads, self.head_dim)
        v = self.to_v(x_src).reshape(-1, len_k, self.num_heads, self.head_dim)
        
        k_r = self.to_k_r(r).reshape(-1, len_q, len_k, self.num_heads, self.head_dim)
        v_r = self.to_v_r(r).reshape(-1, len_q, len_k, self.num_heads, self.head_dim)
        #src_mask: [B, M]
        #dst_mask: [B, N]
        #r: [B, N, M, D]
        q = q.reshape(-1, len_q * self.num_heads, 1, self.head_dim)
        k = (k.unsqueeze(1) + k_r).transpose(2, 3).reshape(
            -1, len_q * self.num_heads, len_k, self.head_dim)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale #[B, head*N, 1, M]
        valid_mask = (
            (src_mask.unsqueeze(1) | dst_mask.unsqueeze(2)) | (edge_mask == 0)
        ).unsqueeze(-2).repeat(1, self.num_heads, 1, 1)
        attn.masked_fill(valid_mask, -np.inf)
        score = F.softmax(attn, dim=-1)
        score.masked_fill(torch.isnan(score), 0)
        
        v = (v.unsqueeze(1) + v_r).transpose(2, 3).reshape(
            -1, len_q * self.num_heads, len_k, self.head_dim)
        output = torch.matmul(score, v).squeeze(-2).reshape(
            -1, len_q, self.num_heads, self.head_dim
        )
        output = output.reshape(-1, len_q, self.num_heads * self.head_dim)
        g = torch.sigmoid(self.to_g(torch.cat([output, x_dst], dim=-1)))
        return self.to_out(output + g * (self.to_s(x_dst) - output))

    def _ff_block(self, x):
        return self.ff_mlp(x)