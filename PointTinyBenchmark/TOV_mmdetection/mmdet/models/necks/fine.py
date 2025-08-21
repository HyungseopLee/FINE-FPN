# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np

class FeatureInteractionNEtowrk(nn.Module):
    def __init__(self, dim, nhead, dropout=0.0, activation='gelu', is_first=False):
        '''
        Feature Interaction NEtwork (FINE) for RT-DETR
        Args:
            dim: input feature dimension
            nhead: number of attention heads
            dropout: dropout rate
            activation: activation function
            is_first: whether this is the lowest level of the feature pyramid
        Returns:
            out: semantically aligned low-level feature
        '''
        
        super().__init__()
        self.eps = 1e-6
        
        # multi-head attention
        self.nhead = nhead
        self.head_dim = dim // nhead
        
        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # bounded gradient
        self.attn_norm = nn.RMSNorm(self.head_dim)
        
        # FFN
        self.linear1 = nn.Linear(dim, dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Spatial Bottlneck Design
        self.avg_pool_high = nn.Identity()
        self.avg_pool_low = nn.Identity()
        if is_first:
            self.avg_pool_low = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.avg_pool_high = nn.AvgPool2d(kernel_size=2, stride=2)
            self.avg_pool_low = nn.AvgPool2d(kernel_size=4, stride=4)
        
    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        input shape: (h, w)
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed
    
    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)
        return nn.Parameter(index, requires_grad=False)
    
    def forward(self, low, high):
        original_low = low
        
        # 1. Spatial Bottleneck Design: Down()
        low = self.avg_pool_low(low)
        high = self.avg_pool_high(high)
        
        bs, c_low, h, w = low.shape
        _, c_high, H, W = high.shape
        tgt_len = h * w
        src_len = H * W
        
        residual = rearrange(low, 'b c h w -> b (h w) c')
        low_flat = rearrange(low, 'b c h w -> (h w) b c')
        high_flat = rearrange(high, 'b c h w -> (h w) b c')
        
        pos = self.build_2d_sincos_position_embedding(w=w, h=h, embed_dim=c_low).to(low.device)
        pos = pos.expand(bs, -1, -1).permute(1, 0, 2)  # [HW, B, C]
        low_flat = self.with_pos_embed(low_flat, pos)  # [HW, B, C]
        high_flat = self.with_pos_embed(high_flat, pos)  # [HW, B, C]
        
        # 2. Cross-level multi-head attention with linear complexity
        low_flat = low_flat.to(self.q_proj.weight.dtype)
        high_flat = high_flat.to(self.k_proj.weight.dtype)
        Q = self.q_proj(low_flat)
        K = self.k_proj(high_flat)
        V = self.v_proj(high_flat) # [HW, B, C]
        Q = rearrange(Q, 'hw b (h d) -> (b h) hw d', h=self.nhead) # [B*h, HW, C/h]
        K = rearrange(K, 'hw b (h d) -> (b h) hw d', h=self.nhead) # [B*h, HW, C/h]
        V = rearrange(V, 'hw b (h d) -> (b h) hw d', h=self.nhead) # [B*h, HW, C/h]
        
        # l1-norm kernel function: QK for training stability (https://github.com/UCDvision/sima/blob/main/sima.py)
        Q = F.normalize(Q, p=1, dim=-2, eps=self.eps) 
        K = F.normalize(K, p=1, dim=-2, eps=self.eps) 
        
        # # CosForm transform for nonlinear reweighting (https://github.com/OpenNLPLab/cosFormer/blob/main/cosformer.py)
        # m = max(src_len, tgt_len)
        # weight_index = self.get_index(m).to(Q)
        # Q_ = torch.cat([
        #     Q * torch.sin(weight_index[:, :tgt_len, :] / m), 
        #     Q * torch.cos(weight_index[:, :tgt_len, :] / m)
        # ], dim=-1)
        # K_ = torch.cat([
        #     K * torch.sin(weight_index[:, :src_len, :] / m), 
        #     K * torch.cos(weight_index[:, :src_len, :] / m)
        # ], dim=-1)
        KV_ = torch.einsum('nld,nlm->ndm', K, V) # [B*h, N, d]
        attn_output = torch.einsum('nld,ndm->nlm', Q, KV_)  # [B*h, N, d]
        # Replace unstable attention scaling with RMSNorm for gradient stability
        attn_output = self.attn_norm(attn_output)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bs, -1) # [L, B, C]
        out = self.out_proj(attn_output) #  [N, B, C]
        out = out.permute(1, 0, 2)  # [B, N, C]
        
        # FFN
        out = self.norm1(out)
        out = residual + self.dropout1(out)
        residual = out
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = self.norm2(out)
        out = residual + self.dropout2(out) # [B, HW, C]
        
        # 3. Spatial Bottleneck Design: Up()
        low_sa = out.permute(0, 2, 1).contiguous().view(bs, c_low, h, w)  # [B, C, H, W]
        low_sa = F.interpolate(low_sa, size=original_low.shape[2:], mode='bilinear', align_corners=False)
        low_sa = low_sa * original_low
        
        # Semantically Aligned low-level feature
        return low_sa



class FeatureInteractionNEtowrkV2(nn.Module):
    def __init__(self, dim, nhead, dropout=0.0, activation='gelu', is_first=False):
        '''
        Feature Interaction NEtwork (FINE) for RT-DETR
        Args:
            dim: input feature dimension
            nhead: number of attention heads
            dropout: dropout rate
            activation: activation function
            is_first: whether this is the lowest level of the feature pyramid
        Returns:
            out: semantically aligned low-level feature
        '''
        
        super().__init__()
        self.eps = 1e-6
        
        # multi-head attention
        self.nhead = nhead
        self.head_dim = dim // nhead
        
        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # bounded gradient
        # self.attn_norm = nn.RMSNorm(self.head_dim)
        self.attn_norm = nn.LayerNorm(self.head_dim)
        
        # FFN
        self.linear1 = nn.Linear(dim, dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Spatial Bottlneck Design
        self.avg_pool_high = nn.Identity()
        self.avg_pool_low = nn.Identity()
        if is_first:
            self.avg_pool_low = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.avg_pool_high = nn.AvgPool2d(kernel_size=2, stride=2)
            self.avg_pool_low = nn.AvgPool2d(kernel_size=4, stride=4)
            
        # Fusion factor token: image-wise learnable scalar
        self.fusion_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.fusion_head = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1)
        )
        
    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
        input shape: (h, w)
        """
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed
    
    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)
        return nn.Parameter(index, requires_grad=False)
    
    def forward(self, low, high):
        original_low = low
        
        # 1. Spatial Bottleneck Design: Down()
        low = self.avg_pool_low(low)
        high = self.avg_pool_high(high)
        
        bs, c_low, h, w = low.shape
        _, c_high, H, W = high.shape
        tgt_len = h * w
        src_len = H * W
        
        residual = rearrange(low, 'b c h w -> b (h w) c')
        low_flat = rearrange(low, 'b c h w -> (h w) b c')
        high_flat = rearrange(high, 'b c h w -> (h w) b c')
        
        pos = self.build_2d_sincos_position_embedding(w=w, h=h, embed_dim=c_low).to(low.device)
        pos = pos.expand(bs, -1, -1).permute(1, 0, 2)  # [HW, B, C]
        low_flat = self.with_pos_embed(low_flat, pos)  # [HW, B, C]
        high_flat = self.with_pos_embed(high_flat, pos)  # [HW, B, C]
        
        # Add fusion factor token to high-level sequence
        fusion_token = self.fusion_token.expand(1, bs, -1)
        low_flat = torch.cat([fusion_token, low_flat], dim=0)  # [1+HW, B, C]
        
        # 2. Cross-level multi-head attention with linear complexity
        low_flat = low_flat.to(self.q_proj.weight.dtype)   # [1+HW, B, C]
        high_flat = high_flat.to(self.k_proj.weight.dtype) # [HW, B, C]
        
        Q = self.q_proj(low_flat)  # [1+HW, B, C]
        K = self.k_proj(high_flat) # [HW, B, C]
        V = self.v_proj(high_flat) # [HW, B, C]
        Q = rearrange(Q, 'hw b (h d) -> (b h) hw d', h=self.nhead) # [B*h, 1+HW, C/h]
        K = rearrange(K, 'hw b (h d) -> (b h) hw d', h=self.nhead) # [B*h, HW, C/h]
        V = rearrange(V, 'hw b (h d) -> (b h) hw d', h=self.nhead) # [B*h, HW, C/h]
        
        # l1-norm kernel function: QK for training stability (https://github.com/UCDvision/sima/blob/main/sima.py)
        Q = F.normalize(Q, p=1, dim=-2, eps=self.eps) 
        K = F.normalize(K, p=1, dim=-2, eps=self.eps)
        KV_ = torch.einsum('nld,nlm->ndm', K, V)
        attn_output = torch.einsum('nld,ndm->nlm', Q, KV_)
        attn_output = self.attn_norm(attn_output)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len+1, bs, -1)
        out = self.out_proj(attn_output)
        out = out.permute(1, 0, 2)
        
        # 3. Fusion factor
        fusion_factor = ((torch.tanh(self.fusion_head(out[:, 0, :])) + 1) / 2).view(bs)  # [B]
        out = out[:, 1:, :]
        
        # FFN
        out = self.norm1(out)
        out = residual + self.dropout1(out)
        residual = out
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = self.norm2(out)
        out = residual + self.dropout2(out) # [B, 1+HW, C]
        
        # 4. Spatial Bottleneck Design: Up()
        low_sa = out.permute(0, 2, 1).contiguous().view(bs, c_low, h, w)  # [B, C, H, W]
        low_sa = F.interpolate(low_sa, size=original_low.shape[2:], mode='bilinear', align_corners=False)
        low_sa = low_sa * original_low
        
        # Semantically Aligned low-level feature
        return low_sa, fusion_factor