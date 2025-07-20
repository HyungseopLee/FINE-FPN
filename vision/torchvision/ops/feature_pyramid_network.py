from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import torch.nn.functional as F
from torch import nn, Tensor

from ..ops.misc import Conv2dNormActivation
from ..utils import _log_api_usage_once

import torch
import math

class ExtraFPNBlock(nn.Module):
    """
    Base class for the extra block in the FPN.

    Args:
        results (List[Tensor]): the result of the FPN
        x (List[Tensor]): the original feature maps
        names (List[str]): the names for each one of the
            original feature maps

    Returns:
        results (List[Tensor]): the extended set of results
            of the FPN
        names (List[str]): the extended set of names for the results
    """

    def forward(
        self,
        results: List[Tensor],
        x: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        pass

def get_activation(act: str, inpace: bool=True):
    """get activation
    """
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act 

    act = act.lower()
    
    if act == 'silu' or act == 'swish':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()

    elif act == 'hardsigmoid':
        m = nn.Hardsigmoid()

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 


class SpatialAlignCrossAttention(nn.Module):
    def __init__(self, dim, dropout=0.0, activation='gelu', use_fusion_token=True):
        super().__init__()

        self.normalize_before = False
        self.dropout = dropout
        self.use_fusion_token = use_fusion_token

        high_token = torch.randn(1, 1, dim)
        self.high_fusion_factor_token = nn.Parameter(high_token)   
        
        self.high_fusion_factor_head = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1)
        )     
        
        self.cross_attn = nn.MultiheadAttention(dim, 8, dropout=0.0, batch_first=True)

        self.linear1 = nn.Linear(dim, dim * 4)
        self.dropout_layer = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 4, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

        self.avg_pool_2x = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_4x = nn.AvgPool2d(kernel_size=4, stride=4)
        

    @staticmethod
    def get_sinusoidal_pos_embed_2d(h, w, dim, device):
        assert dim % 4 == 0, f"Channel dim ({dim}) must be divisible by 4 for 2D positional encoding"
        div_term = torch.exp(torch.arange(0, dim // 4, device=device) * (-torch.log(torch.tensor(10000.0)) / (dim // 4)))

        pos_h = torch.arange(h, device=device).reshape(h, 1)
        pe_h = torch.zeros(h, dim // 2, device=device)
        sin_h = torch.sin(pos_h * div_term)
        cos_h = torch.cos(pos_h * div_term)
        pe_h[:, 0::2] = sin_h
        pe_h[:, 1::2] = cos_h

        pos_w = torch.arange(w, device=device).reshape(w, 1)
        pe_w = torch.zeros(w, dim // 2, device=device)
        sin_w = torch.sin(pos_w * div_term)
        cos_w = torch.cos(pos_w * div_term)
        pe_w[:, 0::2] = sin_w
        pe_w[:, 1::2] = cos_w

        pe_h = pe_h.unsqueeze(1).expand(-1, w, -1)
        pe_w = pe_w.unsqueeze(0).expand(h, -1, -1)
        pe = torch.cat([pe_h, pe_w], dim=-1)
        return pe.reshape(1, h * w, dim)


    @staticmethod
    def get_sinusoidal_pos_embed_1d(seq_len, embed_dim, device):
        """
        Generate 1D sinusoidal positional embeddings.

        Args:
        - seq_len: Length of the sequence.
        - embed_dim: The dimension of the embedding.
        - device: The device to store the tensor.

        Returns:
        - A tensor of shape (1, seq_len, embed_dim).
        """
        # Sinusoidal position encoding (standard method)
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0., embed_dim, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        return pe.unsqueeze(0)  # Add batch dimension (1, seq_len, embed_dim)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed
    
    def rescaled_tanh_0_2(self, x):
        x = torch.tanh(x) # [-1, 1]
        return (x + 1)  # Rescale to [0, 2]

    def forward(self, a3, a4, idx):
        original_a3 = a3

        # Downsample features
        if idx == 2: # mask r-cnn
        # if idx == 1: # retinanet, fcos
            a3_down = self.avg_pool_2x(a3)
        else:
            a3_down = self.avg_pool_4x(a3)
            a4 = self.avg_pool_2x(a4)

        bs, c_a3, h, w = a3_down.shape
        _, c_a4, H, W = a4.shape

        # Flatten spatial dimensions
        q = a3_down.flatten(2).permute(0, 2, 1)  # [bs, hw, dim]
        k = a4.flatten(2).permute(0, 2, 1)       # [bs, HW, dim]
        v = k.clone()
        
        q_pe = self.get_sinusoidal_pos_embed_1d(h*w, c_a3, q.device)
        k_pe = self.get_sinusoidal_pos_embed_1d(H*W, c_a4, k.device)
        q = self.with_pos_embed(q, q_pe)
        k = self.with_pos_embed(k, k_pe)
        
        high_fusion_factor = self.high_fusion_factor_token.expand(bs, -1, -1)
        q = torch.cat([high_fusion_factor, q], dim=1)  # [bs, 1 + hw, dim]
        k = torch.cat([high_fusion_factor, k], dim=1)  # [bs, 1 + HW, dim]
        v = torch.cat([high_fusion_factor, v], dim=1)  # [bs, 1 + HW, dim] 
        
        # Cross-attention
        residual = q
        out, _ = self.cross_attn(query=q, key=k, value=v)  # [bs, 1 + hw, dim]
        if not self.normalize_before:
            out = self.norm1(out)
        out = residual + self.dropout1(out)
        
        residual = out
        out = self.linear2(self.dropout_layer(self.activation(self.linear1(out))))
        if not self.normalize_before:
            out = self.norm2(out)
        out = residual + self.dropout2(out)
            
        # ## image-wise
        high_fusion_factor = self.high_fusion_factor_head(out[:, 0, :])
        high_fusion_factor = self.rescaled_tanh_0_2(high_fusion_factor)
        high_fusion_factor = high_fusion_factor.view(bs, -1, 1, 1)
        
        # 5. Reshape to spatial map
        out = out[:, 1:, :]  # [bs, hw, dim]
        a3_sa = out.permute(0, 2, 1).contiguous().view(bs, c_a3, h, w)
        
        # 6. Upsample
        scale_factor = 2 if idx == 2 else 4 # faster r-cnn, mask r-cnn
        # scale_factor = 2 if idx == 1 else 4 # retinanet, fcos
        a3_sa = F.interpolate(a3_sa, scale_factor=scale_factor, mode='bilinear', align_corners=False)

        a3_sa = a3_sa * original_a3

        # return a3_sa, low_fusion_factor, 2 - low_fusion_factor
        return a3_sa
        # return a3_sa


class SpatialAlignCrossAttentionReLULinear(nn.Module):
    def __init__(self, dim, dropout=0.0, activation='gelu', use_fusion_token=True):
        super().__init__()
        self.normalize_before = False
        self.dropout = dropout
        self.use_fusion_token = use_fusion_token
        self.heads = 8
        self.head_dim = dim // self.heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections to Q, K, V
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        self.out_proj = nn.Linear(dim, dim)

        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.dropout_layer = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()

        self.avg_pool_2x = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_4x = nn.AvgPool2d(kernel_size=4, stride=4)

    @staticmethod
    def get_sinusoidal_pos_embed_2d(h, w, dim, device):
        assert dim % 4 == 0, f"Channel dim ({dim}) must be divisible by 4 for 2D positional encoding"
        div_term = torch.exp(torch.arange(0, dim // 4, device=device) * (-torch.log(torch.tensor(10000.0)) / (dim // 4)))

        pos_h = torch.arange(h, device=device).reshape(h, 1)
        pe_h = torch.zeros(h, dim // 2, device=device)
        sin_h = torch.sin(pos_h * div_term)
        cos_h = torch.cos(pos_h * div_term)
        pe_h[:, 0::2] = sin_h
        pe_h[:, 1::2] = cos_h

        pos_w = torch.arange(w, device=device).reshape(w, 1)
        pe_w = torch.zeros(w, dim // 2, device=device)
        sin_w = torch.sin(pos_w * div_term)
        cos_w = torch.cos(pos_w * div_term)
        pe_w[:, 0::2] = sin_w
        pe_w[:, 1::2] = cos_w

        pe_h = pe_h.unsqueeze(1).expand(-1, w, -1)
        pe_w = pe_w.unsqueeze(0).expand(h, -1, -1)
        pe = torch.cat([pe_h, pe_w], dim=-1)
        return pe.reshape(1, h * w, dim)


    @staticmethod
    def get_sinusoidal_pos_embed_1d(seq_len, embed_dim, device):
        """
        Generate 1D sinusoidal positional embeddings.

        Args:
        - seq_len: Length of the sequence.
        - embed_dim: The dimension of the embedding.
        - device: The device to store the tensor.

        Returns:
        - A tensor of shape (1, seq_len, embed_dim).
        """
        # Sinusoidal position encoding (standard method)
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0., embed_dim, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        return pe.unsqueeze(0)  # Add batch dimension (1, seq_len, embed_dim)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed
    
    def rescaled_tanh_0_2(self, x):
        return torch.tanh(x) + 1  # [-1,1] → [0,2]

    def forward(self, a3, a4, idx):
        original_a3 = a3

        # Downsample features
        if idx == 2: # mask r-cnn
        # if idx == 1: # retinanet, fcos
            a3_down = self.avg_pool_2x(a3)
        else:
            a3_down = self.avg_pool_4x(a3)
            a4 = self.avg_pool_2x(a4)

        bs, c_a3, h, w = a3_down.shape
        _, c_a4, H, W = a4.shape

        # Flatten input for projection
        bs, c_a3, h, w = a3_down.shape
        _, c_a4, H, W = a4.shape
        a3_flat = a3_down.flatten(2).permute(0, 2, 1)  # [B, hw, C]
        a4_flat = a4.flatten(2).permute(0, 2, 1)        # [B, HW, C]
        
        residual = a3_flat 
        Q = self.q_proj(a3_flat)
        K = self.k_proj(a4_flat)
        V = self.v_proj(a4_flat)

        # Positional encoding
        q_pe = self.get_sinusoidal_pos_embed_1d(h * w, c_a3, a3_down.device)
        k_pe = self.get_sinusoidal_pos_embed_1d(H * W, c_a4, a4.device)
        Q = self.with_pos_embed(Q, q_pe)
        K = self.with_pos_embed(K, k_pe)
        
        # Reshape for multi-head attention
        Q = Q.reshape(bs, h * w, self.heads, self.head_dim).transpose(1, 2)
        K = K.reshape(bs, H * W, self.heads, self.head_dim).transpose(1, 2)
        V = V.reshape(bs, H * W, self.heads, self.head_dim).transpose(1, 2)

        Q = F.relu(Q)
        K = F.relu(K)
        
        KV = torch.einsum("bhkd,bhkv->bhdv", K, V)
        Z = 1 / (torch.einsum("bhqd,bhkd->bhq", Q, K.sum(dim=2, keepdim=True)) + 1e-6).unsqueeze(-1)
        attention = torch.einsum("bhqd,bhdv->bhqv", Q, KV) * Z

        out = attention.transpose(1, 2).contiguous().view(bs, -1, self.heads * self.head_dim)
        out = self.out_proj(out)
        
        src = residual + self.dropout_layer(out)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout_layer(self.activation(self.linear1(src))))
        src = residual + self.dropout_layer(src)
        if not self.normalize_before:
            src = self.norm2(src)
        
        # 5. Reshape to spatial map
        a3_sa = src.permute(0, 2, 1).contiguous().view(bs, c_a3, h, w)
        
        scale_factor = 2 if idx == 2 else 4 # faster r-cnn, mask r-cnn
        # scale_factor = 2 if idx == 1 else 4 # retinanet, fcos
        
        a3_sa = F.interpolate(a3_sa, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        a3_sa = a3_sa * original_a3
        return a3_sa



# This implementation is based on the original CosFormer paper, https://github.com/OpenNLPLab/cosFormer/blob/main/cosformer.py
import numpy as np
class SpatialAlignCrossAttentionCosFormer(nn.Module):
    def __init__(self, dim, dropout=0.0, activation='gelu', use_fusion_token=True):
        super().__init__()
        self.eps = 1e-6
        self.normalize_before = False
        self.dropout = dropout
        self.use_fusion_token = use_fusion_token
        self.heads = 8
        self.head_dim = dim // self.heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections to Q, K, V
        self.q_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        
        self.k_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # FFN
        self.linear1 = nn.Linear(dim, dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 4, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)
        
        self.avg_pool_2x = nn.AvgPool2d(kernel_size=2, stride=2)
        self.avg_pool_4x = nn.AvgPool2d(kernel_size=4, stride=4)

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
    def get_sinusoidal_pos_embed_1d(seq_len, embed_dim, device):
        """
        Generate 1D sinusoidal positional embeddings.

        Args:
        - seq_len: Length of the sequence.
        - embed_dim: The dimension of the embedding.
        - device: The device to store the tensor.

        Returns:
        - A tensor of shape (1, seq_len, embed_dim).
        """
        # Sinusoidal position encoding (standard method)
        position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0., embed_dim, 2, dtype=torch.float32, device=device) * -(math.log(10000.0) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        return pe.unsqueeze(0)  # Add batch dimension (1, seq_len, embed_dim)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed
    
    def rescaled_tanh_0_2(self, x):
        return torch.tanh(x) + 1  # [-1,1] → [0,2]

    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)
        return nn.Parameter(index, requires_grad=False)
    
    
    def forward(self, a3, a4, idx):
        original_a3 = a3

        # Downsample features
        if idx == 2: # mask r-cnn
        # if idx == 1: # retinanet, fcos
            a3_down = self.avg_pool_2x(a3)
        else:
            a3_down = self.avg_pool_4x(a3)
            a4 = self.avg_pool_2x(a4)

        bs, c_a3, h, w = a3_down.shape
        _, c_a4, H, W = a4.shape
        tgt_len = h * w
        src_len = H * W

        # Flatten input for projection
        bs, c_a3, h, w = a3_down.shape
        _, c_a4, H, W = a4.shape
        
        residual = a3_down.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        a3_flat = a3_down.flatten(2).permute(2, 0, 1) # [HW, B, C]
        a4_flat = a4.flatten(2).permute(2, 0, 1)       # [HW, B, C]
        
        # positional encoding
        pos = self.build_2d_sincos_position_embedding(w=h, h=w, embed_dim=c_a3).to(a3.device)  # [1, HW, C]
        pos = pos.squeeze(0).unsqueeze(1).expand(-1, bs, -1)  # [HW, B, C]
        a3_flat = self.with_pos_embed(a3_flat, pos)  # [HW, B, C]
        a4_flat = self.with_pos_embed(a4_flat, pos)  # [HW, B, C]
        
        # embedding
        Q = self.q_proj(a3_flat) # [HW, B, C]
        K = self.k_proj(a4_flat) # [HW, B, C]
        V = self.v_proj(a4_flat.clone()) # [HW, B, C]
        
        Q = F.relu(Q)
        K = F.relu(K)
        
        # multihead reshape
        # (B * #heads, HW, head_dim)
        Q = Q.contiguous().view(-1, bs * self.heads, self.head_dim).transpose(0, 1) # [B*#h, HW, head_dim]
        K = K.contiguous().view(-1, bs * self.heads, self.head_dim).transpose(0, 1) # [B*#h, HW, head_dim]
        V = V.contiguous().view(-1, bs * self.heads, self.head_dim).transpose(0, 1) # [B*#h, HW, head_dim]
        
        # cos transform
        m = max(src_len, tgt_len)
        weight_index = self.get_index(m).to(Q)
        # (N * h, L, 2 * d)
        Q_ = torch.cat([Q * torch.sin(weight_index[:, :tgt_len, :] / m), Q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        K_ = torch.cat([K * torch.sin(weight_index[:, :src_len, :] / m), K * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)
        
        # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
        KV_ = torch.einsum('nld,nlm->ndm', K_, V)
        # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
        Z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', Q_, torch.sum(K_, axis=1)), self.eps)
        
        # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
        attn_output = torch.einsum('nld,ndm,nl->nlm', Q_, KV_, Z_)
        # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bs, -1)
        out = self.out_proj(attn_output) # [HW, B, C]
        
        out = out.permute(1, 0, 2)  # [B, HW, C]
        
        # FFN
        if not self.normalize_before:
            out = self.norm1(out)
        out = residual + self.dropout1(out)
        residual = out
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        if not self.normalize_before:
            out = self.norm2(out)
        out = residual + self.dropout2(out) # [B, HW, C]
        
        
        # 5. Reshape to spatial map [B, HW, C] -> [B, C, H, W]
        a3_sa = out.permute(0, 2, 1).contiguous().view(bs, c_a3, h, w)  # [B, C, H, W]
        a3_sa = F.interpolate(a3_sa, size=original_a3.shape[2:], mode='bilinear', align_corners=False)  # [B, C, H, W]
        a3_sa = a3_sa * original_a3
        return a3_sa

class FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.

    The feature maps are currently supposed to be in increasing depth
    order.

    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.

    Args:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names
        norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None

    Examples::

        >>> m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 10, 64, 64)
        >>> x['feat2'] = torch.rand(1, 20, 16, 16)
        >>> x['feat3'] = torch.rand(1, 30, 8, 8)
        >>> # compute the FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]

    """

    _version = 2

    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = Conv2dNormActivation(
                in_channels, out_channels, kernel_size=1, padding=0, norm_layer=norm_layer, activation_layer=None
            )
            layer_block_module = Conv2dNormActivation(
                out_channels, out_channels, kernel_size=3, norm_layer=norm_layer, activation_layer=None
            )
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)
            
            
        
        # mask r-cnn is 4
        # retina net is 3
        
        # self.sa_cross_attn = nn.ModuleList()
        # print(f"len(in_channels_list): {len(in_channels_list)}") # 4  (mask r-cnn is 4)
        # for i in range(len(in_channels_list)-2): # 4
        # # for i in range(len(in_channels_list)-1): # 3
        #     self.sa_cross_attn.append(SpatialAlignCrossAttentionCosFormer(out_channels, activation='gelu'))

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if extra_blocks is not None:
            if not isinstance(extra_blocks, ExtraFPNBlock):
                raise TypeError(f"extra_blocks should be of type ExtraFPNBlock not {type(extra_blocks)}")
        self.extra_blocks = extra_blocks

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            num_blocks = len(self.inner_blocks)
            for block in ["inner_blocks", "layer_blocks"]:
                for i in range(num_blocks):
                    for type in ["weight", "bias"]:
                        old_key = f"{prefix}{block}.{i}.{type}"
                        new_key = f"{prefix}{block}.{i}.0.{type}"
                        if old_key in state_dict:
                            state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from the highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        # baseline
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # # print(f"len(x): {len(x)}") # 4
        # # mask-rcnn is 4, idx = 2, 1, 0
        # # retinanet, fcos is 3, idx = 1, 0
        # for idx in range(len(x) - 2, -1, -1): 
        #     inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
        #     feat_shape = inner_lateral.shape[-2:]
            
        #     # low: inter_lateral
        #     # high: last_inner
            
        #     # print(f"idx: {idx}")
        #     # print(f"low: {inner_lateral.shape}")
        #     # print(f"high: {last_inner.shape}")
            
        #     if idx == 2 or idx == 1: # mask r-cnn, faster r-cnn
        #     # if idx == 1 or idx == 0: # retinanet, fcos
        #         # inner_lateral, low_ff, high_ff = self.sa_cross_attn[len(x) - 2 - idx](inner_lateral, last_inner, idx)
        #         inner_lateral = self.sa_cross_attn[len(x) - 2 - idx](inner_lateral, last_inner, idx)
                
        #     inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="bilinear")
        #     last_inner = inner_lateral + inner_top_down
        #     results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))
            

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(ExtraFPNBlock):
    """
    Applies a max_pool2d on top of the last feature map
    """

    def forward(
        self,
        x: List[Tensor],
        y: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))
        return x, names


class LastLevelP6P7(ExtraFPNBlock):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            nn.init.kaiming_uniform_(module.weight, a=1)
            nn.init.constant_(module.bias, 0)
        self.use_P5 = in_channels == out_channels

    def forward(
        self,
        p: List[Tensor],
        c: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        p5, c5 = p[-1], c[-1]
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        p.extend([p6, p7])
        names.extend(["p6", "p7"])
        return p, names
