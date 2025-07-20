"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import copy
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np
from .utils import get_activation

from ...core import register
from typing import Optional
from torch import Tensor

__all__ = ['HybridEncoder']



class DepthwiseConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            groups=ch_in,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, output_padding=None, bias=False, act='silu'):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            output_padding=output_padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class TConvFromDownsample(nn.Module):
    def __init__(self, ch_in=256, ch_out=256, act='relu'):
        super().__init__()
        self.ch_in = ch_in  
        self.ch_out = ch_out  
        self.norm_1 = nn.BatchNorm2d(ch_in)  
        # self.norm_2 = nn.BatchNorm2d(ch_out)  
        self.act = nn.Identity() if act is None else get_activation(act)

    def adjust_channels(self, weight, in_channels, out_channels):
            """
            backbone weight -> encoder weight
            
            ex1:
                backbone (c_in=256, c_out=512) -> neck (c_in=256, c_out=256)
                slice weight[:, :256, :, :]
                
            ex2:
                backbone (c_in=128, c_out=256) -> neck (c_in=256, c_out=256)
                 weight[:, :128, :, :] ->  repeat*2 -> weight[:, :256, :, :]
            """
            
            curr_out_channels, curr_in_channels = weight.shape[0], weight.shape[1]

            if curr_in_channels > in_channels:
                weight = weight[:, :in_channels, :, :]
            elif curr_in_channels < in_channels:
                repeat_factor = in_channels // curr_in_channels
                remainder = in_channels % curr_in_channels
                if repeat_factor > 1:
                    weight = weight.repeat(1, repeat_factor, 1, 1)
                if remainder > 0:
                    weight = torch.cat([weight, weight[:, :remainder, :, :]], dim=1)

            if curr_out_channels > out_channels:
                weight = weight[:out_channels, :, :, :]
            elif curr_out_channels < out_channels:
                repeat_factor = out_channels // curr_out_channels
                remainder = out_channels % curr_out_channels
                if repeat_factor > 1:
                    weight = weight.repeat(repeat_factor, 1, 1, 1)
                if remainder > 0:
                    extra = weight[:remainder, :, :, :]
                    weight = torch.cat([weight, extra], dim=0)

            return weight

    # # exp1
    # def forward(self, x, upsample_params):
    #     '''
    #     Transposed Conv로 업샘플링을 수행하며, backbone의 다운샘플링 파라미터를 재사용.
    #     모든 레이어의 입출력 채널은 256으로 고정.

    #     x: feat_high (입력 텐서)
    #     upsample_params: backbone에서 사용된 다운샘플링 파라미터
    #         (branch2a): c -> 2c
    #         (branch2b): 2c -> 2c
    #     '''
    #     # branch2a Transposed Conv
    #     branch2a_conv = upsample_params.branch2a.conv
    #     branch2a_tconv = nn.ConvTranspose2d(
    #         in_channels=self.ch_in,  # 인코더에서는 256
    #         out_channels=self.ch_out,  # 인코더에서는 256
    #         kernel_size=branch2a_conv.kernel_size,
    #         stride=2,  # 업샘플링
    #         padding=branch2a_conv.padding,
    #         output_padding=1,  # stride=2에 맞게
    #         bias=branch2a_conv.bias is not None
    #     )

    #     # weight 조정 (backbone의 weight를 인코더 채널에 맞게 변환)
    #     weight = branch2a_conv.weight
    #     weight = weight.permute(1, 0, 2, 3)  # Transpose: in_channels <-> out_channels
    #     weight = self.adjust_channels(weight, self.ch_in, self.ch_out)
    #     branch2a_tconv.weight = nn.Parameter(weight)

    #     # branch2b Transposed Conv
    #     branch2b_conv = upsample_params.branch2b.conv
    #     branch2b_tconv = nn.ConvTranspose2d(
    #         in_channels=self.ch_in,  # 인코더에서는 256
    #         out_channels=self.ch_out,  # 인코더에서는 256
    #         kernel_size=branch2b_conv.kernel_size,
    #         stride=1,  # 공간적 업샘플링 없음
    #         padding=branch2b_conv.padding,
    #         bias=branch2b_conv.bias is not None
    #     )

    #     # weight 조정
    #     weight = branch2b_conv.weight
    #     weight = weight.permute(1, 0, 2, 3)  # Transpose: in_channels <-> out_channels
    #     weight = self.adjust_channels(weight, self.ch_in, self.ch_out)
    #     branch2b_tconv.weight = nn.Parameter(weight)

    #     # print(f"x: {x.shape}")
    #     x = branch2a_tconv(x)
    #     # print(f"(branch2a) x: {x.shape}")
    #     x = self.act(self.norm_1(x))
    #     x = branch2b_tconv(x)
    #     # print(f"(branch2b) x: {x.shape}")
    #     x = self.act(self.norm_2(x))
        
    #     return x



    # exp2
    def forward(self, x, upsample_params):
        '''
        Transposed Conv로 업샘플링을 수행하며, backbone의 다운샘플링 파라미터를 재사용.
        모든 레이어의 입출력 채널은 256으로 고정.

        x: feat_high (입력 텐서)
        upsample_params: backbone에서 사용된 다운샘플링 파라미터
            (branch2a): c -> 2c
            (branch2b): 2c -> 2c
        '''
        # branch2a Transposed Conv
        branch2a_conv = upsample_params.branch2a.conv
        branch2a_tconv = nn.ConvTranspose2d(
            in_channels=self.ch_in,  # 인코더에서는 256
            out_channels=self.ch_out,  # 인코더에서는 256
            kernel_size=branch2a_conv.kernel_size,
            stride=2,  # 업샘플링
            padding=branch2a_conv.padding,
            output_padding=1,  # stride=2에 맞게
            bias=branch2a_conv.bias is not None
        )

        # weight 조정 (backbone의 weight를 인코더 채널에 맞게 변환)
        weight = branch2a_conv.weight
        weight = weight.permute(1, 0, 2, 3)  # Transpose: in_channels <-> out_channels
        weight = self.adjust_channels(weight, self.ch_in, self.ch_out)
        branch2a_tconv.weight = nn.Parameter(weight)

        # print(f"x: {x.shape}")
        x = branch2a_tconv(x)
        # print(f"(branch2a) x: {x.shape}")
        x = self.act(self.norm_1(x))

        
        return x



class SpatialAlignCrossAttention(nn.Module):
    def __init__(self, dim, dropout=0.0, activation='gelu', use_fusion_token=True, head_reduction=4):
        super().__init__()

        self.normalize_before = False
        self.dropout = dropout
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
    
    def rescaled_sigmoid_0_2(self, x):
        x = torch.sigmoid(x) # [0, 1]
        return 2 * x # Rescale to [0, 2]

    def forward(self, a3_down, a4, idx):
        original_a3 = a3_down

        # # Downsample features
        # if idx == 2:
        #     a3_down = self.avg_pool_2x(a3)
        # else:
        #     a3_down = self.avg_pool_4x(a3)
        #     a4 = self.avg_pool_2x(a4)
            
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
            
        # 5. Reshape to spatial map
        a3_sa = out.permute(0, 2, 1).contiguous().view(bs, c_a3, h, w)
        
        # 6. Upsample
        # scale_factor = 2 if idx == 2 else 4
        # a3_sa = F.interpolate(a3_sa, scale_factor=scale_factor, mode='bilinear', align_corners=False)

        a3_sa = a3_sa * original_a3

        return a3_sa



class SpatialAlignCrossAttentionReLULinear(nn.Module):
    def __init__(self, c1, c2, nhead, is_yolov6=False, dropout=0.0, activation="gelu"):
        super().__init__()
        self.eps = 1e-5
        self.normalize_before = False
        self.dropout = dropout
        self.heads = nhead
        self.head_dim = c1 // self.heads
        self.scale = self.head_dim ** -0.5
        
        self.is_yolov6 = is_yolov6
        # bilinear interpolation upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        if self.is_yolov6:
            self.upsample = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2)
        
        # others: low=c1 < high=c2. high=c2 -> c1
        # yolov6: low=c1 > high=c2. low=c1 -> c2
        self.high_to_low_embed = nn.Sequential(
            nn.Conv2d(c2, c1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c1),
        )
        
        # Linear projections to Q, K, V
        self.q_proj = nn.Sequential(
            nn.Linear(c1, c1),
            nn.LayerNorm(c1)
        )
        self.k_proj = nn.Sequential(
            nn.Linear(c1, c1),
            nn.LayerNorm(c1)
        )
        self.v_proj = nn.Linear(c1, c1)
        self.out_proj = nn.Linear(c1, c1)

        self.linear1 = nn.Linear(c1, c1 * 4)
        self.dropout_layer = nn.Dropout(dropout)
        self.linear2 = nn.Linear(c1 * 4, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
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
    
    
    def forward(self, a3, a4, idx):
        original_a4 = a4
        original_a3 = a3

        a4 = self.high_to_low_embed(a4)
        
        # Downsample features
        # v8: 10, 
        # v5, v10: 11
        # v6: 12
        if idx == 11:
            a3_down = self.avg_pool_2x(a3)
        else:
            a3_down = self.avg_pool_4x(a3)
            a4 = self.avg_pool_2x(a4)
            
        bs, c_a3, h, w = a3_down.shape
        _, c_a4, H, W = a4.shape
            
        # Flatten input for projection
        residual = a3_down.flatten(2).permute(0, 2, 1)  # [B, hw, C]
        a3_flat = a3_down.flatten(2).permute(0, 2, 1)  # [B, hw, C]
        a4_flat = a4.flatten(2).permute(0, 2, 1)        # [B, HW, C]
        
        # # Positional encoding
        # pos = self.get_sinusoidal_pos_embed_1d(h * w, c_a3, a3_down.device)
        # a3_flat = self.with_pos_embed(a3_flat, pos)
        # a4_flat = self.with_pos_embed(a4_flat, pos)
        
        a3_flat = a3_flat.to(self.q_proj[0].weight.dtype)
        a4_flat = a4_flat.to(self.k_proj[0].weight.dtype)
        Q = self.q_proj(a3_flat) # [B, hw, C]
        K = self.k_proj(a4_flat)
        V = self.v_proj(a4_flat.clone())

        # Attention
        ## Reshape for multi-head attention
        Q = Q.view(bs, h * w, self.heads, self.head_dim).permute(0, 2, 3, 1)  # [bs, heads, head_dim, hw]
        K = K.view(bs, H * W, self.heads, self.head_dim).permute(0, 2, 3, 1)  # [bs, heads, head_dim, HW]
        V = V.view(bs, H * W, self.heads, self.head_dim).permute(0, 2, 3, 1)  # [bs, heads, head_dim, HW]

        Q = F.relu(Q)
        K = F.relu(K)

        # Attention computation
        trans_K = K.transpose(-1, -2)  # [bs, heads, HW, head_dim]

        # Pad V for linear attention
        V = F.pad(V, (0, 0, 0, 1), mode="constant", value=1)  # [bs, heads, head_dim+1, HW]
        VK = torch.matmul(V, trans_K)                         # [bs, heads, head_dim+1, head_dim]
        out = torch.matmul(VK, Q)                             # [bs, heads, head_dim+1, hw]

        if out.dtype == torch.bfloat16:
            out = out.float()
        denom = out[:, :, -1:].clamp(min=self.eps)
        out = out[:, :, :-1] / denom  # [bs, heads, head_dim, hw]

        out = out.transpose(1, 2).contiguous().view(bs, -1, self.heads * self.head_dim) # [bs, hw, c_a3]
        out = self.out_proj(out)

        src = residual + self.dropout_layer(out)
        if not self.normalize_before:
            src = self.norm1(src)

        ## FFN
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout_layer(self.activation(self.linear1(src))))
        src = residual + self.dropout_layer(src)
        if not self.normalize_before:
            src = self.norm2(src)
        
        # 5. Reshape to spatial map [B, HW, C] -> [B, C, H, W]
        a3_sa = out.permute(0, 2, 1).contiguous().view(bs, c_a3, h, w)  # [B, C, H, W]
        a3_sa = F.interpolate(a3_sa, size=original_a3.shape[2:], mode='bilinear', align_corners=False)
        a3_sa = a3_sa * original_a3
        
        if self.is_yolov6 :
            return a3_sa, self.upsample(original_a4)  # Return both the aligned feature and the upsampled feature
        else: 
            return a3_sa, F.interpolate(original_a4, scale_factor=2., mode='bilinear')
    
    


# This implementation is based on the original CosFormer paper, https://github.com/OpenNLPLab/cosFormer/blob/main/cosformer.py
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
        # self.q_proj = nn.Linear(dim, dim)
        # self.k_proj = nn.Linear(dim, dim)
        
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
    
    
    # ours
    def forward(self, a3, a4, idx):
        original_a3 = a3

        if idx == 2:
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
        
        # 2D positional encoding
        pos = self.build_2d_sincos_position_embedding(w=h, h=w, embed_dim=c_a3).to(a3_down.device)  # [1, HW, C]
        pos = pos.squeeze(0).unsqueeze(1).expand(-1, bs, -1)
        a3_flat = self.with_pos_embed(a3_flat, pos)  # [HW, B, C]
        a4_flat = self.with_pos_embed(a4_flat, pos)  # [HW, B, C]
        
        
        # # 1D positional encoding
        # pos = self.get_sinusoidal_pos_embed_1d(h * w, c_a3, a3.device)  # [1, HW, C]
        # pos = pos.squeeze(0).unsqueeze(1).expand(-1, bs, -1)  # [HW, B, C]
        # a3_flat = self.with_pos_embed(a3_flat, pos)  # [HW, B, C]
        # a4_flat = self.with_pos_embed(a4_flat, pos)  # [HW, B, C]
        
        # embedding
        Q = self.q_proj(a3_flat) # [HW, B, C]
        K = self.k_proj(a4_flat) # [HW, B, C]
        V = self.v_proj(a4_flat.clone()) # [HW, B, C]
        
        Q = F.relu(Q)  # [HW, B, C]
        K = F.relu(K)  # [HW, B, C]
        
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
        scale_factor = 2 if idx == 2 else 4
        a3_sa = F.interpolate(a3_sa, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        a3_sa = a3_sa * original_a3
        return a3_sa


    # # Linear, but no resolution-consistent design
    # def forward(self, a3, a4, idx):
    #     original_a3 = a3


    #     bs, c_a3, h, w = a3.shape
    #     _, c_a4, H, W = a4.shape
    #     tgt_len = h * w
    #     src_len = H * W

    #     # Flatten input for projection
    #     bs, c_a3, h, w = a3.shape
    #     _, c_a4, H, W = a4.shape
        
    #     residual = a3.flatten(2).permute(0, 2, 1)  # [B, HW, C]
    #     a3_flat = a3.flatten(2).permute(2, 0, 1) # [HW, B, C]
    #     a4_flat = a4.flatten(2).permute(2, 0, 1)       # [HW, B, C]
        
    #     # positional encoding
    
    #     pos_a3 = self.build_2d_sincos_position_embedding(h, w, c_a3, a3.device)  # [1, HW, C]
    #     pos_a4 = self.build_2d_sincos_position_embedding(H, W, c_a4, a4.device)
    #     pos_a3 = pos_a3.squeeze(0).unsqueeze(1).expand(-1, bs, -1)
    #     pos_a4 = pos_a4.squeeze(0).unsqueeze(1).expand(-1, bs, -1)
    #     a3_flat = self.with_pos_embed(a3_flat, pos_a3)  # [HW, B, C]
    #     a4_flat = self.with_pos_embed(a4_flat, pos_a4)  # [HW, B, C]
        
    #     # embedding
    #     Q = self.q_proj(a3_flat) # [HW, B, C]
    #     K = self.k_proj(a4_flat) # [HW, B, C]
    #     V = self.v_proj(a4_flat.clone()) # [HW, B, C]
        
    #     Q = F.relu(Q)  # [HW, B, C]
    #     K = F.relu(K)  # [HW, B, C]
        
    #     # multihead reshape
    #     # (B * #heads, HW, head_dim)
    #     Q = Q.contiguous().view(-1, bs * self.heads, self.head_dim).transpose(0, 1) # [B*#h, HW, head_dim]
    #     K = K.contiguous().view(-1, bs * self.heads, self.head_dim).transpose(0, 1) # [B*#h, HW, head_dim]
    #     V = V.contiguous().view(-1, bs * self.heads, self.head_dim).transpose(0, 1) # [B*#h, HW, head_dim]
        
    #     # cos transform
    #     m = max(src_len, tgt_len)
    #     weight_index = self.get_index(m).to(Q)
    #     # (N * h, L, 2 * d)
    #     Q_ = torch.cat([Q * torch.sin(weight_index[:, :tgt_len, :] / m), Q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
    #     # (N * h, S, 2 * d)
    #     K_ = torch.cat([K * torch.sin(weight_index[:, :src_len, :] / m), K * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)
        
    #     # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, 2 * d, d)
    #     KV_ = torch.einsum('nld,nlm->ndm', K_, V)
    #     # (N * h, L, 2 * d) (N * h, 2 * d) -> (N * h, L)
    #     Z_ = 1 / torch.clamp_min(torch.einsum('nld,nd->nl', Q_, torch.sum(K_, axis=1)), self.eps)
    #     # (N * h, L, 2 * d) (N * h, d, 2 * d) (N * h, L) -> (N * h, L, d)
    #     attn_output = torch.einsum('nld,ndm,nl->nlm', Q_, KV_, Z_)
    #     # (N * h, L, d) -> (L, N * h, d) -> (L, N, E)
    #     attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bs, -1)
    #     out = self.out_proj(attn_output) # [HW, B, C]
        
    #     out = out.permute(1, 0, 2)  # [B, HW, C]
        
    #     # FFN
    #     if not self.normalize_before:
    #         out = self.norm1(out)
    #     out = residual + self.dropout1(out)
    #     residual = out
    #     out = self.linear2(self.dropout(self.activation(self.linear1(out))))
    #     if not self.normalize_before:
    #         out = self.norm2(out)
    #     out = residual + self.dropout2(out) # [B, HW, C]
        
        
    #     # 5. Reshape to spatial map [B, HW, C] -> [B, C, H, W]
    #     a3_sa = out.permute(0, 2, 1).contiguous().view(bs, c_a3, h, w)  # [B, C, H, W]
    #     a3_sa = a3_sa * original_a3
    #     return a3_sa



import torchvision.ops
import math

def _pair(x):
    """Convert scalar or tuple to tuple of length 2."""
    if isinstance(x, (int, float)):
        return (x, x)
    return x

class DCNv2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        deformable_groups=1,
    ):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups

        # Deformable convolution using torchvision
        self.deform_conv = torchvision.ops.DeformConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=deformable_groups,
            bias=True
        )

        self.relu = nn.ReLU(inplace=True)
        
        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        stdv = 1.0 / math.sqrt(n)
        self.deform_conv.weight.data.uniform_(-stdv, stdv)
        if self.deform_conv.bias is not None:
            self.deform_conv.bias.data.zero_()

    def forward(self, input, offset, mask):
        """
        Args:
            input: Input tensor [bs, in_channels, H, W]
            offset: Offset tensor [bs, 2 * deformable_groups * kh * kw, H, W]
            mask: Modulation mask [bs, deformable_groups * kh * kw, H, W]
        
        Returns:
            Output tensor [bs, out_channels, H_out, W_out]
        """
        # torchvision DeformConv2d expects mask as modulation scalar
        # Multiply mask with offset to match DCNv2 behavior
        assert offset.shape[1] == 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
        assert mask.shape[1] == self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
        
        return self.relu(self.deform_conv(input, offset, mask))

class DCN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        deformable_groups=1,
        extra_offset_mask=False,
    ):
        super(DCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = deformable_groups
        self.extra_offset_mask = extra_offset_mask

        # Deformable convolution
        self.deform_conv = DCNv2(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            deformable_groups=self.deformable_groups
        )

        # Offset and mask prediction
        channels_ = self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(self.in_channels, channels_, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True)
        self.init_offset()
        
    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input, main_path=None):
        """
        Args:
            input: Input tensor [bs, in_channels, H, W] or list of tensors if extra_offset_mask=True
            main_path: Unused (kept for compatibility)
        
        Returns:
            Output tensor [bs, out_channels, H_out, W_out]
        """
        if self.extra_offset_mask:
            assert isinstance(input, (list, tuple)) and len(input) == 2, "Expected two inputs for extra_offset_mask"
            feature_input, offset_input = input 
            out = self.conv_offset_mask(offset_input)
        else:
            feature_input = input
            out = self.conv_offset_mask(input)

        # Split into offsets and mask
        o1, o2, mask = torch.chunk(out, 3, dim=1)  # Each: [bs, deformable_groups * kh * kw, H, W]
        offset = torch.cat((o1, o2), dim=1)        # [bs, 2 * deformable_groups * kh * kw, H, W]
        mask = torch.sigmoid(mask)                  # [bs, deformable_groups * kh * kw, H, W]

        return self.deform_conv(feature_input, offset, mask)
   

# This code is part of https://github.com/EMI-Group/FaPN/blob/main/detectron2/modeling/backbone/fan.py   
class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = nn.Conv2d(in_chan, in_chan, kernel_size=1, bias=False)
        self.norm_atten = nn.GroupNorm(32, in_chan) if norm == "GN" else norm
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
        nn.init.xavier_uniform_(self.conv.weight, gain=1)
        nn.init.xavier_uniform_(self.conv_atten.weight, gain=1)

    def forward(self, x):
        atten = self.sigmoid(self.norm_atten(self.conv_atten(F.avg_pool2d(x, x.size()[2:]))))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat
    
            
class FaPN(nn.Module):
    def __init__(self, c1, c2, norm="GN"):
        '''
        c1: low level feature channels
        c2: high level feature channels
        '''
        super(FaPN, self).__init__()
        self.fsm = FeatureSelectionModule(c1, c1, norm)
        self.offset  = nn.Sequential(
            nn.Conv2d(c1 + c2, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.feature_align = DCN(
            in_channels=c2,
            out_channels=c2,
            kernel_size=3,
            stride=1,
            padding=1,
            deformable_groups=8,
            extra_offset_mask=True
        )
        
    def forward(self, feat_high, feat_low):
        
        feat_low_fsm = self.fsm(feat_low)
        upsample_feat_high = F.interpolate(feat_high, scale_factor=2., mode='nearest')
        offset = self.offset(torch.cat([feat_low_fsm, upsample_feat_high * 2], dim=1))
        feat_high_align = F.relu(self.feature_align([upsample_feat_high, offset]))
        
        out = torch.cat([feat_high_align, feat_low_fsm], dim=1)
        
        return out
        
  
class DWConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DWConv, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None, 
                 version='v2'):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size        
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if version == 'v1':
                proj = nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim))
            elif version == 'v2':
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                raise AttributeError()
                
            self.input_proj.append(proj)
            
        
        # # 2025.04.16 @HyungseopLee: FaPN
        # self.feature_selection = nn.ModuleList()
        # for in_channel in in_channels:
        #     feature_select = FeatureSelectionModule(in_channel, hidden_dim)
        #     self.feature_selection.append(feature_select)
            
        # self.feature_align = nn.ModuleList()
        # for i in range(len(in_channels) - 1):
        #     feature_align = DCN(
        #         in_channels=hidden_dim,
        #         out_channels=hidden_dim,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         deformable_groups=8,
        #         extra_offset_mask=True,
        #     )
        #     self.feature_align.append(feature_align)
        
        # self.offset = nn.ModuleList()
        # for i in range(len(in_channels) - 1):
        #     offset = nn.Sequential(
        #         nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(hidden_dim),
        #     )
        #     self.offset.append(offset)

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # # Transposed Conv for upsampling
        # self.upsample_convs = nn.ModuleList()
        # for _ in range(len(in_channels) - 1):
        #     self.upsample_convs.append(
        #         TConvNormLayer(hidden_dim, hidden_dim, 3, 2, 1, 1, act='silu')
        #     )
        
        # self.upsample_second_convs = nn.ModuleList()
        # for _ in range(len(in_channels) - 1):
        #     self.upsample_second_convs.append(
        #         ConvNormLayer(hidden_dim, hidden_dim, 3, 1, act=act)
        #     )
            
        
        
        # self.sa_cross_attn = nn.ModuleList()
        # for _ in range(len(in_channels) - 1):
        #     self.sa_cross_attn.append(
        #         # SpatialAlignCrossAttentionReLULinear(dim=hidden_dim, dropout=dropout, activation=enc_act)
        #         SpatialAlignCrossAttentionCosFormer(dim=hidden_dim, dropout=dropout, activation=enc_act)
        #     )
            
        # self.bu_sa_cross_attn = nn.ModuleList()
        # for _ in range(len(in_channels) - 1):
        #     self.bu_sa_cross_attn.append(
        #         BottomUpSpatialAlignCrossAttention(dim=hidden_dim, dropout=dropout, activation=enc_act)
        #     )
        

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        """
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

    # def forward(self, feats, down_s2_s3, down_s3_s4):
    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        
        # baseline
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
        # # FaPN
        # proj_feats = [self.feature_selection[i](feat) for i, feat in enumerate(feats)]
    
        
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory :torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            
            # exp1: baseline
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='bilinear')
            
            # # # exp 2: Transposed Conv for upsampling
            # upsample_feat = self.upsample_convs[len(self.in_channels) - 1 - idx](feat_high)
            
            # # 2025.04.16 @HyungseopLee
            # # FaPN (align high to low)
            # upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            # offset = self.offset[len(self.in_channels)-1-idx](torch.cat([feat_low, upsample_feat * 2], dim=1))
            # feat_align = self.feature_align[len(self.in_channels)-1-idx]((upsample_feat, offset))
            
            # # Ours (align low to high)
            # feat_low = self.sa_cross_attn[len(self.in_channels) - 1 - idx](feat_low, feat_high, idx=idx) 
            # upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='bilinear')
                
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1): # 0, 1, 2
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            
            # # 1. align low(downsample_feat) to high(feat_high)
            # downsample_feat = self.bu_sa_cross_attn[idx](downsample_feat, feat_high, idx)
            
            # # 2. align high to low
            # feat_high = self.bu_sa_cross_attn[idx](feat_high, downsample_feat, idx)
            
            
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        return outs
        # return outs