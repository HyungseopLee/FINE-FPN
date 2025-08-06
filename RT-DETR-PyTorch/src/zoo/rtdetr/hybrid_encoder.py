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
from .prior_works import AdaFPNBlock, MultiLevelGlobalContext


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


from einops import rearrange
import numpy as np
class SemanticAlignTransNormer(nn.Module):
    def __init__(self, dim, nhead, dropout=0.0, activation='gelu', is_first=False):
        '''
        c1: #channels of low-level feature map
        c2: #channels of high-level feature map
        '''
        
        super().__init__()
        self.eps = 1e-6
        
        self.nhead = nhead
        self.head_dim = dim // nhead
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # https://arxiv.org/pdf/2210.10340
        self.attn_norm = nn.RMSNorm(self.head_dim)
        
        # FFN
        self.linear1 = nn.Linear(dim, dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.activation = get_activation(activation)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

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
    
    def forward(self, a3, a4):
        '''
        a3 is low-level feature
        a4 is high-level feature
        '''
        original_a3 = a3
        
        a3 = self.avg_pool_low(a3)
        a4 = self.avg_pool_high(a4)
        
        bs, c_a3, h, w = a3.shape
        _, c_a4, H, W = a4.shape
        tgt_len = h * w
        src_len = H * W
        
        residual = rearrange(a3, 'b c h w -> b (h w) c')
        a3_flat = rearrange(a3, 'b c h w -> (h w) b c')
        a4_flat = rearrange(a4, 'b c h w -> (h w) b c')
        
        # positional encoding
        pos = self.build_2d_sincos_position_embedding(w=w, h=h, embed_dim=c_a3).to(a3.device)
        pos = pos.expand(bs, -1, -1).permute(1, 0, 2)  # [HW, B, C]
        a3_flat = self.with_pos_embed(a3_flat, pos)  # [HW, B, C]
        a4_flat = self.with_pos_embed(a4_flat, pos)  # [HW, B, C]
        
        # embedding
        a3_flat = a3_flat.to(self.q_proj.weight.dtype)
        a4_flat = a4_flat.to(self.k_proj.weight.dtype)
        Q = self.q_proj(a3_flat)
        K = self.k_proj(a4_flat)
        V = self.v_proj(a4_flat) # [HW, B, C]
        
        # Memory-efficient multi-head reshape using einops
        Q = rearrange(Q, 'hw b (h d) -> (b h) hw d', h=self.nhead)
        K = rearrange(K, 'hw b (h d) -> (b h) hw d', h=self.nhead)
        V = rearrange(V, 'hw b (h d) -> (b h) hw d', h=self.nhead)
        
        # SIMA: n1-norm kernel function: QK normalization for stability
        # https://github.com/UCDvision/sima/blob/main/sima.py
        Q = F.normalize(Q, p=1, dim=-2, eps=self.eps) 
        K = F.normalize(K, p=1, dim=-2, eps=self.eps) 
        
        # CosFormer transform
        m = max(src_len, tgt_len)
        weight_index = self.get_index(m).to(Q)
        # (N * h, L, 2 * d)
        Q_ = torch.cat([
            Q * torch.sin(weight_index[:, :tgt_len, :] / m), 
            Q * torch.cos(weight_index[:, :tgt_len, :] / m)
        ], dim=-1)
        K_ = torch.cat([
            K * torch.sin(weight_index[:, :src_len, :] / m), 
            K * torch.cos(weight_index[:, :src_len, :] / m)
        ], dim=-1)

        # NormAttention
        KV_ = torch.einsum('nld,nlm->ndm', K_, V)
        attn_output = torch.einsum('nld,ndm->nlm', Q_, KV_)  # [N*h, L, d]
        attn_output = self.attn_norm(attn_output)

        # [N*h, L, d] ->  [L, N*h, d] -> [L, B, C]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bs, -1) # [L, B, C]
        out = self.out_proj(attn_output) #  [L, B, C]
        out = out.permute(1, 0, 2)  # [B, L, C]
        
        # FFN
        out = self.norm1(out)
        out = residual + self.dropout1(out)
        residual = out
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = self.norm2(out)
        out = residual + self.dropout2(out) # [B, HW, C]
        
        # 5. Reshape to spatial map [B, HW, C] -> [B, C, H, W]
        a3_sa = out.permute(0, 2, 1).contiguous().view(bs, c_a3, h, w)  # [B, C, H, W]
        a3_sa = F.interpolate(a3_sa, size=original_a3.shape[2:], mode='bilinear', align_corners=False)
        a3_sa = a3_sa * original_a3
        
        return a3_sa


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
        
        # # channel projection
        # self.input_proj = nn.ModuleList()
        # for in_channel in in_channels:
        #     if version == 'v1':
        #         proj = nn.Sequential(
        #             nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
        #             nn.BatchNorm2d(hidden_dim))
        #     elif version == 'v2':
        #         proj = nn.Sequential(OrderedDict([
        #             ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
        #             ('norm', nn.BatchNorm2d(hidden_dim))
        #         ]))
        #     else:
        #         raise AttributeError()
                
        #     self.input_proj.append(proj)
            
        
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

        # # 2025.07.18 @HyungseopLee 
        # # AdaFPN https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9497077
        # self.ada_fpn_blocks = nn.ModuleList()
        # for i in range(len(self.in_channels) - 1):
        #     high_ch = self.hidden_dim
        #     low_ch = self.hidden_dim   
        #     out_ch = self.hidden_dim 
        #     self.ada_fpn_blocks.append(AdaFPNBlock(high_ch, low_ch, out_ch))
        
        # MGC
        self.mgc = MultiLevelGlobalContext(in_channels, hidden_dim)

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
        # for i in range(len(in_channels) - 1):
        #     is_first=False
        #     if i == 0:
        #         is_first = True
                
        #     self.sa_cross_attn.append(
        #         # SpatialAlignCrossAttentionReLULinear(dim=hidden_dim, dropout=dropout, activation=enc_act)
        #         # SpatialAlignCrossAttentionCosFormer(dim=hidden_dim, dropout=dropout, activation=enc_act, is_first=is_first)
        #         SemanticAlignTransNormer(dim=hidden_dim, nhead=16, dropout=0.0, activation=enc_act, is_first=is_first)
        #     )

        # self.bu_sa_cross_attn = nn.ModuleList()
        # for i in range(len(in_channels) - 1):
        #     is_first=False
        #     if i == 0:
        #         is_first = True
                
        #     self.bu_sa_cross_attn.append(
        #         BUSpatialAlignTransNormer(dim=hidden_dim, nhead=16, dropout=dropout, activation=enc_act, is_first=is_first)
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
        
        # Apply MGC module before channel projection
        proj_feats = self.mgc(feats)
        
        # for i, feat in enumerate(proj_feats):
        #     print(f'feat {i} shape: {feat.shape}')
        
        # # baseline
        # proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        
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
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            
            # # # exp 2: Deconv
            # upsample_feat = self.upsample_convs[len(self.in_channels) - 1 - idx](feat_high)
            
            # exp3: FaPN (align high to low)
            # upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            # offset = self.offset[len(self.in_channels)-1-idx](torch.cat([feat_low, upsample_feat * 2], dim=1))
            # feat_align = self.feature_align[len(self.in_channels)-1-idx]((upsample_feat, offset))
            
            # # exp4: AdaFPN 
            # inner_out = self.ada_fpn_blocks[len(self.in_channels)-1-idx](feat_high, feat_low)
            # inner_outs.insert(0, inner_out)
            
            # # Ours (align low to high)
            # feat_low = self.sa_cross_attn[len(self.in_channels) - 1 - idx](feat_low, feat_high) 
            # upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='bilinear')
                
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1): # 0, 1, 2
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)

        return outs