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

class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x



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
        
        

class AdaUp(nn.Module):
    """
    Adaptive Upsampling module that predicts content-aware offsets for sampling points
    instead of using fixed bilinear interpolation.
    """
    def __init__(self, in_channels, out_channels, scale_factor=2, num_sampling_points=4):
        super(AdaUp, self).__init__()
        self.scale_factor = scale_factor
        self.num_sampling_points = num_sampling_points
        
        # Offset prediction network
        self.offset_conv1 = nn.Conv2d(in_channels + out_channels, 
                                     in_channels, kernel_size=1, padding=0)
        self.offset_conv2 = nn.Conv2d(in_channels, 
                                     num_sampling_points * 2, kernel_size=3, padding=1)
        
        # Feature processing
        self.feature_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Initialize offset prediction to zero
        nn.init.zeros_(self.offset_conv2.weight)
        nn.init.zeros_(self.offset_conv2.bias)
        
    def forward(self, high_feat, low_feat):
        B, C_high, H, W = high_feat.shape
        _, C_low, H_low, W_low = low_feat.shape
        
        # Initial upsampling using bilinear interpolation
        upsampled_high = F.interpolate(high_feat, scale_factor=self.scale_factor, 
                                      mode='bilinear', align_corners=False)
        
        print(f"High feature shape: {upsampled_high.shape}, Low feature shape: {low_feat.shape}")
        
        # Concatenate features for offset prediction
        concat_feat = torch.cat([upsampled_high, low_feat], dim=1)
        
        # Predict offsets
        offset_feat = self.offset_conv1(concat_feat)
        offset_feat = F.relu(offset_feat)
        offsets = self.offset_conv2(offset_feat)
        
        # Process high-level features
        processed_high = self.feature_conv(high_feat)
        
        # Apply adaptive sampling
        upsampled_feat = self.adaptive_sampling(processed_high, offsets)
        
        return upsampled_feat
    
    def adaptive_sampling(self, feat, offsets):
        B, C, H, W = feat.shape
        H_up, W_up = H * self.scale_factor, W * self.scale_factor
        
        # Create base grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H_up, device=feat.device),
            torch.linspace(-1, 1, W_up, device=feat.device),
            indexing='ij'
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        
        # Reshape and normalize offsets
        offsets = offsets.view(B, self.num_sampling_points, 2, H_up, W_up)
        offsets = offsets.permute(0, 3, 4, 1, 2)
        offsets = offsets / torch.tensor([W_up, H_up], device=feat.device).view(1, 1, 1, 1, 2) * 2
        
        # Sample features at offset positions
        sampled_feats = []
        for i in range(self.num_sampling_points):
            offset_grid = base_grid + offsets[:, :, :, i, :]
            sampled_feat = F.grid_sample(feat, offset_grid, mode='bilinear', 
                                       padding_mode='border', align_corners=False)
            sampled_feats.append(sampled_feat)
        
        # Average the sampled features
        upsampled_feat = torch.stack(sampled_feats, dim=0).mean(dim=0)
        return upsampled_feat
    
class AFF(nn.Module):
    """
    Adaptive Feature Fusion module that learns pixel-wise weights
    for fusing features from different levels.
    """
    def __init__(self, in_channels, out_channels, reduction=16):
        super(AFF, self).__init__()
        
        # Feature processing
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Attention weight prediction
        self.attention_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // reduction, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, 2, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, high_feat, low_feat):
        # Ensure same spatial dimensions
        if high_feat.shape[2:] != low_feat.shape[2:]:
            high_feat = F.interpolate(high_feat, size=low_feat.shape[2:], 
                                    mode='bilinear', align_corners=False)
        
        # Concatenate features
        concat_feat = torch.cat([high_feat, low_feat], dim=1)
        
        # Process concatenated features
        feat = self.conv1(concat_feat)
        feat = self.relu(feat)
        feat = self.conv2(feat)
        feat = self.relu(feat)
        
        # Predict attention weights
        attention_weights = self.attention_conv(feat)
        
        # Split attention weights
        weight_high = attention_weights[:, 0:1, :, :]
        weight_low = attention_weights[:, 1:2, :, :]
        
        # Apply attention weights
        fused_feat = weight_high * high_feat + weight_low * low_feat
        
        return fused_feat    
    
class AdaFPNBlock(nn.Module):
    """
    AdaFPN Block that combines AdaUp and AFF modules
    """
    def __init__(self, high_channels, low_channels, out_channels):
        super(AdaFPNBlock, self).__init__()
        
        # Adaptive upsampling
        self.ada_up = AdaUp(high_channels, out_channels)
        
        # Adaptive feature fusion
        self.aff = AFF(out_channels + low_channels, out_channels)
        
    def forward(self, high_feat, low_feat):
        # Adaptive upsampling
        upsampled_feat = self.ada_up(high_feat, low_feat)
        
        # Adaptive feature fusion
        fused_feat = self.aff(upsampled_feat, low_feat)
        
        return fused_feat            

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
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
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
            
        # # Deconv
        # self.deconv = nn.ModuleList()
        # for i in range(len(in_channels_list) - 2):
        #     if in_channels == 0:
        #         raise ValueError("in_channels=0 is currently not supported")
        #     self.deconv.append(ConvTranspose(out_channels, out_channels, kernel_size=2, stride=2))
        
        
        # # 2025.04.16 @HyungseopLee: FaPN
        # self.feature_selection = nn.ModuleList()
        # for in_channel in in_channels_list:
        #     feature_select = FeatureSelectionModule(in_channel, out_channels)
        #     self.feature_selection.append(feature_select)
            
        # self.feature_align = nn.ModuleList()
        # for i in range(len(in_channels_list) - 2):
        #     feature_align = DCN(
        #         in_channels=out_channels,
        #         out_channels=out_channels,
        #         kernel_size=3,
        #         stride=1,
        #         padding=1,
        #         deformable_groups=8,
        #         extra_offset_mask=True,
        #     )
        #     self.feature_align.append(feature_align)
        
        # self.offset = nn.ModuleList()
        # for i in range(len(in_channels_list) - 2):
        #     offset = nn.Sequential(
        #         nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        #         nn.BatchNorm2d(out_channels),
        #     )
        #     self.offset.append(offset)
        
        
        # # AdaFPN https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9497077
        # self.ada_fpn_blocks = nn.ModuleList()
        # for i in range(len(in_channels_list) - 2):
        #     high_ch = out_channels
        #     low_ch = out_channels   
        #     out_ch = out_channels 
        #     self.ada_fpn_blocks.append(AdaFPNBlock(high_ch, low_ch, out_ch))
        

        
        
        # # mask r-cnn is 4
        # # retina net is 3

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

        # # FaPN
        # results = []
        # last_inner = self.feature_selection[-1](x[-1])  # Apply feature selection
        # results.append(self.get_result_from_layer_blocks(last_inner, -1))
        
        # for idx in range(len(x) - 2, -1, -1):
        #     # inner_lateral = self.feature_selection[idx](x[idx])  # Apply feature selection
        #     inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
        #     feat_shape = inner_lateral.shape[-2:]
        #     if idx == 0: # mask r-cnn
        #         # pass
        #         inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
        #         last_inner = inner_lateral + inner_top_down
        #     else :
        #         # offset = self.offset[len(x) - 2 - idx](torch.cat([inner_lateral, inner_top_down * 2], dim=1))
        #         # inner_top_down = self.feature_align[len(x) - 2 - idx]([inner_top_down, offset])
        #         last_inner = self.ada_fpn_blocks[len(x) - 2 - idx](last_inner, inner_lateral)
        #     results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))
        #     # results.insert(0, self.get_result_from_layer_blocks(inner_top_down, idx))
            

        # # Deconvolution
        # for idx in range(len(x) - 2, -1, -1):
        #     inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
        #     feat_shape = inner_lateral.shape[-2:]
        #     # deconv
        #     if idx == 0 :
        #         inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
        #     else :
        #         inner_top_down = self.deconv[len(x) - 2 - idx](last_inner)
        #     last_inner = inner_lateral + inner_top_down
        #     results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))
        

        # baseline
        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        
        # print(f"len(x): {len(x)}") # 4
        # # mask-rcnn is 4, idx = 2, 1, 0
        # retinanet, fcos is 3, idx = 1, 0
        
        # for idx in range(len(x) - 2, -1, -1): 
        #     inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
        #     feat_shape = inner_lateral.shape[-2:]
            
        #     name = names[idx]
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
            
        #     # self.feature_hooks[name] = {
        #     #     'fused': last_inner.detach().cpu(),
        #     #     'low': inner_lateral.detach().cpu(),
        #     # }
            
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
