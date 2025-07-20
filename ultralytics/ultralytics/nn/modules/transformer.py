# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Transformer modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from .conv import Conv, DWConv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch

__all__ = (
    "TransformerEncoderLayer",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "CrossAttentionUpsample",
    "SpatialAlignCrossAttnFusion",
    "SpatialAlignCrossAttentionReLULinear",
    "SpatialAlignCrossAttentionCosFormer"
)


class TransformerEncoderLayer(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        from ...utils.torch_utils import TORCH_1_9

        if not TORCH_1_9:
            raise ModuleNotFoundError(
                "TransformerEncoderLayer() requires torch>=1.9 to use nn.MultiheadAttention(batch_first=True)."
            )
        self.ma = nn.MultiheadAttention(c1, num_heads, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.fc1 = nn.Linear(c1, cm)
        self.fc2 = nn.Linear(cm, c1)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    @staticmethod
    def with_pos_embed(tensor, pos=None):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        q = k = self.with_pos_embed(src, pos)
        src2 = self.ma(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with pre-normalization."""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.ma(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src2))))
        return src + self.dropout2(src2)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class AIFI(TransformerEncoderLayer):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


class TransformerLayer(nn.Module):
    """Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)."""

    def __init__(self, c, num_heads):
        """Initializes a self-attention mechanism using linear transformations and multi-head attention."""
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        """Apply a transformer block to the input x and return the output."""
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x


class TransformerBlock(nn.Module):
    """Vision Transformer https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1, c2, num_heads, num_layers):
        """Initialize a Transformer module with position embedding and specified number of heads and layers."""
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        """Forward propagates the input through the bottleneck module."""
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class MLPBlock(nn.Module):
    """Implements a single block of a multi-layer perceptron."""

    def __init__(self, embedding_dim, mlp_dim, act=nn.GELU):
        """Initialize the MLPBlock with specified embedding dimension, MLP dimension, and activation function."""
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLPBlock."""
        return self.lin2(self.act(self.lin1(x)))


class MLP(nn.Module):
    """Implements a simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act=nn.ReLU, sigmoid=False):
        """Initialize the MLP with specified input, hidden, output dimensions and number of layers."""
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid = sigmoid
        self.act = act()

    def forward(self, x):
        """Forward pass for the entire MLP."""
        for i, layer in enumerate(self.layers):
            x = getattr(self, "act", nn.ReLU())(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x.sigmoid() if getattr(self, "sigmoid", False) else x


class LayerNorm2d(nn.Module):
    """
    2D Layer Normalization module inspired by Detectron2 and ConvNeXt implementations.

    Original implementations in
    https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py
    and
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py.
    """

    def __init__(self, num_channels, eps=1e-6):
        """Initialize LayerNorm2d with the given parameters."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        """Perform forward pass for 2D layer normalization."""
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MSDeformAttn(nn.Module):
    """
    Multiscale Deformable Attention Module based on Deformable-DETR and PaddleDetection implementations.

    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/modules/ms_deform_attn.py
    """

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """Initialize MSDeformAttn with the given parameters."""
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, but got {d_model} and {n_heads}")
        _d_per_head = d_model // n_heads
        # Better to set _d_per_head to a power of 2 which is more efficient in a CUDA implementation
        assert _d_per_head * n_heads == d_model, "`d_model` must be divisible by `n_heads`"

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        """Reset module parameters."""
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(self, query, refer_bbox, value, value_shapes, value_mask=None):
        """
        Perform forward pass for multiscale deformable attention.

        https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py

        Args:
            query (torch.Tensor): [bs, query_length, C]
            refer_bbox (torch.Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (torch.Tensor): [bs, value_length, C]
            value_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, len_q = query.shape[:2]
        len_v = value.shape[1]
        assert sum(s[0] * s[1] for s in value_shapes) == len_v

        value = self.value_proj(value)
        if value_mask is not None:
            value = value.masked_fill(value_mask[..., None], float(0))
        value = value.view(bs, len_v, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(bs, len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(bs, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(bs, len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        num_points = refer_bbox.shape[-1]
        if num_points == 2:
            offset_normalizer = torch.as_tensor(value_shapes, dtype=query.dtype, device=query.device).flip(-1)
            add = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            sampling_locations = refer_bbox[:, :, None, :, None, :] + add
        elif num_points == 4:
            add = sampling_offsets / self.n_points * refer_bbox[:, :, None, :, None, 2:] * 0.5
            sampling_locations = refer_bbox[:, :, None, :, None, :2] + add
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {num_points}.")
        output = multi_scale_deformable_attn_pytorch(value, value_shapes, sampling_locations, attention_weights)
        return self.output_proj(output)


class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer inspired by PaddleDetection and Deformable-DETR implementations.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_transformer.py
    """

    def __init__(self, d_model=256, n_heads=8, d_ffn=1024, dropout=0.0, act=nn.ReLU(), n_levels=4, n_points=4):
        """Initialize the DeformableTransformerDecoderLayer with the given parameters."""
        super().__init__()

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.act = act
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional embeddings to the input tensor, if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """Perform forward pass through the Feed-Forward Network part of the layer."""
        tgt2 = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        return self.norm3(tgt)

    def forward(self, embed, refer_bbox, feats, shapes, padding_mask=None, attn_mask=None, query_pos=None):
        """Perform the forward pass through the entire decoder layer."""
        # Self attention
        q = k = self.with_pos_embed(embed, query_pos)
        tgt = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), embed.transpose(0, 1), attn_mask=attn_mask)[
            0
        ].transpose(0, 1)
        embed = embed + self.dropout1(tgt)
        embed = self.norm1(embed)

        # Cross attention
        tgt = self.cross_attn(
            self.with_pos_embed(embed, query_pos), refer_bbox.unsqueeze(2), feats, shapes, padding_mask
        )
        embed = embed + self.dropout2(tgt)
        embed = self.norm2(embed)

        # FFN
        return self.forward_ffn(embed)


class DeformableTransformerDecoder(nn.Module):
    """
    Implementation of Deformable Transformer Decoder based on PaddleDetection.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/transformers/deformable_transformer.py
    """

    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        """Initialize the DeformableTransformerDecoder with the given parameters."""
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx

    def forward(
        self,
        embed,  # decoder embeddings
        refer_bbox,  # anchor
        feats,  # image features
        shapes,  # feature shapes
        bbox_head,
        score_head,
        pos_mlp,
        attn_mask=None,
        padding_mask=None,
    ):
        """Perform the forward pass through the entire decoder."""
        output = embed
        dec_bboxes = []
        dec_cls = []
        last_refined_bbox = None
        refer_bbox = refer_bbox.sigmoid()
        for i, layer in enumerate(self.layers):
            output = layer(output, refer_bbox, feats, shapes, padding_mask, attn_mask, pos_mlp(refer_bbox))

            bbox = bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox))

            if self.training:
                dec_cls.append(score_head[i](output))
                if i == 0:
                    dec_bboxes.append(refined_bbox)
                else:
                    dec_bboxes.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.eval_idx:
                dec_cls.append(score_head[i](output))
                dec_bboxes.append(refined_bbox)
                break

            last_refined_bbox = refined_bbox
            refer_bbox = refined_bbox.detach() if self.training else refined_bbox

        return torch.stack(dec_bboxes), torch.stack(dec_cls)

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


class CrossAttention(nn.Module):
    def __init__(self,
                 c1,              # Channel dimension (e.g., 512 or 256)
                 nhead=8,                # Number of attention heads
                 dim_feedforward=2048, # Feedforward network dimension
                 dropout=0.1,          # Dropout rate
                 activation="gelu",    # Activation function
                 normalize_before=False):  # Pre-norm or post-norm
        super().__init__()
        self.normalize_before = normalize_before

        # Cross-attention: feat_high (query) attends to feat_low (key/value)
        self.cross_attn = nn.MultiheadAttention(c1, nhead, dropout=dropout, batch_first=True)

        # Feedforward network
        self.linear1 = nn.Linear(c1, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, c1)

        # Normalization layers
        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

        # Average pooling for downsampling feat_low
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, feat_high, feat_low, pos_embed_high=None, pos_embed_low=None):
        """
        Args:
            feat_high (torch.Tensor): Lower resolution features [bs, c, h, w]
            feat_low (torch.Tensor): Higher resolution features [bs, c, 2h, 2w]
            pos_embed_high (torch.Tensor, optional): Positional embedding for feat_high [bs, c, h, w]
            pos_embed_low (torch.Tensor, optional): Positional embedding for feat_low [bs, c, 2h, 2w]
        
        Returns:
            torch.Tensor: Output features with shape [bs, c, h, w]
        """
        bs, c, h, w = feat_high.shape
        _, _, H, W = feat_low.shape
        assert H == 2 * h and W == 2 * w, "feat_low must have twice the spatial resolution of feat_high"

        # Reshape feature maps to [bs, seq_len, c] for attention
        feat_high_flat = feat_high.view(bs, c, -1).permute(0, 2, 1)  # [bs, h*w, c]
        feat_low_pooled = self.avg_pool(feat_low)  # Downsample: [bs, c, h, w]
        feat_low_flat = feat_low_pooled.view(bs, c, -1).permute(0, 2, 1)  # [bs, h*w, c]

        # Add positional embeddings if provided
        if pos_embed_high is not None:
            pos_embed_high = pos_embed_high.view(bs, c, -1).permute(0, 2, 1)  # [bs, h*w, c]
            feat_high_flat = feat_high_flat + pos_embed_high
        if pos_embed_low is not None:
            pos_embed_low = self.avg_pool(pos_embed_low)  # Downsample to match feat_low_pooled
            pos_embed_low = pos_embed_low.view(bs, c, -1).permute(0, 2, 1)  # [bs, h*w, c]
            feat_low_flat = feat_low_flat + pos_embed_low

        # Cross-attention
        residual = feat_high_flat
        if self.normalize_before:
            feat_high_flat = self.norm1(feat_high_flat)
        q = feat_high_flat  # Query from low-res
        k = feat_low_flat   # Key from high-res (pooled)
        v = feat_low_flat   # Value from high-res (pooled)
        out, _ = self.cross_attn(q, k, v)  # [bs, h*w, c]

        out = residual + self.dropout1(out)
        if not self.normalize_before:
            out = self.norm1(out)

        # Feedforward network
        residual = out
        if self.normalize_before:
            out = self.norm2(out)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = residual + self.dropout2(out)
        if not self.normalize_before:
            out = self.norm2(out)

        # Reshape back to [bs, c, h, w]
        out = out.permute(0, 2, 1).view(bs, c, h, w)
        return out
    


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
    

def get_activation(name):
    return nn.GELU() if name.lower() == 'gelu' else nn.ReLU()
        
class SpatialAlignCrossAttnFusion(nn.Module):
    def __init__(self, c1, c2, nhead, dropout=0.0, activation="gelu"):
        super().__init__()

        self.normalize_before = False
        self.dropout = dropout
        self.high_to_low_embed = nn.Sequential(
            nn.Conv2d(c2, c1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(c1),
        )
        
        self.cross_attn = nn.MultiheadAttention(c1, num_heads=nhead, dropout=0.0, batch_first=True)

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
        # print(f"idx={idx}, a3.shape: {a3.shape}, a4.shape: {a4.shape}")
        original_a3 = a3

        a4 = self.high_to_low_embed(a4)  # Project to match a3

        # Downsample features
        # v8: 10, v5, v10: 11
        # print(f"idx={idx}, a3.shape: {a3.shape}, a4.shape: {a4.shape}")
        if idx == 11:
            a3_down = self.avg_pool_2x(a3)
        else:
            a3_down = self.avg_pool_4x(a3)
            a4 = self.avg_pool_2x(a4)
            
        # print(f"idx={idx}, a3_down.shape: {a3_down.shape}, a4.shape: {a4.shape}")
            
        bs, c_a3, h, w = a3_down.shape
        _, _, H, W = a4.shape
        
        # Flatten spatial dimensions
        q = a3_down.flatten(2).permute(0, 2, 1)  # [bs, hw, dim]
        k = a4.flatten(2).permute(0, 2, 1)       # [bs, HW, dim]
        v = k.clone()
        

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
        # v8: 10, v5, v10: 11
        a3_sa = F.interpolate(a3_sa, size=original_a3.shape[2:], mode='bilinear', align_corners=False)
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
        
        # others: low=c1 < high=c2. high=c2 -> c1
        # yolov6: low=c1 > high=c2. low=c1 -> c2
        self.is_yolov6 = is_yolov6
        # bilinear interpolation upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        if self.is_yolov6:
            self.upsample = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2)
        
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
            
        # print(f"idx={idx}, a3_down.shape: {a3_down.shape}, a4.shape: {a4.shape}")
            
        # Flatten input for projection
        residual = a3_down.flatten(2).permute(0, 2, 1)  # [B, hw, C]
        bs, c_a3, h, w = a3_down.shape
        _, c_a4, H, W = a4.shape
        
        residual = a3_down.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        a3_flat = a3_down.flatten(2).permute(2, 0, 1) # [HW, B, C]
        a4_flat = a4.flatten(2).permute(2, 0, 1)       # [HW, B, C]
        
        # pos = self.build_2d_sincos_position_embedding(w=h, h=w, embed_dim=c_a3).to(a3.device)  # [1, HW, C]
        # pos = pos.squeeze(0).unsqueeze(1).expand(-1, bs, -1)  # [HW, B, C]
        # a3_flat = self.with_pos_embed(a3_flat, pos)  # [HW, B, C]
        # a4_flat = self.with_pos_embed(a4_flat, pos)  # [HW, B, C]
        
        # Positional encoding
        pos = self.get_sinusoidal_pos_embed_1d(h * w, c_a3, a3_down.device)
        a3_flat = self.with_pos_embed(a3_flat, pos)
        a4_flat = self.with_pos_embed(a4_flat, pos)
        
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
        # 5. Reshape to spatial map [B, HW, C] -> [B, C, H, W]
        a3_sa = out.permute(0, 2, 1).contiguous().view(bs, c_a3, h, w)  # [B, C, H, W]
        a3_sa = F.interpolate(a3_sa, size=original_a3.shape[2:], mode='bilinear', align_corners=False)
        a3_sa = a3_sa * original_a3
        
        if self.is_yolov6 :
            return a3_sa, self.upsample(original_a4)  # Return both the aligned feature and the upsampled feature
        else: 
            return a3_sa, F.interpolate(original_a4, scale_factor=2., mode='bilinear')
    
    
# This implementation is based on the original CosFormer paper, https://github.com/OpenNLPLab/cosFormer/blob/main/cosformer.py
import numpy as np
class SpatialAlignCrossAttentionCosFormer(nn.Module):
    def __init__(self, c1, c2, nhead=8, is_yolov6=False, dropout=0.0, activation="gelu"):
        super().__init__()
        self.min_eps = 1e-4
        self.max_eps = 1e1
        
        self.normalize_before = False
        
        self.heads = nhead
        self.head_dim = c1 // nhead
        self.is_yolov6 = is_yolov6

        self.upsample = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2) if is_yolov6 \
                        else nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.high_to_low_embed = nn.Sequential(
            nn.Conv2d(c2, c1, kernel_size=1),
            nn.BatchNorm2d(c1)
        )

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
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(c1 * 4, c1)
        self.activation = get_activation(activation)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.avg_pool_2x = nn.AvgPool2d(2)
        self.avg_pool_4x = nn.AvgPool2d(4)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        
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
    
    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)
        return nn.Parameter(index, requires_grad=False)
    
    def forward(self, a3, a4, idx):
        a3 = torch.nan_to_num(a3, nan=0.0, posinf=1e4, neginf=-1e4)
        a4 = torch.nan_to_num(a4, nan=0.0, posinf=1e4, neginf=-1e4)
        
        original_a3, original_a4 = a3, a4
        
        assert not torch.isnan(a3).any(), "NaN detected in a3_down"
        assert not torch.isnan(a4).any(), "NaN detected in a4"
        a4 = self.high_to_low_embed(a4)
        
        # assert not torch.isnan(a3).any(), f"NaN detected in a3"
        # assert not torch.isnan(a4).any(), f"NaN detected in a4"
        
        if idx == 10:
            a3_down = self.avg_pool_2x(a3)
        else:
            a3_down = self.avg_pool_4x(a3)
            a4 = self.avg_pool_2x(a4)

        assert not torch.isnan(a3_down).any(), "NaN detected in a3_down"
        assert not torch.isnan(a4).any(), "NaN detected in proj_a4"

        bs, c, h, w = a3_down.shape
        tgt_len, src_len = h * w, a4.shape[2] * a4.shape[3]

        residual = a3_down.flatten(2).permute(0, 2, 1)
        a3_flat = a3_down.flatten(2).permute(2, 0, 1)
        a4_flat = a4.flatten(2).permute(2, 0, 1)

        pos = self.build_2d_sincos_position_embedding(h, w, c).to(a3.device)
        pos = pos.squeeze(0).unsqueeze(1).expand(-1, bs, -1)
        a3_flat = self.with_pos_embed(a3_flat, pos)
        a4_flat = self.with_pos_embed(a4_flat, pos)
        
        assert not torch.isnan(a3_flat).any(), "NaN detected in a3_flat"
        assert not torch.isnan(a4_flat).any(), "NaN detected in a4_flat"

        # Project Q, K, V
        a3_flat = a3_flat.to(self.q_proj[0].weight.dtype)
        a4_flat = a4_flat.to(self.k_proj[0].weight.dtype)
        Q = self.q_proj(a3_flat) # [HW, B, C]
        K = self.k_proj(a4_flat) # [HW, B, C]
        V = self.v_proj(a4_flat.clone()) # [HW, B, C]
        Q = F.relu(Q)  # [HW, B, C]
        K = F.relu(K)  # [HW, B, C]

        assert not torch.isnan(Q).any(), "NaN detected in Q"
        assert not torch.isnan(K).any(), "NaN detected in K"
        assert not torch.isnan(V).any(), "NaN detected in V"

        Q = Q.view(tgt_len, bs, self.heads, self.head_dim).permute(1, 2, 0, 3).reshape(-1, tgt_len, self.head_dim)
        K = K.view(src_len, bs, self.heads, self.head_dim).permute(1, 2, 0, 3).reshape(-1, src_len, self.head_dim)
        V = V.view(src_len, bs, self.heads, self.head_dim).permute(1, 2, 0, 3).reshape(-1, src_len, self.head_dim)

        m = max(src_len, tgt_len)
        weight_index = self.get_index(m).to(Q)

        Q_ = torch.cat([
            Q * torch.sin(weight_index[:, :tgt_len, :] / m),
            Q * torch.cos(weight_index[:, :tgt_len, :] / m)
        ], dim=-1)

        K_ = torch.cat([
            K * torch.sin(weight_index[:, :src_len, :] / m),
            K * torch.cos(weight_index[:, :src_len, :] / m)
        ], dim=-1)


        KV_ = torch.einsum('nld,nlm->ndm', K_, V)
        Z_den = torch.einsum('nld,nd->nl', Q_, torch.sum(K_, axis=1))  # [N*h, L]
        Z_ = 1 / torch.clamp(Z_den, min=self.min_eps, max=self.max_eps)

        attn_output = torch.einsum('nld,ndm,nl->nlm', Q_, KV_, Z_)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bs, -1)
        out = self.out_proj(attn_output).permute(1, 0, 2) # [B, HW, C]

        # FFN
        if not self.normalize_before:
            out = self.norm1(out)
        out = residual + self.dropout1(out)
        residual = out
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        if not self.normalize_before:
            out = self.norm2(out)
        out = residual + self.dropout2(out) # [B, HW, C]

        a3_sa = out.permute(0, 2, 1).contiguous().view(bs, c, h, w)
        a3_sa = F.interpolate(a3_sa, size=original_a3.shape[2:], mode='bilinear', align_corners=False)
        a3_sa = a3_sa * original_a3

        a4_upsampled = self.upsample(original_a4)
        return a3_sa, a4_upsampled



class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - more efficient than LayerNorm"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms
    

from einops import rearrange
class SpatialAlignTransnormer(nn.Module):
    def __init__(self, c1, c2, nhead=8, is_yolov6=False, dropout=0.0, activation="gelu", max_seq_len=22500):
        super().__init__()
        self.eps = 1e-4
        self.normalize_before = False
        
        self.heads = nhead
        self.head_dim = c1 // nhead
        self.is_yolov6 = is_yolov6
        
        # Cache weight indices for efficiency
        
        weight_index = np.pi / 2 * torch.arange(1, max_seq_len + 1).reshape(1, -1, 1)
        self.register_buffer('weight_index_cached', weight_index.float())
        self.register_buffer('sin_weights', torch.sin(self.weight_index_cached / max_seq_len))
        self.register_buffer('cos_weights', torch.cos(self.weight_index_cached / max_seq_len))

        # cos-based nonlinear reweighting        

        # # forward 1
        base_index = (np.pi / 2) * torch.arange(1, max_seq_len+1).reshape(1, -1, 1)
        # self.register_buffer('weight_index_base', base_index.float())
        
        # forward 2
        self.register_buffer('base_index', torch.arange(1, max_seq_len+1).float())
        
        # bilinear interpolation upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        if self.is_yolov6:
            self.upsample = nn.ConvTranspose2d(c2, c2, kernel_size=2, stride=2)
        
        # others: low=c1 < high=c2. high=c2 -> c1
        # yolov6: low=c1 > high=c2. low=c1 -> c2
        self.high_to_low_embed = nn.Sequential(
            nn.Conv2d(c2, c1, kernel_size=1),
            nn.BatchNorm2d(c1)
        )

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
        
        self.attn_norm = RMSNorm(self.head_dim)
        
        # FFN
        self.linear1 = nn.Linear(c1, c1 * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(c1 * 4, c1)
        self.activation = get_activation(activation)

        self.norm1 = nn.LayerNorm(c1)
        self.norm2 = nn.LayerNorm(c1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.avg_pool_2x = nn.AvgPool2d(2)
        self.avg_pool_4x = nn.AvgPool2d(4)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        
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
    
    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)
        return nn.Parameter(index, requires_grad=False)
    
    # def forward(self, a3, a4, idx):
    #     a3 = torch.nan_to_num(a3, nan=0.0, posinf=1e4, neginf=-1e4)
    #     a4 = torch.nan_to_num(a4, nan=0.0, posinf=1e4, neginf=-1e4)
        
    #     original_a3, original_a4 = a3, a4
        
    #     assert not torch.isnan(a3).any(), "NaN detected in a3_down"
    #     assert not torch.isnan(a4).any(), "NaN detected in a4"
    #     a4 = self.high_to_low_embed(a4)
        
    #     # Downsample features
    #     # v8: 10, 
    #     # v5, v10: 11
    #     # v6: 12
    #     if idx == 10:
    #         a3_down = self.avg_pool_2x(a3)
    #     else:
    #         a3_down = self.avg_pool_4x(a3)
    #         a4 = self.avg_pool_2x(a4)

    #     bs, c_a3, h, w = a3_down.shape
    #     _, c_a4, H, W = a4.shape
    #     tgt_len = h * w
    #     src_len = H * W
        
    #     residual = rearrange(a3_down, 'b c h w -> b (h w) c')
    #     a3_flat = rearrange(a3_down, 'b c h w -> (h w) b c')
    #     a4_flat = rearrange(a4, 'b c h w -> (h w) b c')
        
    #     # positional encoding
    #     pos = self.build_2d_sincos_position_embedding(w=w, h=h, embed_dim=c_a3).to(a3.device)
    #     pos = pos.expand(bs, -1, -1).permute(1, 0, 2)  # [HW, B, C]
    #     a3_flat = self.with_pos_embed(a3_flat, pos)  # [HW, B, C]
    #     a4_flat = self.with_pos_embed(a4_flat, pos)  # [HW, B, C]
        
    #     # embedding
    #     a3_flat = a3_flat.to(self.q_proj[0].weight.dtype)
    #     a4_flat = a4_flat.to(self.k_proj[0].weight.dtype)
    #     Q = self.q_proj(a3_flat)
    #     K = self.k_proj(a4_flat)
    #     V = self.v_proj(a4_flat)
        
    #     # QK normalization for stability
    #     Q = F.normalize(Q, p=2, dim=-1, eps=self.eps)
    #     K = F.normalize(K, p=2, dim=-1, eps=self.eps) # [hw, b, C]

    #     # Memory-efficient multi-head reshape using einops
    #     Q = rearrange(Q, 'hw b (h d) -> (b h) hw d', h=self.heads)
    #     K = rearrange(K, 'hw b (h d) -> (b h) hw d', h=self.heads)
    #     V = rearrange(V, 'hw b (h d) -> (b h) hw d', h=self.heads)
        
    #     # CosFormer transform
    #     m = max(src_len, tgt_len)
    #     weight_index = self.weight_index_base[:, :m, :].to(Q.device) / m
    #     sin_weights = torch.sin(weight_index)
    #     cos_weights = torch.cos(weight_index)

    #     Q_ = torch.cat([
    #         Q *  sin_weights[:, :tgt_len, :],
    #         Q *  cos_weights[:, :tgt_len, :]
    #     ], dim=-1)

    #     K_ = torch.cat([
    #         K *  sin_weights[:, :src_len, :],
    #         K *  cos_weights[:, :src_len, :]
    #     ], dim=-1)

    #     # NormAttention
    #     KV_ = torch.einsum('nld,nlm->ndm', K_, V)
    #     attn_output = torch.einsum('nld,ndm->nlm', Q_, KV_)
    #     attn_output = self.attn_norm(attn_output)

    #     # [N*h, L, d] -> [L, N*h, d] -> [L, B, C]
    #     attn_output = rearrange(attn_output, '(b h) l d -> b l (h d)', b=bs, h=self.heads)
    #     out = self.out_proj(attn_output)  # [B, L, C]
        
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
    #     a3_sa = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
    #     a3_sa = F.interpolate(a3_sa, size=original_a3.shape[2:], mode='bilinear', align_corners=False)
    #     a3_sa = a3_sa * original_a3
        
    #     if self.is_yolov6 :
    #         return a3_sa, self.upsample(original_a4)  # Return both the aligned feature and the upsampled feature
    #     else: 
    #         return a3_sa, F.interpolate(original_a4, scale_factor=2., mode='bilinear')
        
        
    def forward(self, a3, a4, idx):
        original_a3, original_a4 = a3, a4
        a4 = self.high_to_low_embed(a4)
        
        # v8: 10, 
        # v5, v10: 11
        # v6: 12
        if idx == 10:
            a3_down = self.avg_pool_2x(a3)
        else:
            a3_down = self.avg_pool_4x(a3)
            a4 = self.avg_pool_2x(a4)

        bs, c, h, w = a3_down.shape
        tgt_len, src_len = h * w, a4.shape[2] * a4.shape[3]

        residual = a3_down.flatten(2).permute(0, 2, 1)
        a3_flat = a3_down.flatten(2).permute(2, 0, 1)
        a4_flat = a4.flatten(2).permute(2, 0, 1)

        pos = self.build_2d_sincos_position_embedding(h, w, c).to(a3.device)
        pos = pos.squeeze(0).unsqueeze(1).expand(-1, bs, -1)
        a3_flat = self.with_pos_embed(a3_flat, pos)
        a4_flat = self.with_pos_embed(a4_flat, pos)
        
        # Project Q, K, V
        a3_flat = a3_flat.to(self.q_proj[0].weight.dtype)
        a4_flat = a4_flat.to(self.k_proj[0].weight.dtype)
        Q = self.q_proj(a3_flat) # [HW, B, C]
        K = self.k_proj(a4_flat) # [HW, B, C]
        V = self.v_proj(a4_flat) # [HW, B, C]
        
        # Kernel Function
        Q = F.normalize(Q, p=2, dim=-1, eps=self.eps)
        K = F.normalize(K, p=2, dim=-1, eps=self.eps)

        # multi-head reshape
        Q = Q.contiguous().view(-1, bs * self.heads, self.head_dim).transpose(0, 1)  # [N*h, L, d]
        K = K.contiguous().view(-1, bs * self.heads, self.head_dim).transpose(0, 1)  # [N*h, L, d]
        V = V.contiguous().view(-1, bs * self.heads, self.head_dim).transpose(0, 1)  # [N*h, L, d]

        m = max(src_len, tgt_len)
        weight_index = self.base_index[:m].to(Q).view(1, m, 1) / m
        sin_weights = torch.sin(weight_index)
        cos_weights = torch.cos(weight_index)
        Q_ = torch.cat([
            Q * cos_weights,
            Q * sin_weights,
        ], dim=-1)
        K_ = torch.cat([
            K * cos_weights,
            K * sin_weights,
        ], dim=-1)
    
        KV_ = torch.einsum('nld,nlm->ndm', K_, V)
        attn_output = torch.einsum('nld,ndm->nlm', Q_, KV_) # [N*h, L, d]
        
        # [N*h, L, d] -> [L, N*h, d] -> [L, B, C]
        attn_output = self.attn_norm(attn_output)
        attn_output = rearrange(attn_output, '(b h) l d -> l b (h d)', b=bs, h=self.heads)
        out = self.out_proj(attn_output)
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

        a3_sa = out.permute(0, 2, 1).contiguous().view(bs, c, h, w)
        a3_sa = F.interpolate(a3_sa, size=original_a3.shape[2:], mode='bilinear', align_corners=False)
        a3_sa = a3_sa * original_a3

        a4_upsampled = self.upsample(original_a4)
        return a3_sa, a4_upsampled
