import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from ppdet.modeling.ops import get_act_fn
from ..shape_spec import ShapeSpec
from ..backbones.csp_darknet import BaseConv
from ..backbones.cspresnet import RepVggBlock
from ppdet.modeling.transformers.detr_transformer import TransformerEncoder
from ..initializer import xavier_uniform_, linear_init_
from ..layers import MultiHeadAttention
from paddle import ParamAttr
from paddle.regularizer import L2Decay

__all__ = ['HybridEncoder']


class CSPRepLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(
                hidden_channels, hidden_channels, act=act)
            for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = BaseConv(
                hidden_channels,
                out_channels,
                ksize=1,
                stride=1,
                bias=bias,
                act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


@register
class TransformerLayer(nn.Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False):
        super(TransformerLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        self.self_attn = MultiHeadAttention(d_model, nhead, attn_dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = nn.Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self._reset_parameters()

    def _reset_parameters(self):
        linear_init_(self.linear1)
        linear_init_(self.linear2)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src = self.self_attn(q, k, value=src, attn_mask=src_mask)

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

from einops import rearrange
import numpy as np
class SpatialAlignTransNormer(nn.Layer):
    def __init__(self, dim, nhead, dropout=0.0, activation='gelu', is_first=False):
        super().__init__()
        self.eps = 1e-6
        self.nhead = nhead
        self.head_dim = dim // nhead

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_norm = nn.LayerNorm(self.head_dim, epsilon=self.eps)

        self.linear1 = nn.Linear(dim, dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.avg_pool_high = nn.Identity()
        self.avg_pool_low = nn.Identity()
        if is_first:
            self.avg_pool_low = nn.AvgPool2D(kernel_size=2, stride=2)
        else:
            self.avg_pool_high = nn.AvgPool2D(kernel_size=2, stride=2)
            self.avg_pool_low = nn.AvgPool2D(kernel_size=4, stride=4)

    def get_index(self, seq_len):
        index = np.pi / 2 * paddle.arange(1, seq_len + 1).reshape([1, -1, 1])
        params = self.create_parameter(index.shape, default_initializer=nn.initializer.Assign(index))
        params.stop_gradient = True
        return params

    def forward(self, a3, a4):
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

        a3_flat = a3_flat.astype(self.q_proj.weight.dtype)
        a4_flat = a4_flat.astype(self.k_proj.weight.dtype)

        Q = self.q_proj(a3_flat)
        K = self.k_proj(a4_flat)
        V = self.v_proj(a4_flat)

        Q = rearrange(Q, 'hw b (h d) -> (b h) hw d', h=self.nhead)
        K = rearrange(K, 'hw b (h d) -> (b h) hw d', h=self.nhead)
        V = rearrange(V, 'hw b (h d) -> (b h) hw d', h=self.nhead)

        # kernel function
        Q = F.normalize(Q, p=1, axis=-2, epsilon=self.eps)
        K = F.normalize(K, p=1, axis=-2, epsilon=self.eps)
        

        m = max(src_len, tgt_len)
        weight_index = self.get_index(m)
        Q_ = paddle.concat([
            Q * paddle.sin(weight_index[:, :tgt_len, :] / m),
            Q * paddle.cos(weight_index[:, :tgt_len, :] / m)
        ], axis=-1)
        K_ = paddle.concat([
            K * paddle.sin(weight_index[:, :src_len, :] / m),
            K * paddle.cos(weight_index[:, :src_len, :] / m)
        ], axis=-1)

        KV_ = paddle.einsum('nld,nlm->ndm', K_, V)
        attn_output = paddle.einsum('nld,ndm->nlm', Q_, KV_)
        attn_output = self.attn_norm(attn_output)

        attn_output = attn_output.transpose([1, 0, 2]).reshape([tgt_len, bs, -1])
        out = self.out_proj(attn_output)
        out = out.transpose([1, 0, 2])

        out = self.norm1(out)
        out = residual + self.dropout1(out)
        residual = out
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = self.norm2(out)
        out = residual + self.dropout2(out)

        a3_sa = out.transpose([0, 2, 1]).reshape([bs, c_a3, h, w])
        a3_sa = F.interpolate(a3_sa, size=original_a3.shape[2:], mode='bilinear', align_corners=False)
        a3_sa = a3_sa * original_a3

        return a3_sa

@register
@serializable
class HybridEncoder(nn.Layer):
    __shared__ = ['depth_mult', 'act', 'trt', 'eval_size']
    __inject__ = ['encoder_layer']

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 encoder_layer='TransformerLayer',
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 trt=False,
                 eval_size=None):
        super(HybridEncoder, self).__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size

        # channel projection
        self.input_proj = nn.LayerList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channel, hidden_dim, kernel_size=1, bias_attr=False),
                    nn.BatchNorm2D(
                        hidden_dim,
                        weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                        bias_attr=ParamAttr(regularizer=L2Decay(0.0)))))
        # encoder transformer
        self.encoder = nn.LayerList([
            TransformerEncoder(encoder_layer, num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
           
        # top-down fpn
        self.lateral_convs = nn.LayerList()
        self.fpn_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        # bottom-up pan
        self.downsample_convs = nn.LayerList()
        self.pan_blocks = nn.LayerList()
        for idx in range(len(in_channels) - 1):
            self.downsample_convs.append(
                BaseConv(
                    hidden_dim, hidden_dim, 3, stride=2, act=act))
            self.pan_blocks.append(
                CSPRepLayer(
                    hidden_dim * 2,
                    hidden_dim,
                    round(3 * depth_mult),
                    act=act,
                    expansion=expansion))

        self.sa_cross_attn = nn.LayerList()
        for i in range(len(in_channels) - 1):
            is_first=False
            if i == 0:
                is_first = True

            self.sa_cross_attn.append(
                SpatialAlignTransNormer(dim=hidden_dim, nhead=16, activation='gelu', is_first=is_first)
            )


        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w,
                                           h,
                                           embed_dim=256,
                                           temperature=10000.):
        grid_w = paddle.arange(int(w), dtype=paddle.float32)
        grid_h = paddle.arange(int(h), dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @omega[None]
        out_h = grid_h.flatten()[..., None] @omega[None]

        return paddle.concat(
            [
                paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
                paddle.cos(out_h)
            ],
            axis=1)[None, :, :]

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)
        # get projection features
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).transpose(
                    [0, 2, 1])
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None)
                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.transpose([0, 2, 1]).reshape(
                    [-1, self.hidden_dim, h, w])

        # top-down fpn
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](
                feat_high)
            inner_outs[0] = feat_high
            
            # # baseline 
            # upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            
            # Ours (align low to high)
            feat_low = self.sa_cross_attn[len(self.in_channels) - 1 - idx](feat_low, feat_high)
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='bilinear')
            
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](
                paddle.concat(
                    [upsample_feat, feat_low], axis=1))
            inner_outs.insert(0, inner_out)

        # bottom-up pan
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](paddle.concat(
                [downsample_feat, feat_height], axis=1))
            outs.append(out)

        return outs

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'feat_strides': [i.stride for i in input_shape]
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.hidden_dim, stride=self.feat_strides[idx])
            for idx in range(len(self.in_channels))
        ]
