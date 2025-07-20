import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.ops
import math

__all__ = ['DCNv2', 'DCN', 'FeatureSelectionModule', 'FaPN', 'DWConv',
           'TConvNormLayer', 'TConvFromDownsample', 'AdaUp', 'AFF', 'AdaFPNBlock']

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