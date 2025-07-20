"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

from ...core import register


__all__ = ['RTDETR', ]


@register()
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module, 
        encoder: nn.Module, 
        decoder: nn.Module, 
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        
    def forward(self, x, targets=None):
        x = self.backbone(x)
        
        # # get specific backbone layer parameter (backbone.res_layers[2][0], backbone.res_layers[3][0])
        # down_s2_s3 = self.backbone.res_layers[2].blocks[0]
        # down_s3_s4 = self.backbone.res_layers[3].blocks[0]

        # print(f"\tdown_s2_s3: {down_s2_s3}") # 128 -> 256
        # print(f"\tdown_s3_s4: {down_s3_s4}") # 256 -> 512
        
        x = self.encoder(x)
        x = self.decoder(x, targets)
        
        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
