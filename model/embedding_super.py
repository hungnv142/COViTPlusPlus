import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .layernorm_super import LayerNormSuper

import math


class OverlapPatchembedSuper(nn.Module):
    def __init__(self, img_size=256, patch_size=7, stride=3, in_chans=3, embed_dim=256):
        super(OverlapPatchembedSuper, self).__init__()

        self.img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.H, self.W = self.img_size[0] // patch_size[0], self.img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,\
                            padding=(patch_size[0] // 2, patch_size[1] // 2))

        self.norm = LayerNormSuper(embed_dim)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x):
        x = self.proj(x)

        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)

        return x, H, W


    def calc_sampled_param_num(self):
        return  self.proj.weight.numel() + self.proj.bias.numel()

    
    def get_complexity(self, sequence_length):
        total_flops = 0

        total_flops +=(self.img_size[0] / self.proj.stride[0] * self.img_size[1] / self.proj.stride[1])\
                                 * self.proj.kernel_size[0] ** 2 * self.proj.in_channels * self.proj.out_channels
        total_flops += self.norm.get_complexity(sequence_length)
        
        return total_flops
