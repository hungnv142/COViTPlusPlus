import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import math

from .linear_super import LinearSuper


class DWConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, groups, kernel_size=3, stride=1, padding=1):
        super(DWConv, self).__init__(in_channels, out_channels, kernel_size=kernel_size,\
                                    stride=stride, padding=padding, groups=groups)
        self._reset_parameters()


    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight) 
        nn.init.constant_(self.bias, 0.)


    def forward(self, x, H, W):
        self.H, self.W = H, W
        B, N, C = x.shape

        x = x.transpose(1, 2).view(B, C, H, W)
        x = F.conv2d(x, self.weight, self.bias,\
               stride=self.stride, padding=self.padding, groups=self.groups)

        x = x.flatten(2).transpose(1, 2)
        return x


    def get_complexity(self):
        return (self.H / self.stride[0] * self.W / self.stride[1])\
             * self.kernel_size[0] ** 2 * ((self.in_channels/self.groups) * (self.out_channels/self.groups) * self.groups)



class MlpSuper(nn.Module):
    def __init__(self, super_in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or super_in_features
        hidden_features = hidden_features or super_in_features

        self.in_features = super_in_features
        self.out_features = out_features

        self.fc1 = LinearSuper(super_in_features, hidden_features)
        self.dwconv = DWConv(hidden_features, hidden_features, hidden_features)
        self.act = act_layer()

        self.fc2 = LinearSuper(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)

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


    def get_complexity(self, sequence_length):
        total_flops = 0

        total_flops += self.fc1.get_complexity(sequence_length)
        total_flops += self.dwconv.get_complexity()
        total_flops += self.fc2.get_complexity(sequence_length)

        return total_flops


    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x
