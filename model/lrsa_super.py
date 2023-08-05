import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import math
import numpy as np

from .linear_super import LinearSuper
from .layernorm_super import LayerNormSuper


class Conv2D(nn.Conv2d):
    def __init__(self, channels, kernel_size=1):
        super(Conv2D, self).__init__(in_channels=channels, out_channels=channels, kernel_size=kernel_size)
        self._reset_parameters()


    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.weight) 
        nn.init.constant_(self.bias, 0.)


    def forward(self, x):
        _, _, self.H, self.W = x.shape

        x = F.conv2d(x, self.weight, self.bias,\
               stride=1, padding=0, dilation=1, groups=1)
        return x


    def get_complexity(self):
        return (self.H / self.stride[0] * self.W / self.stride[1]) * self.kernel_size[0] ** 2 * self.in_channels * self.out_channels


    def calc_sampled_param_num(self):
        return self.weight.numel() + self.bias.numel()



class AttentionSuper(nn.Module):
    def __init__(self, super_embed_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert super_embed_dim % num_heads == 0, f"super_embed_dim {super_embed_dim} should be divided by num_heads {num_heads}."

        self.num_heads = num_heads
        head_dim = super_embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.super_embed_dim = super_embed_dim
        self.dim = super_embed_dim

        self.q = LinearSuper(super_embed_dim, super_embed_dim, bias=qkv_bias)
        self.kv = LinearSuper(super_embed_dim, super_embed_dim*2, bias=qkv_bias)

        self.proj = LinearSuper(super_embed_dim, super_embed_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr = Conv2D(super_embed_dim)

        self.norm = LayerNormSuper(super_embed_dim)
        self.act = nn.GELU()

        self.sample_num_heads = None
        self.sample_scale = None

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


    def set_sample_config(self, sample_qkv_embed_dim, sample_pooling_dim=None, attn_drop=None, drop=None):

        self.sample_qkv_embed_dim = sample_qkv_embed_dim
        self.sample_pooling_dim = sample_pooling_dim

        self.dim = self.sample_qkv_embed_dim
        
        self.scale = (self.sample_qkv_embed_dim // self.num_heads) ** -0.5

        self.q.set_sample_config(sample_in_dim=self.super_embed_dim, sample_out_dim=self.sample_qkv_embed_dim)
        self.kv.set_sample_config(sample_in_dim=self.super_embed_dim, sample_out_dim=2*self.sample_qkv_embed_dim)

        # self.norm.set_sample_config(sample_embed_dim=self.sample_qkv_embed_dim)
        self.proj.set_sample_config(sample_in_dim=self.sample_qkv_embed_dim, sample_out_dim=self.super_embed_dim)


    def calc_sampled_param_num(self):
        return 0


    def get_complexity(self, sequence_length):
        total_flops = 0

        total_flops += self.q.get_complexity(sequence_length)
        total_flops += self.kv.get_complexity(sequence_length)

        total_flops += self.sr.get_complexity()

        total_flops += self.sample_pooling_dim * self.sample_pooling_dim * self.sample_qkv_embed_dim
        total_flops += self.proj.get_complexity(sequence_length)

        return total_flops


    def forward(self, x, H, W):
        B, N, C = x.shape
  
        q = self.q(x).reshape(B, N, self.num_heads, self.sample_qkv_embed_dim // self.num_heads).permute(0, 2, 1, 3)
    
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_ = F.adaptive_avg_pool2d(x_, self.sample_pooling_dim)
        x_ = self.sr(x_)
        x_ = x_.reshape(B, C, -1).permute(0, 2, 1)

        x_ = self.norm(x_)
        x_ = self.act(x_)

        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.sample_qkv_embed_dim // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.sample_qkv_embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
