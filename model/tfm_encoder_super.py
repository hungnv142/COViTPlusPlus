import math
import torch

import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import math

from .linear_super import LinearSuper
from .lrsa_super import AttentionSuper
from .layernorm_super import LayerNormSuper
from .embedding_super import OverlapPatchembedSuper
from .mlp_super import MlpSuper



class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        # the configs of super arch of the encoder, three dimension [embed_dim, mlp_ratio, and num_heads]
        self.super_embed_dim = dim
        self.super_num_heads = num_heads
        self.super_attn_dropout = attn_drop
        self.super_dropout = drop

        self.sample_num_heads_this_layer = None
        self.sample_attn_dropout = None
        self.sample_dropout = None

        self.is_identity_layer = None

        self.norm1 = LayerNormSuper(dim)

        self.attn = AttentionSuper(dim,\
                            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,\
                            attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = LayerNormSuper(dim)

        self.mlp = MlpSuper(super_in_features=dim, hidden_features=int(dim*mlp_ratio),\
                                                    out_features=dim, act_layer=act_layer, drop=drop)
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


    def set_sample_config(self, is_identity_layer, sample_qkv_embed_dim=None, sample_pooling_dim=None,\
                        sample_dropout=None, sample_attn_dropout=None):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_qkv_dim_this_layer = sample_qkv_embed_dim

        self.sample_attn_dropout = sample_dropout
        self.sample_dropout = sample_attn_dropout
        self.sample_pooling_dim = sample_pooling_dim

        self.norm1.set_sample_config(sample_embed_dim=self.super_embed_dim)
        self.attn.set_sample_config(sample_qkv_embed_dim=self.sample_qkv_dim_this_layer, \
                            attn_drop=self.sample_attn_dropout, drop=self.sample_dropout, sample_pooling_dim=self.sample_pooling_dim)

        self.norm2.set_sample_config(sample_embed_dim=self.super_embed_dim)


    def get_complexity(self, sequence_length):
        if self.is_identity_layer:
            return 0.0

        total_flops = 0

        total_flops += self.norm1.get_complexity(sequence_length)
        total_flops += self.attn.get_complexity(sequence_length)
        total_flops += self.norm2.get_complexity(sequence_length)
        total_flops += self.mlp.get_complexity(sequence_length)

        return total_flops


    def forward(self, x, H, W):
        if self.is_identity_layer:
            return x

        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x), H, W)

        return x
