import math
import torch

import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

import math

from .linear_super import LinearSuper
from .lrsa_super import AttentionSuper
from .layernorm_super import LayerNormSuper
from .embedding_super import OverlapPatchembedSuper
from .mlp_super import MlpSuper
from .tfm_encoder_super import TransformerEncoderLayer
from .last_mlp import LastMLP


class PVTSuper(nn.Module):
    def __init__(self, img_size, num_classes, in_chans=3, super_embed_dims=[32, 64, 160, 256], sample_embed_dims=None, num_heads=[1, 2, 5, 8],\
                    mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,\
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2,2,2,2], num_stages=4):
        super(PVTSuper, self).__init__()

        self.num_classes = num_classes
        self.num_stages = num_stages
        self.num_classes = num_classes

        self.super_embed_dims = super_embed_dims
        self.super_mlp_ratios = mlp_ratios
        self.super_depths = depths
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate

        self.sample_embed_dims = sample_embed_dims
        self.sample_depths = None

        self.sample_dropout = None
        self.sample_attn_dropout = None

        self.sequence_length = []
        
        patch_sizes = [7, 5, 3, 3]
        strides = [4, 3, 2, 2]

        for i in range(num_stages):
            patch_embed = OverlapPatchembedSuper(img_size = img_size if i == 0 else img_size[0] // (2 ** (i + 1)),
                                            patch_size=patch_sizes[i],
                                            stride=strides[i],
                                            in_chans=in_chans if i == 0 else super_embed_dims[i-1],
                                            embed_dim=super_embed_dims[i])

            block = nn.ModuleList([TransformerEncoderLayer(dim=super_embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], \
                                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate) for j in range(depths[i])])

            norm = norm_layer(super_embed_dims[i])

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.head = LastMLP(num_classes=self.num_classes, embed_dim=super_embed_dims[-1])

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def get_classifier(self):
        return self.head


    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.super_embed_dims, num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            B, C, H, W = x.shape
            self.sequence_length.append(H*W)

            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return x.mean(dim=1)


    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


    def set_sample_config(self, config: dict):
        self.sample_embed_dims = config['embed_dims']
        self.sample_depths = config['depths']
        self.sample_pooling_dim = config['sample_pooling_dim']

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            for super_depth in range(self.super_depths[i]):
                if super_depth < self.sample_depths[i]:
                    sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dims[i], self.super_embed_dims[i])
                    sample_attn_dropout = calc_dropout(self.super_attn_dropout, self.sample_embed_dims[i], self.super_embed_dims[i])

                    for layer in block:
                        layer.set_sample_config(is_identity_layer=False,\
                                            sample_qkv_embed_dim=self.sample_embed_dims[i],\
                                            sample_pooling_dim = self.sample_pooling_dim,\
                                            sample_dropout=sample_dropout,\
                                            sample_attn_dropout=sample_attn_dropout)
                else: 
                    for layer in block:
                        layer.set_sample_config(is_identity_layer=True)



    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):
                index = list(name.split('.')[0])
                if index != []:
                    index = int(index[-1])
                    if f'block{index}' in name.split('.')[0]:
                        if int(name.split('.')[1]) < config['depths'][int(index)-1]:
                            numels.append(module.calc_sampled_param_num())
                    else:
                        numels.append(module.calc_sampled_param_num())
            elif hasattr(module, 'calc_sampled_param_num_head'):
                numels.append(module.calc_sampled_param_num_head())

        return sum(numels)



    def get_complexity(self):
        assert len(self.sequence_length) != 0 

        total_flops = 0
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            total_flops += patch_embed.get_complexity(self.sequence_length[i])
            total_flops += sum([module.get_complexity(self.sequence_length[i]) for module in block])
            total_flops += self.sequence_length[i] * self.super_embed_dims[i]

        total_flops += self.head.get_complexity(self.super_embed_dims[-1])

        return total_flops


def calc_dropout(dropout, sample_embed_dims, super_embed_dims):
    return dropout * 1.0 * sample_embed_dims / super_embed_dims
