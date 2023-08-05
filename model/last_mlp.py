import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from mmcv.cnn.utils.flops_counter import get_model_complexity_info


import math

class LastMLP(nn.Sequential):
    def __init__(self, num_classes, embed_dim):
        super(LastMLP, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.classifier = nn.Sequential(nn.Linear(self.embed_dim, 64), nn.GELU(), nn.BatchNorm1d(64),\
                                nn.Dropout(0.5),\
                                nn.Linear(64, 16),nn.GELU(), nn.BatchNorm1d(16),\
                                nn.Dropout(0.3),\
                                nn.Linear(16, self.num_classes))
        self.apply(self._init_weights)


    def get_complexity(model, sequence_length):
        flops, _ = get_model_complexity_info(model, (sequence_length, ), print_per_layer_stat=False, as_strings=False)
        return flops
    
    
    def calc_sampled_param_num_head(self):
        return sum(p.numel() for p in self.classifier.parameters())


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
        return self.classifier(x)
