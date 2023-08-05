import argparse
import torch
import math 

from .utils import blockPrinting
from model.supernet import PVTSuper


def li_sra_flops(h, w, dim, pooling_dim):
    return 2 * h * w * pooling_dim * pooling_dim * dim


@blockPrinting
def get_flops(model, input_shape):
    C, H, W = input_shape
    input = torch.rand(1, C, H, W)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    input = input.to(device)
    model.to(device)
    
    model.eval()
    model(input)

    flops = model.get_complexity()

    stage1 = li_sra_flops(H // 4, W // 4,
                                  model.block1[0].attn.dim, model.block1[0].attn.sample_pooling_dim) * len(model.block1)
    stage2 = li_sra_flops(H // 12, W // 12,
                                  model.block2[0].attn.dim, model.block1[0].attn.sample_pooling_dim) * len(model.block2)
    stage3 = li_sra_flops(H // 24, W // 24,
                                  model.block3[0].attn.dim, model.block1[0].attn.sample_pooling_dim) * len(model.block3)
    stage4 = li_sra_flops(H // 48, W // 48,
                                  model.block4[0].attn.dim, model.block1[0].attn.sample_pooling_dim) * len(model.block4)

    flops += stage1 + stage2 + stage3 + stage4

    return math.floor(flops) / 10**9
