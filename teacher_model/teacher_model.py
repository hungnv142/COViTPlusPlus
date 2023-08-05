import numpy as np
import os
import torch
from .pvt_v2 import PyramidVisionTransformerV2
from torch import nn
from functools import partial
from torchvision import models


class COVID_PVTv2(nn.Module):

    def __init__(self, config):
        super(COVID_PVTv2, self).__init__()
        self.config = config

        self.pvt = PyramidVisionTransformerV2()
        self.embed_dim = 512

        self.classifier = nn.Sequential(nn.Linear(self.embed_dim, 64), nn.GELU(), nn.BatchNorm1d(64),\
                            nn.Dropout(0.5),\
                            nn.Linear(64, 16),nn.GELU(), nn.BatchNorm1d(16),\
                            nn.Dropout(0.3),\
                            nn.Linear(16, self.config.dataset.num_classes))

    def forward(self, x):
        x = self.pvt(x)
        x = self.classifier(x)
        return x



def load_checkpoint(checkpoint, model):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    """
    checkpoint_loader = torch.load(checkpoint, map_location='cpu')
    print('Teacher model checkpoint dist contains: ',checkpoint_loader.keys())

    my_state_dict = model.state_dict()
    for k, v in checkpoint_loader['state_dict'].items():
        k = k.replace('module.', '')
        my_state_dict.update({k: v})

    model.load_state_dict(my_state_dict, strict=True)

    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))

    return model
