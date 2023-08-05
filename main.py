import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import json
import os
import sys

from pathlib import Path

from timm.data import Mixup
from timm.data.loader import MultiEpochsDataLoader
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEmaV2


model_kwconfig = {'model_name': config.model, 'pretrained': False,\
                'num_classes': config.nb_classes, 'drop_rate': config.drop,\
                'drop_path_rate': config.drop_path, 'drop_block_rate': config.drop_block}

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model.to(device)

if config.finetune:
    state_dict = load_interpolated_state_dict(model_state_dict=model.state_dict(), ckpt_path=config.finetune_path)
    model.load_state_dict(state_dict)

model = create_model(**model_kwconfig)
