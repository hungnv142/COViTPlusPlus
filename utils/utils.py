import io
import os, sys
import time
import yaml
from collections import defaultdict, deque
import datetime
from easydict import EasyDict as edict

import torch
import torch.distributed as dist
import numpy as np

cfg = edict()



def read_filepaths(file, root):
    """Collect data from annotation files.
    """
    print('Collecting data from : {}'. format(file))

    paths, labels = [], []

    with open(file, 'r') as f:
        lines = f.read().splitlines()
        for idx, line in enumerate(lines):
            if len(line.split(' ')) == 3:
                _, path, label = line.split(' ')
            elif len(line.split(' ')) == 4:
                _, path, label, dataset = line.split(' ')
            elif len(line.split(' ')) == 5:
                _, _, path, label, dataset = line.split(' ')
            elif len(line.split(' ')) == 6:
                _, _, path1, path2, label, dataset = line.split(' ')
                path = path1 + path2

            label = label.lower()

            # Ignore duplicates filenames
            if path in paths:
                continue

            # Ignore not exists filenames
            img_path = root + path
            if not os.path.exists(img_path):
                continue

            paths.append(path)
            labels.append(label)

    print('Data collected! {}'. format(file))

    return paths, labels


def update_config_from_file(filename):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        _update_config(cfg, exp_config)


def seeding(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    SEED = config.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if (config.cuda):
        torch.cuda.manual_seed(SEED)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

    
def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def blockPrinting(func):
    def func_wrapper(*args, **kwargs):
        # block all printing to the console
        sys.stdout = open(os.devnull, 'w')
        # call the method in question
        value = func(*args, **kwargs)
        # enable all printing to the console
        sys.stdout = sys.__stdout__
        # pass the return value of the method back
        return value

    return func_wrapper