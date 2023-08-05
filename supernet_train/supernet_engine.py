import datetime
import os
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import yaml
from pathlib import Path
import pandas as pd

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler

from .supernet_train import train_one_epoch, evaluate

from model.supernet import PVTSuper

from dataloader_n_aug.dataloader import get_balance_train_data, get_test_data
from teacher_model.teacher_model import COVID_PVTv2, load_checkpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.utils import *
from utils.metric import MetricTracker

from omegaconf import OmegaConf
from timm.utils.model import unwrap_model


def select_scheduler_optimizer(model, config):
    config = config['model_PVT_V2']
    opt = config['optimizer']['type']
    lr = config['optimizer']['lr']
    dec = config['optimizer']['weight_decay']
    optimizer = None

    if (opt == 'AdamW'):
        print("Create optimizer Adam with lr: ", lr)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=dec)

    elif (opt == 'SGD'):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    scheduler = ReduceLROnPlateau(optimizer, factor=config['scheduler']['scheduler_factor'],
                                      patience=config['scheduler']['scheduler_patience'],
                                      min_lr=config['scheduler']['scheduler_min_lr'],
                                      verbose=config['scheduler']['scheduler_verbose'])
    return optimizer, scheduler




def engine():
    cwd = os.getcwd()
    config_file = 'config/config.yml'
    model_config_file = 'config/model_cfg_evolution.yml'

    metric_ftns = ['loss', 'acc']
    train_metrics = MetricTracker(*[m for m in metric_ftns], mode='train')
    valid_metrics = MetricTracker(*[m for m in metric_ftns], mode='validation')

    config = OmegaConf.load((os.path.join(cwd, config_file)))['config']
    model_cfg = OmegaConf.load((os.path.join(cwd, model_config_file)))

    seeding(config)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    cudnn.benchmark = True

    data_loader_train = get_balance_train_data(config)
    data_loader_val = get_test_data(config)

    model = PVTSuper(img_size=model_cfg.input_size,
                                    super_embed_dims=[32, 64, 160, 256],\
                                    sample_embed_dims=model_cfg.SUPERNET.EMBED_DIMS,
                                    depths=model_cfg.SUPERNET.DEPTHS,
                                    mlp_ratios=model_cfg.SUPERNET.MLP_RATIOS,
                                    qkv_bias=True,
                                    drop_rate=model_cfg.drop_rate,
                                    attn_drop_rate=model_cfg.attn_drop_rate,
                                    num_classes=config.dataset.num_classes,
                                    num_stages=4)
                                    

    choices = {'embed_dims': model_cfg.SUPERNET.EMBED_DIMS, 'depths': model_cfg.SUPERNET.DEPTHS,
               'sample_pooling_dim': model_cfg.SUPERNET.SAMPLE_POOLING_DIM , 'mlp_ratios': model_cfg.SUPERNET.MLP_RATIOS}  

    model.to(device)

    if config.teacher_model:
        teacher_model = COVID_PVTv2(config)
        model_path = os.path.join(cwd, "teacher_model/model_best_checkpoint.pth")
        
        teacher_model = load_checkpoint(model_path, teacher_model)
        teacher_model.to(device)
        
        teacher_loss = LabelSmoothingCrossEntropy(smoothing=0.05)
    
    else:
        teacher_model = None
        teacher_loss = None

    start_epoch = 0
    optimizer, lr_scheduler = select_scheduler_optimizer(model, config)
    loss_scaler = NativeScaler()

    criterion = LabelSmoothingCrossEntropy(smoothing=0.05)

    output_dir = os.path.join(cwd, "output_dir")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    retrain_config = None

    if config.mode == 'super':  
        config_ = choices
        model = unwrap_model(model)
        model.set_sample_config(config=config_)

        print(' - SUPERNET config:', choices)
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(' - SUPERNET number of params:', parameters)  

    if config.mode == 'retrain':
        retrain_config = {'embed_dims': model_cfg.RETRAIN.EMBED_DIMS, 'depths': model_cfg.RETRAIN.DEPTHS,
                            'sample_pooling_dim': model_cfg.RETRAIN.SAMPLE_POOLING_DIM}

        model = unwrap_model(model)
        model.set_sample_config(config=retrain_config)

        print(" - SAMPLED model config: {}".format(retrain_config))
        parameters = model.get_sampled_params_numel(retrain_config)
        print(" - SAMPLED model parameters: {}".format(parameters))
        
    if config.resume_training:
        checkpoint_loader = torch.load(config.resume_training_dir, map_location='cpu')
        print('!!! RESUME TRAINING !!!')
        print('Checkpoint model dist contains: ',checkpoint_loader.keys())

        checkpoint = model.state_dict()

        for k, v in checkpoint_loader['model'].items():
            k = k.replace('module.', '')
            if k not in checkpoint.copy().keys() or\
                  (checkpoint[k].shape != checkpoint_loader['model'][k].shape):
                 continue
            checkpoint.update({k: v})

        model.load_state_dict(checkpoint, strict=True)

        if not os.path.exists(config.resume_training):
            raise ("File doesn't exist {}".format(config.resume_training))

        optimizer.load_state_dict(checkpoint_loader['optimizer'])
        lr_scheduler.load_state_dict(checkpoint_loader['lr_scheduler'])
        start_epoch = checkpoint_loader['epoch']
        loss_scaler.load_state_dict(checkpoint_loader['scaler'])

        # optimizer.param_groups[0]['lr'] = 1.875e-5
        # lr_scheduler.optimizer.param_groups[0]['lr'] = 1.875e-5

        print('Current learning rate: ', lr_scheduler.optimizer.param_groups[0]['lr'])

    if config.validation:
        _, val_acc, _, _, _ = evaluate(data_loader_val, model, device, valid_metrics=valid_metrics, mode=config.mode, choices=choices,\
                                                     retrain_config=retrain_config, config=config)
        print(f"- Accuracy of the network on test images: {val_acc:.4f}")
        return


    print("!!! Start training !!!")
    start_time = time.time()
    max_accuracy = 0.0
    
    log_stats = {'train_acc': [],
                     'test_acc': [],
                     'epoch': [],
                     'n_parameters': [],
                     'sensitivity': [],
                     'pos_pred_val': []}

    for epoch in range(start_epoch, config.epochs):
        train_loss, train_acc = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            teacher_model=teacher_model,
            train_metrics=train_metrics,
            teach_loss=teacher_loss,
            choices=choices, 
            config=config
        )

        lr_scheduler.step(epoch)
        print(f"- Accuracy of the network on train images: {train_acc:.4f}")

        val_loss, val_acc, s, ppv, n_parameters = evaluate(data_loader_val, model, device, valid_metrics, epoch,\
                         mode=config.mode, choices=choices, retrain_config=retrain_config, config=config)
        print(f"- Accuracy of the network on test images: {val_acc:.4f}")

        if max_accuracy < val_acc:
            checkpoint_path = os.path.join(output_dir, 'supermodel_checkpoint.pth')
            state =  {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch+1,
                        'scaler': loss_scaler.state_dict()}
            torch.save(state, checkpoint_path)

        max_accuracy = max(max_accuracy, val_acc)
        print(f' ** Max accuracy: {max_accuracy:.4f}')

        s = pd.Series(s).to_json(orient='values')
        ppv = pd.Series(ppv).to_json(orient='values')

        log_stats['train_acc'].append(train_acc)
        log_stats['test_acc'].append(val_acc)
        log_stats['epoch'].append(epoch)
        log_stats['n_parameters'].append(n_parameters)
        log_stats['sensitivity'].append(s)
        log_stats['pos_pred_val'].append(ppv)
        
        f = open(os.path.join(output_dir, 'log.txt'), "w")
        f.write(json.dumps(log_stats))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        print('-'*70)
