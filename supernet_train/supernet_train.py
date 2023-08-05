import math
import sys
import random
import time
import numpy as np
from tqdm import tqdm as tqdm

from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

# from utils import utils
from utils.metric import MetricTracker, sensitivity, positive_predictive_value



def sample_configs(choices):
    config = {}
    
    config['sample_pooling_dim'] = random.choice(choices['sample_pooling_dim'])
    config['embed_dims'] = random.choice(choices['embed_dims'])
    config['depths'] = random.choice(choices['depths'])

    return config



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    teacher_model: torch.nn.Module = None, train_metrics = None,
                    teach_loss: torch.nn.Module = None, choices=None, config=None):
    model.train()
    criterion.train()
    torch.manual_seed(48)
    torch.autograd.set_detect_anomaly(True)

    len_epoch = config.dataloader.train.batch_size * len(data_loader)
    confusion_matrix = torch.zeros(config.dataset.num_classes, config.dataset.num_classes)

    train_metrics.reset()   
    
    for batch_idx, (samples, targets) in enumerate(tqdm(data_loader, desc=f'Training for eps: {epoch}')):
        optimizer.zero_grad()

        samples = samples.to(device, non_blocking=True)
        ground_truth = torch.tensor(targets, dtype=torch.long).to(device, non_blocking=True)

        outputs = model(samples)

        if teacher_model:
            with torch.no_grad():
                teach_output = teacher_model(samples)
            _, teacher_label = teach_output.topk(1, 1, True, True)
            loss = 1 / 2 * criterion(outputs, ground_truth) + 1 / 2 * teach_loss(outputs, teacher_label.squeeze())
        else:
            loss = criterion(outputs, ground_truth)
        
        loss.backward()
        optimizer.step()
        
        predictions = torch.tensor(torch.argmax(outputs, dim=1), dtype=torch.float32)
        accuracy = np.sum(predictions.cpu().numpy() == targets.cpu().numpy())

        writer_step = (epoch - 1) * len_epoch + batch_idx

        train_metrics.update(key='loss', value= loss.item(), n=1, writer_step=writer_step)
        train_metrics.update(key='acc', value= accuracy, n=targets.size(0), writer_step=writer_step)

        for tar, pred in zip(targets.cpu().view(-1), predictions.cpu().view(-1)):
            confusion_matrix[tar.long(), pred.long()] += 1
        _progress(config, batch_idx, epoch, len_epoch, metrics=train_metrics, mode='train')
    
    _progress(config, batch_idx, epoch, len_epoch, metrics=train_metrics, mode='train', print_summary=True)

    train_loss, train_acc = train_metrics.avg('loss'), train_metrics.avg('acc')

    return train_loss, train_acc



@torch.no_grad()
def evaluate(data_loader, model, device, valid_metrics, epoch=0, mode='super', choices=None, retrain_config=None, config=None):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    len_epoch = config.dataloader.train.batch_size * len(data_loader)
    confusion_matrix = torch.zeros(config.dataset.num_classes, config.dataset.num_classes)

    if mode == 'super':
        config_ = choices
        model = unwrap_model(model)
        model.set_sample_config(config=config_)

        print(' - SUPERNET config:', choices)
        parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(' - SUPERNET number of params:', parameters)

    elif mode == 'retrain':
        config_ = retrain_config
        model = unwrap_model(model)
        model.set_sample_config(config=config_)

        print(" - Sampled model config: {}".format(config_))
        parameters = model.get_sampled_params_numel(config_)
        print(" - Sampled model parameters: {}".format(parameters))

    else:
        raise RuntimeError('INVALID CONFIGURATION OF VALID MODE')
    

    for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc=f'Evaluate for epoch {epoch}')):
        images = images.to(device, non_blocking=True)
        ground_truths = torch.tensor(targets, dtype=torch.long).to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, ground_truths)

        predictions = torch.tensor(torch.argmax(outputs, dim=1), dtype=torch.float32)
        accuracy = np.sum(predictions.cpu().numpy() == targets.cpu().numpy())

        writer_step = (epoch - 1) * len_epoch + batch_idx

        valid_metrics.update(key='loss', value= loss.item(), n=1, writer_step=writer_step)
        valid_metrics.update(key='acc', value= accuracy, n=targets.size(0), writer_step=writer_step)
        
        for tar, pred in zip(targets.cpu().view(-1), predictions.cpu().view(-1)):
            confusion_matrix[tar.long(), pred.long()] += 1
        # _progress(config, batch_idx, epoch, len_epoch, metrics=valid_metrics, mode='validation')
    
    _progress(config=config, batch_idx=0, epoch=epoch, len_epoch=len_epoch, metrics=valid_metrics, mode='validation', print_summary=True)

    s = sensitivity(confusion_matrix.numpy())
    ppv = positive_predictive_value(confusion_matrix.numpy())

    print(f"Sens: {s} , PPV: {ppv}")

    val_loss, val_acc = valid_metrics.avg('loss'), valid_metrics.avg('acc')

    return val_loss, val_acc, s, ppv, parameters



def _progress(config, batch_idx, epoch, len_epoch, metrics, mode='', print_summary=False):
    metrics_string = metrics.calc_all_metrics()
    log_step = config.log_interval
    iter = batch_idx * config.dataloader.train.batch_size

    if (iter % log_step == 0 and iter !=0):
            print(f"Train Sample [{batch_idx * config.dataloader.train.batch_size:5d}/{len_epoch:5d}]\t {metrics_string}")
    elif print_summary:
        print(f'{mode} Summary  Epoch: [{epoch}/{config.epochs}]\t {metrics_string}')