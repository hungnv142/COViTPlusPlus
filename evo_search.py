import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import json
import os
import sys
import random
import copy
import pickle
import itertools
from pathlib import Path

from timm.models import create_model

from utils.metric import MetricTracker
from utils.flops import get_flops
from utils.utils import blockPrinting, get_rank

from supernet_train.supernet_train import evaluate


def decode_cand_tuple(cand_tuple):
    # sample_pooling_dim, depths, embed_dims
    return cand_tuple[0], list(cand_tuple[1:5]), list(cand_tuple[5:])


def decode_embed_dims(embed_dim):
    pass


class EvolutionSearch(object):
    def __init__(self, config, main_config, model, choices, test_loader):
        if (main_config.cuda):
            use_cuda = torch.cuda.is_available()
            self.device = torch.device("cuda" if use_cuda else "cpu")
        else:
            self.device = torch.device("cpu")
        
        self.config = config
        self.main_config = main_config

        self.model = model
        self.max_epochs = config.max_epochs

        self.select_num = config.select_num
        self.population_num = config.population_num
        
        self.m_prob = config.m_prob
        self.s_prob = config.s_prob

        self.crossover_num = config.crossover_num
        self.mutation_num = config.mutation_num

        self.parameters_limits = config.param_limits
        self.min_parameters_limits = config.min_param_limits

        self.output_dir = config.evo_output_dir

        self.mlp_ratios = config.SUPERNET.MLP_RATIOS

        self.test_loader = test_loader

        metric_ftns = ['loss', 'acc']
        self.valid_metrics = MetricTracker(*[m for m in metric_ftns], mode='validation')

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 30: []}
        self.epoch = 0
        self.checkpoint_path = config.resume_dir
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []

        self.choices = choices
    
    

    def save_checkpoint(self):
        info = {}
        info['top_accuracies'] = self.top_accuracies
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        checkpoint_path = os.path.join(self.output_dir, "{}-checkpoint-{}.pth".format(self.mlp_ratios, self.epoch))
        torch.save(info, checkpoint_path)
        print(' ** Save checkpoint to', checkpoint_path)
    


    def load_checkpoint(self):
        assert os.path.exists(self.checkpoint_path), f"Checkpoint {self.checkpoint_path} does not exist."
         
        info = torch.load(self.checkpoint_path)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('Load checkpoint from  ', self.checkpoint_path)
        return True
    


    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]

        if 'visited' in info:
            return False
        
        if len(cand) < 9:
            return False
        
        sample_pooling_dim, depths, embed_dims = decode_cand_tuple(cand)

        sampled_config = {}
        sampled_config['depths'] = depths
        sampled_config['embed_dims'] = embed_dims
        sampled_config['sample_pooling_dim'] = sample_pooling_dim
        
        n_parameters = self.model.get_sampled_params_numel(sampled_config)

        info['params'] =  n_parameters / 10.**6

        if info['params'] > self.parameters_limits:
            print('!!! Parameters limit exceed ( {} ) !!!'.format(info['params']))
            return False

        if info['params'] < self.min_parameters_limits:
            print('!!! Under minimum parameters limit ( {} ) !!!'.format(info['params']))
            return False
        
        print("Rank:", get_rank(), cand, info['params'])

        val_loss, val_acc, s, ppv , _ = evaluate(self.test_loader, self.model, self.device, self.valid_metrics, epoch=self.epoch,\
                         mode='retrain', choices=self.choices, retrain_config=sampled_config, config=self.main_config)
        
        info['test_acc'] = val_acc
        info['covid_sens'] = s[2]
        info['flops (G)'] = get_flops(self.model, (3, self.main_config.dataset.img_size[0], self.main_config.dataset.img_size[1]))

        info['visited'] = True

        return True



    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
    
        print('Select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]


    
    def stack_random_cand(self, random_func, *, batchsize=10):

        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand


    def get_random_cand(self):
        cand_tuple = list()

        cand_tuple.append([random.choice(self.choices['sample_pooling_dim'])])
        cand_tuple.append(random.choice(self.choices['depths']))    
        cand_tuple.append(random.choice(self.choices['embed_dims']))

        return tuple(itertools.chain(*cand_tuple))


    def get_random(self, num):
        print('Random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            print('+  random {}/{}'.format(len(self.candidates), num))
        print('+  random_num = {}'.format(len(self.candidates)))



    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        assert k in self.keep_top_k
        print('Mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            sample_pooling_dim, depths, embed_dims = decode_cand_tuple(cand)

            cand_tuple = list()

            random_s = random.random()
            if random_s < s_prob:
                cand_tuple.append([random.choice(self.choices['sample_pooling_dim'])])

            random_s = random.random()
            if random_s < m_prob:
                cand_tuple.append(random.choice(self.choices['depths']))
            
            random_s = random.random()
            if random_s < s_prob:
                cand_tuple.append(random.choice(self.choices['embed_dims']))

            return tuple((itertools.chain(*cand_tuple)))

        cand_iter = self.stack_random_cand(random_func)

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)

            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('+ mutation {}/{}'.format(len(res), mutation_num))

        print('Mutation_num = {}'.format(len(res)))

        return res



    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('Crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            max_iters_tmp = 30

            while len(p1) != len(p2) and max_iters_tmp > 0:
                max_iters_tmp -= 1
                p1 = random.choice(self.keep_top_k[k])
                p2 = random.choice(self.keep_top_k[k])
            
            return tuple(random.choice([i, j]) for i, j in zip(p1, p2))
        
        cand_iter = self.stack_random_cand(random_func)

        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('+  crossover {}/{}'.format(len(res), crossover_num))

        print('Crossover_num = {}'.format(len(res)))
        return res


    #@blockPrinting
    def search(self):
        print(
            ' + population_num = {} \n + select_num = {} \n + mutation_num = {} \n + crossover_num = {} \n + random_num = {} \n + max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        
        if self.config.load_checkpoint == True:
            self.load_checkpoint()

        else:
            self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            print(' +   Epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['test_acc'])
            self.update_top_k(
                self.candidates, k=30, key=lambda x: self.vis_dict[x]['test_acc'])

            print(' +   Epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[30])))
            tmp_accuracy = []

            for i, cand in enumerate(self.keep_top_k[30]):
                print('No.{} {} ; Top-1 test acc = {}, params = {}, covid-19-sens = {}, flops = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['test_acc'], self.vis_dict[cand]['params'], self.vis_dict[cand]['covid_sens'],\
                                                                                         self.vis_dict[cand]['flops (G)']))
                tmp_accuracy.append(self.vis_dict[cand]['test_acc'])
            self.top_accuracies.append(tmp_accuracy)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob, self.s_prob)

            crossover = self.get_crossover(self.select_num, self.crossover_num)

            if self.epoch < 1:
                self.get_random(self.population_num)

            self.candidates = mutation + crossover

            self.epoch += 1

            self.save_checkpoint()