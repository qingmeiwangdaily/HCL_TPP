import pickle
import time
import numpy
import random
import os
import datetime

import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from anhp.utils.log import LogWriter, LogReader
from anhp.eval.sigtest import Bootstrapping
from anhp.esm.manager import Manager

import argparse

__author__ = 'Hongyuan Mei, Chenghao Yang'


class Tester(Manager):

    def __init__(self, args):
        tic = time.time()
        super(Tester, self).__init__(args)
        self.model.load_state_dict(torch.load(args.PathModel))
        self.create_thinningsampler(args.NumSample, args.NumExp)
        print(f"time spent on initializatin : {time.time() - tic:.2f}")

    def run(self):
        print(f"start testing on {self.args.Split} data of domain {self.args.Domain} ... ")
        bs = Bootstrapping()
        if self.args.Split == "train":
            dataloader = self.train_loader
        elif self.args.Split == "dev":
            dataloader = self.dev_loader
        else:
            dataloader = self.test_loader

        tic = time.time()
        # testing with loglik
        print("testing on loglik")
        total_loglik, total_acc, (event_ll, non_event_ll), total_num_token, total_num_event, all_res_loglik, all_res_loglik_token, \
        all_type_ll_token, all_time_ll_token = \
            self.run_one_iteration(self.model, dataloader, "eval", None)
        avg_loglik = total_loglik / total_num_token
        # significant tests with loglik
        print(f"loglik is {avg_loglik:.4f}")
        print(f"acc is {total_acc:.4f}")
        print(f"time is {(time.time() - tic):.2f}")
        print("bootstrapping for significant tests")
        bs.run(all_res_loglik)

        all_res = {
            'loglik': all_res_loglik,
            'loglik_token': all_res_loglik_token,
            "type_ll_token": all_type_ll_token,
            "time_ll_token": all_time_ll_token,
            'pred': None
        }
        # all_res = {}

        if self.args.Predict:
            tic = time.time()
            print("testing on prediction")
            # testing with prediction
            # get prediction and ground truth
            all_res_pred = self.run_prediction(self.model, dataloader)
            # finish
            all_res['pred'] = all_res_pred
        with open(self.args.PathResult, 'wb') as f:
            pickle.dump(all_res, f)
        print("\ntesting finished")


