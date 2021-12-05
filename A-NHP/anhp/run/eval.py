# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
eval neural Datalog through time (NDTT)
@author: hongyuan
"""

import pickle
import time
import numpy
import random
import os
import datetime
from itertools import chain
from collections import OrderedDict

import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from anhp.utils.log import LogReader, LogWriter
from anhp.eval.sigtest import Bootstrapping

import argparse
__author__ = 'Hongyuan Mei'


class Eval(object):

    def __init__(self, args):
        tic = time.time()
        self.args = args
        self.logger = LogWriter(self.args.EvalLogPath, vars(self.args))
        self.report(f"time spent on initializatin : {time.time()-tic:.2f}")

    def macro_avg(self, x): 
        num, denom = 0.0, 0.0 
        for num_i, denom_i in x: 
            num += num_i
            denom += denom_i
        return num / denom
    
    def report(self, msg):
        self.logger.checkpoint(msg + "\n")
        print(msg)

    #@profile
    def run(self):
        self.report(f"start evaluting on {self.args.Split} data of domain {self.args.Domain} ... ")
        bs = Bootstrapping()

        self.report(f"reading saved predictions and ground-truth ... ")
        with open(self.args.PathResult, 'rb') as f:
            all_res = pickle.load(f)

        # eval with loglik 
        self.report("eval on loglik")
        avg_loglik = self.macro_avg(all_res['loglik'])
        # significant tests with loglik 
        self.report(f"loglik is {avg_loglik:.4f}")
        self.report("bootstrapping for significant tests")
        bs.run(all_res['loglik'])

        if 'loglik_token' in all_res: 
            self.report(f"\ntoken-level loglik tracked")
            self.report(f"avg. token-level loglik is {self.macro_avg(all_res['loglik_token']):.4f}")
            self.report(f"bootstrapping for significant tests")
            bs.run(all_res['loglik_token'])

        if 'type_ll_token' in all_res:
            self.report(f"\ntoken-level type-loglik tracked")
            self.report(f"avg. token-level loglik is {self.macro_avg(all_res['type_ll_token']):.4f}")
            self.report(f"bootstrapping for significant tests")
            bs.run(all_res['type_ll_token'])

        if 'time_ll_token' in all_res:
            self.report(f"\ntoken-level time-loglik tracked")
            self.report(f"avg. token-level loglik is {self.macro_avg(all_res['time_ll_token']):.4f}")
            self.report(f"bootstrapping for significant tests")
            bs.run(all_res['time_ll_token'])

        self.report("\neval on prediction")
        # testing with prediction 
        recall_percents = [0.05, 0.10]
        # get prediction and ground truth 
        all_res_pred = all_res['pred']
        # compute and store square error and error # for each seq 
        # to normalize by total variance/deviation 
        # we need to get avg. dt first 
        all_dtime = []
        for res_seq in all_res_pred: 
            for prediction_and_groundtruth in res_seq: 
                _, _, _, _, next_event_dtime, next_event_name = prediction_and_groundtruth
                if next_event_name != 'eos': 
                    all_dtime.append(next_event_dtime)
        avg_dtime = numpy.mean(all_dtime)
        # compute and store : 
        # square error normalized by total variance 
        # absolute err normalized by total deviation 
        # recall at X-% for X-% in percents 
        """
        these are seq-level collection 
        """
        se_uncond = [] # se_cond = []
        err_cond = [] # err_uncond = []
        sen_uncond = [] # sen_cond = []
        aen_uncond = [] # aen_cond = []
        """
        these are token-level collection
        """
        se_uncond_token = [] # se_cond = []
        err_cond_token = [] # err_uncond = []
        sen_uncond_token = [] # sen_cond = []
        aen_uncond_token = [] # aen_cond = []
        """
        don't use recall at prec or mrr any more 
        we now evaluate on error rate on the last argument of an event
        """
        #recall_at_prec_uncond, recall_at_prec_cond = OrderedDict(), OrderedDict()
        #for prec in recall_percents: 
        #    recall_at_prec_uncond[prec] = []
        #    recall_at_prec_cond[prec] = []
        #mrr_uncond, mrr_cond = [], []

        for res_seq in all_res_pred: 
            se_uncond_i, var_i = 0.0, 0.0 # se_cond_i = 0.0
            ae_uncond_i, dev_i = 0.0, 0.0 # ae_cond_i = 0.0
            err_cond_i = 0.0 # err_uncond_i = 0.0 
            #recall_uncond_i_prec, recall_cond_i_prec = {}, {}
            #for prec in recall_percents: 
            #    recall_uncond_i_prec[prec] = 0.0 
            #    recall_cond_i_prec[prec] = 0.0
            #mrr_uncond_i, mrr_cond_i = 0.0, 0.0 
            cnt = 0.0
            for prediction_and_groundtruth in res_seq: 
                time_uncond, dtime_uncond, top_event_names, \
                next_event_time, next_event_dtime, next_event_name = prediction_and_groundtruth
                
                if next_event_name != 'eos': 

                    """
                    collect delta 
                    """
                    delta_se_uncond = (dtime_uncond - next_event_dtime) ** 2 
                    delta_var = (avg_dtime - next_event_dtime) ** 2
                    delta_ae_uncond = abs(dtime_uncond - next_event_dtime)
                    delta_dev = abs(avg_dtime - next_event_dtime)
                    delta_err_cond = 1.0 if top_event_names[0] != next_event_name else 0.0
                    """
                    accumulate for sequence level
                    """
                    # square err
                    se_uncond_i += delta_se_uncond
                    #se_cond_i += (dtime_cond - next_event_dtime) ** 2
                    var_i += delta_var
                    # absolute err 
                    ae_uncond_i += delta_ae_uncond
                    #ae_cond_i += abs(dtime_cond - next_event_dtime)
                    dev_i += delta_dev
                    # error rate : 1.0 - prediction accuracy/precision
                    err_cond_i += delta_err_cond
                    # recall at X-% 
                    #for prec in recall_percents: 
                    #    k = max(1, int(prec * len(top_event_names_uncond)))
                    #    if next_event_name in top_event_names_uncond[:k]: 
                    #        recall_uncond_i_prec[prec] += 1.0 
                    #    if next_event_name in top_event_names_cond[:k]: 
                    #        recall_cond_i_prec[prec] += 1.0 
                    # mrr 
                    #mrr_uncond_i += 1.0 / (top_event_names_uncond.index(next_event_name) + 1)
                    #mrr_cond_i += 1.0 / (top_event_names_cond.index(next_event_name) + 1)
                    # count # tokens
                    cnt += 1.0 

                    """
                    accumulate for token level
                    """
                    se_uncond_token.append( (delta_se_uncond, 1.0) )
                    sen_uncond_token.append( (delta_se_uncond, delta_var) )
                    aen_uncond_token.append( (delta_ae_uncond, delta_dev) )
                    err_cond_token.append( (delta_err_cond, 1.0) )
            # square err
            se_uncond.append( (se_uncond_i, cnt) )
            #se_cond.append( (se_cond_i, cnt) )
            # square err normalized by variance
            sen_uncond.append( (se_uncond_i, var_i) )
            #sen_cond.append( (se_cond_i, var_i) )
            # absolute error normalized by deviation 
            aen_uncond.append( (ae_uncond_i, dev_i) )
            #aen_cond.append( (ae_cond_i, dev_i) )
            # error rate : 1.0 - prediction accuracy/precision
            #err_uncond.append( (err_uncond_i, cnt) )
            err_cond.append( (err_cond_i, cnt) )
            # recall at X-%
            #for prec in recall_percents: 
            #    recall_at_prec_uncond[prec].append( (recall_uncond_i_prec[prec], cnt) )
            #    recall_at_prec_cond[prec].append( (recall_cond_i_prec[prec], cnt) )
            # mrr 
            #mrr_uncond.append( (mrr_uncond_i, cnt) )
            #mrr_cond.append( (mrr_cond_i, cnt) )
        
        self.report(f"\npresenting SEQ-LEVEL results ...")

        self.report(f"\nunnormalized time eval")

        unnorm_time = {
            'MSE': se_uncond, #'MSE_cond': se_cond
        }
        for k, v in unnorm_time.items(): 
            res = self.macro_avg(v)
            self.report(f"\nresult of {k} is {res:.4f}")
            self.report("bootstrapping for significant tests")
            bs.run(v)

        self.report(f"\nnormalized time eval")

        norm_time = {
            'Normalized_SE': sen_uncond, #'Normalized_SE_cond': sen_cond, 
            'Normalized_AE': aen_uncond, #'Normalized_AE_cond': aen_cond
        }
        for k, v in norm_time.items(): 
            res = self.macro_avg(v)
            self.report(f"\nresult of {k} is {res:.4f}")
            self.report("bootstrapping for significant tests")
            bs.run(v)

        self.report(f"\nunnormalized type eval")
        
        err_rates = {
            #'Error_rate_uncond': err_uncond, 
            'Error_rate': err_cond
        }
        for k, v in err_rates.items(): 
            res = self.macro_avg(v)
            self.report(f"\n{k} is : {(100.0*res):.2f}%")
            self.report("bootstrapping for significant tests")
            bs.run(v)

        #self.report()

        self.report(f"\npresenting TOKEN-LEVEL results ...")

        self.report(f"\nunnormalized time eval")

        unnorm_time = {
            'MSE': se_uncond_token, #'MSE_cond': se_cond
        }
        for k, v in unnorm_time.items(): 
            res = self.macro_avg(v)
            self.report(f"\nresult of {k} is {res:.4f}")
            self.report("bootstrapping for significant tests")
            bs.run(v)

        self.report(f"\nnormalized time eval")

        norm_time = {
            'Normalized_SE': sen_uncond_token, 
            'Normalized_AE': aen_uncond_token,
        }
        for k, v in norm_time.items(): 
            res = self.macro_avg(v)
            self.report(f"\nresult of {k} is {res:.4f}")
            self.report("bootstrapping for significant tests")
            bs.run(v)

        self.report(f"\nunnormalized type eval")
        
        err_rates = {
            #'Error_rate_uncond': err_uncond, 
            'Error_rate': err_cond_token
        }
        for k, v in err_rates.items(): 
            res = self.macro_avg(v)
            self.report(f"\n{k} is : {(100.0*res):.2f}%")
            self.report("bootstrapping for significant tests")
            bs.run(v)
        #self.report(f"\nnormalized type eval")

        #for p in recall_percents: 
        #    res_uncond = self.macro_avg(recall_at_prec_uncond[p])
        #    res_cond = self.macro_avg(recall_at_prec_cond[p])
        #    self.report(f"\nRecall at top {100.0*p}% (unconditional) is : {(100.0*res_uncond):.2f}%")
        #    self.report(f"bootstrapping for significant tests")
        #    bs.run(recall_at_prec_uncond[p])
        #    self.report(f"\nRecall at top {100.0*p}% (conditional) is : {(100.0*res_cond):.2f}%")
        #    self.report(f"bootstrapping for significant tests")
        #    bs.run(recall_at_prec_cond[p])
        
        #mrrs = {
        #    'MRR_uncond': mrr_uncond, 
        #    'MRR_cond': mrr_cond
        #}
        #for k, v in mrrs.items(): 
        #    res = self.macro_avg(v)
        #    self.report(f"\n{k} is : {res:.4f}, 1/MRR is : {(1.0/res):.4f}")
        #    self.report("bootstrapping for significant tests")
        #    bs.run(v)


def main():

    parser = argparse.ArgumentParser(description='evaluating neural Datalog through time (NDTT)')

    parser.add_argument(
        '-d', '--Domain', required=True, type=str, help='which domain to work on?'
    )
    parser.add_argument(
        '-fn', '--FolderName', required=True, type=str,
        help='base name of the folder to store the model (and log)?'
    )
    parser.add_argument(
        '-s', '--Split', required=True, type=str, help='what split to test?',
        choices = ['train', 'dev', 'test']
    )
    parser.add_argument(
        '-ps', '--PathStorage', type=str, default='../..',
        help='Path of storage which stores domains (with data), logs, results, etc. \
            Must be local (e.g. no HDFS allowed)'
    )
    parser.add_argument(
        '-sd', '--Seed', default=12345, type=int, help='random seed'
    )
    
    args = parser.parse_args()
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()

    args.Version = torch.__version__
    args.ID = id_process
    args.TIME = time_current

    path_storage = os.path.abspath(args.PathStorage)
    args.PathDomain = os.path.join(path_storage, 'domains', args.Domain)

    if args.Domain == args.FolderName:
        """
        get testing results with ground-truth configuration, including database and model parameters...
        often used for sanity check
        """
        args.PathLog = None
        args.Database = 'gen'
        args.PathModel = os.path.join(args.PathDomain, 'gen_model')
        args.PathResult = os.path.join(args.PathDomain, f'results_gen_{args.Split}')
    else:
        path_logs = os.path.join( args.PathDomain, 'ContKVLogs', args.FolderName )
        print(path_logs)
        assert os.path.exists(path_logs)
        args.PathLog = os.path.join(path_logs, 'log.txt')
        log = LogReader(args.PathLog)
        saved_args = log.getArgs()
        """
        why do we combine local paths and saved_args ? 
        1. we need saved args cuz we want to automatically load the right database---it is recorded in log 
        2. we should use local paths cuz the log and model may be downloaded from a remote server 
        so the prefix of the path may not match 
        """
        # args.Database = os.path.join(args.PathDomain, os.path.basename(saved_args['Database']) )
        args.PathModel = os.path.join(path_logs, os.path.basename(saved_args['PathSave']) )
        args.PathResult = os.path.join(path_logs, f'results_{args.Split}')
        args.EvalLogPath = os.path.join(path_logs, "eval.out")

    ev = Eval(args)
    ev.run()

if __name__ == "__main__": main()
