# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
test neural Datalog through time (NDTT)
@author: hongyuan
"""

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

# from xfmr_nhp_fast.esm.tester import Tester
from anhp.esm.generator import Generator
from anhp.utils.log import LogReader

# from shutil import copytree
from distutils.dir_util import copy_tree as copytree

import argparse
__author__ = 'Hongyuan Mei, Chenghao Yang'


def main():

    parser = argparse.ArgumentParser(description='testing neural Datalog through time (NDTT)')

    parser.add_argument(
        '-d', '--Domain', default="robocup", type=str, help='which domain to work on?'
    )
    parser.add_argument(
        '-fn', '--FolderName', default="test", type=str,
        help='base name of the folder to store the model (and log)?'
    )
    # parser.add_argument(
    #     '-s', '--Split', required=True, type=str, help='what split to test?',
    #     choices = ['train', 'dev', 'test']
    # )
    # parser.add_argument(
    #     '-r', '--Ratio', default=1.0, type=float, help='fraction of data to use'
    # )
    parser.add_argument(
        '-ps', '--PathStorage', type=str, default='../..',
        help='Path of storage which stores domains (with data), logs, results, etc. \
            Must be local (e.g. no HDFS allowed)'
    )
    parser.add_argument(
        "-sp", "--SavePath", type=str, default="../../domains/Gen_XFMR_NHP_Data",
        help="Path to store the output data"
    )
    # parser.add_argument(
    #     '-tp', '--TrackPeriod', default=1, type=int, help='# seqs each print while doing prediction'
    # )
    # parser.add_argument(
    #     '-m', '--Multiplier', default=1, type=int,
    #     help='constant of N=O(I), where N is # of sampled time points for integral'
    # )
    # parser.add_argument(
    #     '-dm', '--DevMultiplier', default=1, type=int,
    #     help='constant of N=O(I), where N is # of sampled time points for integral'
    # )
    parser.add_argument(
        '-pretrain', '--LoadFromPretrain', action='store_true', help='load pretrain model?'
    )
    parser.add_argument(
        "-min_len", "--MinLength", default=30, type=int,
        help="the minimum length of the generated sequences"
    )
    parser.add_argument(
        "-max_len", "--MaxLength", default=100, type=int,
        help="the maximum length of the generated sequences"
    )
    parser.add_argument(
        "-look_ahead", "--LookAheadTime", default=10, type=float,
        help="the maximum look-ahead time used in thinning"
    )
    parser.add_argument(
        "-num_seqs", "--NumSeqs", default=1000, type=int,
        help="the overall number (train) (+2000 = all seqs) of the generated sequences"
    )
    parser.add_argument(
        "-patience", "--Patience", default=5, type=int,
        help="the maximum iteration used in adaptive thinning"
    )
    # thinning related
    parser.add_argument(
        '-nobj', '--NumObject', default=1, type=int, 
        help="default==1 : number of objects to predict, from last to first (first obj==subj)"
    )
    parser.add_argument(
        '-ns', '--NumSample', default=1, type=int,
        help='default==100 : number of sampled next event times via thinning algorithm, used to compute predictions'
    )
    parser.add_argument(
        '-nexp', '--NumExp', default=10, type=int,
        help='default==500 : number of i.i.d. Exp(intensity_bound) draws at one time in thinning algorithm'
    )
    parser.add_argument(
        "-load_data", "--LoadStandardData", action="store_true", help="load standard dataset "
                                                                      "so you do not need to decide event_num?"
    )
    parser.add_argument(
        "-en", "--EventNum", default=10, type=int,
        help="if you decide to determine event_num by your own (you do not use -load_data), "
             "please input corresponding number"
    )
    #parser.add_argument(
    #    '-a', '--Adaptive', action='store_true', help='use adaptive thinning algorithm?'
    #)
    #parser.add_argument(
    #    '-f', '--Fractional', action='store_true', help='use fractional thinning algorithm?'
    #)
    parser.add_argument(
        '-np', '--NumProcess', default=1, type=int, help='# of processes used, default is 1'
    )
    parser.add_argument(
        '-nt', '--NumThread', default=1, type=int, help='OMP NUM THREADS'
    )
    parser.add_argument(
        '-gpu', '--UseGPU', action='store_true', help='use GPU?'
    )
    # parser.add_argument(
    #     '-ignfir', '--IgnoreFirst', action='store_true', help='whether ignore the first interval in log-like computation?'
    # )
    parser.add_argument(
        '-sd', '--Seed', default=12345, type=int, help='random seed'
    )
    parser.add_argument(
        '-teDim', '--TimeEmbeddingDim', default=10, type=int,
        help='the dimensionality of time embedding'
    )
    parser.add_argument('-d_model', "--ModelDim", type=int, default=32)
    parser.add_argument(
        '-layer', '--Layer', default=1, type=int,
        help='the number of layers of Transformer'
    )
    parser.add_argument('-n_head', "--NumHead", type=int, default=1)

    parser.add_argument('-dp', "--Dropout", type=float, default=0.1)
    # parser.add_argument(
    #     '-v', '--Verbose', action='store_true', help='show a lot of messages while testing?'
    # )

    args = parser.parse_args()
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()
    random.seed(args.Seed)
    torch.cuda.manual_seed(args.Seed)
    torch.manual_seed(args.Seed)
    numpy.random.seed(args.Seed)

    args.Version = torch.__version__
    args.ID = id_process
    args.TIME = time_current

    path_storage = os.path.abspath(args.PathStorage)
    args.PathDomain = os.path.join(path_storage, 'domains', args.Domain)
    args.SavePath = os.path.join(args.SavePath, f"layer_{args.Layer}_K_{args.EventNum}_dim_{args.ModelDim}_tedim_{args.TimeEmbeddingDim}"
                                                f"_minlen_{args.MinLength}_maxlen_{args.MaxLength}_numseqs_{args.NumSeqs}")

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
        # if args.Predict:
        args.BatchSize = 1
        if args.LoadFromPretrain:
            path_logs = os.path.join( args.PathDomain, 'ContKVLogs', args.FolderName )
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
            # args.Database = saved_args['Database']
            args.BatchSize = saved_args["BatchSize"]
            args.TimeEmbeddingDim = saved_args["TimeEmbeddingDim"]
            args.ModelDim = saved_args["ModelDim"]
            args.Layer = saved_args["Layer"]
            args.NumHead = saved_args["NumHead"]
            args.Dropout = saved_args["Dropout"]
            args.PathModel = os.path.join(path_logs, os.path.basename(saved_args['PathSave']) )
        else:
            args.PathModelDir = os.path.join(args.SavePath, "ContKVLogs", "ground_truth")
            args.PathSave = os.path.join(args.PathModelDir, "saved_model")
            os.makedirs(args.PathModelDir, exist_ok=True)

    copytree(os.path.join(path_storage, "xfmr_nhp_fast_contKV", "model"), os.path.join(args.SavePath, "model"))
    copytree(os.path.join(path_storage, "xfmr_nhp_fast_contKV", "esm"), os.path.join(args.SavePath, "esm"))
    args.NumProcess = 1
    #if args.MultiProcessing: 
    #    if args.NumProcess < 1:
    #        args.NumProcess = os.cpu_count()
    
    if args.NumThread < 1: 
        args.NumThread = 1

    print(f"mp num threads in torch : {torch.get_num_threads()}")
    if torch.get_num_threads() != args.NumThread: 
        print(f"not equal to NumThread arg ({args.NumThread})")
        torch.set_num_threads(args.NumThread)
        print(f"set to {args.NumThread}")
        assert torch.get_num_threads() == args.NumThread, "not set yet?!"

    generator = Generator(args)
    generator.run()

if __name__ == "__main__": main()
