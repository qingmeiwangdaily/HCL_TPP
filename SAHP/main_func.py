import argparse
import datetime
import glob
import os
import pickle
import numpy as np
import time

import torch
from torch import autograd

from utils.load_synth_data import process_loaded_sequences
from train_functions.train_sahp import make_model, train_eval_sahp

DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_SIZE = 16
DEFAULT_LEARN_RATE = 5e-5

parser = argparse.ArgumentParser(description="Train the models.")
parser.add_argument('-e', '--epochs', type=int, default = 1000,
                    help='number of epochs.')
parser.add_argument('-b', '--batch', type=int,
                    dest='batch_size', default=DEFAULT_BATCH_SIZE,
                    help='batch size. (default: {})'.format(DEFAULT_BATCH_SIZE))
parser.add_argument('--lr', default=DEFAULT_LEARN_RATE, type=float,
                    help="set the optimizer learning rate. (default {})".format(DEFAULT_LEARN_RATE))
parser.add_argument('--hidden', type=int,
                    dest='hidden_size', default=DEFAULT_HIDDEN_SIZE,
                    help='number of hidden units. (default: {})'.format(DEFAULT_HIDDEN_SIZE))
parser.add_argument('--d-model', type=int, default=DEFAULT_HIDDEN_SIZE)
parser.add_argument('--atten-heads', type=int, default=8)
parser.add_argument('--pe', type=str,default='add',help='concat, add')
parser.add_argument('--nLayers', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--train-ratio', type=float, default=0.8,
                    help='override the size of the training dataset.')
parser.add_argument('--lambda-l2', type=float, default=3e-4,
                    help='regularization loss.')
parser.add_argument('--dev-ratio', type=float, default=0.1,
                    help='override the size of the dev dataset.')
parser.add_argument('--early-stop-threshold', type=float, default=1e-2,
                    help='early_stop_threshold')
parser.add_argument('--log-dir', type=str,
                    dest='log_dir', default='logs',
                    help="training logs target directory.")
parser.add_argument('--save_model', default=False,
                    help="do not save the models state dict and loss history.")
parser.add_argument('--bias', default=False,
                    help="use bias on the activation (intensity) layer.")
parser.add_argument('--samples', default=10,
                    help="number of samples in the integral.")
parser.add_argument('-m', '--model', default='sahp',
                    type=str, choices=['sahp'],
                    help='choose which models to train.')
parser.add_argument('-t', '--task', type=str, default='data_bookorder/fold1',
                    help = 'task type')
args = parser.parse_args()
print(args)

if torch.cuda.is_available():
    USE_CUDA = True
else:
    USE_CUDA = False

SYNTH_DATA_FILES = glob.glob("../data/simulated/*.pkl")
# TYPE_SIZE_DICT = {'data_retweet': 3, 'bookorder':8, 'meme':5000, 'mimic':75, 'stackOverflow':22,
#                   'synthetic':2}
TYPE_SIZE_DICT = {'data_bookorder/fold1':8,'data_retweet': 3,  'meme':5000, 'mimic':75, 'stackOverflow':22,
                  'synthetic':2}
REAL_WORLD_TASKS = list(TYPE_SIZE_DICT.keys())[:5]
SYNTHETIC_TASKS = list(TYPE_SIZE_DICT.keys())[5:]

start_time = time.time()

if __name__ == '__mai' \
               'n__':
    cuda_num = 'cuda:{}'.format(args.cuda)
    device = torch.device(cuda_num if USE_CUDA else 'cpu')
    print("Training on device {}".format(device))

    process_dim = TYPE_SIZE_DICT[args.task]
    print("Loading {}-dimensional process.".format(process_dim), end=' \n')

    if args.task in SYNTHETIC_TASKS:
        print("Available files:")
        for i, s in enumerate(SYNTH_DATA_FILES):
            print("{:<8}{:<8}".format(i, s))

        chosen_file_index = -1
        chosen_file = SYNTH_DATA_FILES[chosen_file_index]
        print('chosen file:%s'+str(chosen_file))

        with open(chosen_file, 'rb') as f:
            loaded_hawkes_data = pickle.load(f)

        mu = loaded_hawkes_data['mu']
        alpha = loaded_hawkes_data['alpha']
        decay = loaded_hawkes_data['decay']
        tmax = loaded_hawkes_data['tmax']
        print("Simulated Hawkes process parameters:")
        for label, val in [("mu", mu), ("alpha", alpha), ("decay", decay), ("tmax", tmax)]:
            print("{:<20}{}".format(label, val))

        seq_times, seq_types, seq_lengths, _ = process_loaded_sequences(loaded_hawkes_data, process_dim)

        seq_times = seq_times.to(device)
        seq_types = seq_types.to(device)
        seq_lengths = seq_lengths.to(device)

        total_sample_size = seq_times.size(0)
        print("Total sample size: {}".format(total_sample_size))

        train_ratio = args.train_ratio
        train_size = int(train_ratio * total_sample_size)
        dev_ratio = args.dev_ratio
        dev_size = int(dev_ratio * total_sample_size)
        print("Train sample size: {:}/{:}".format(train_size, total_sample_size))
        print("Dev sample size: {:}/{:}".format(dev_size, total_sample_size))

        # Define training data
        train_times_tensor = seq_times[:train_size]
        train_seq_types = seq_types[:train_size]
        train_seq_lengths = seq_lengths[:train_size]
        print("No. of event tokens in training subset:", train_seq_lengths.sum())

        # Define development data
        dev_times_tensor = seq_times[train_size:]#train_size+dev_size
        dev_seq_types = seq_types[train_size:]
        dev_seq_lengths = seq_lengths[train_size:]
        print("No. of event tokens in development subset:", dev_seq_lengths.sum())

        test_times_tensor = dev_times_tensor
        test_seq_types = dev_seq_types
        test_seq_lengths = dev_seq_lengths
        print("No. of event tokens in test subset:", test_seq_lengths.sum())

    elif args.task in REAL_WORLD_TASKS:
        train_path = 'data/' + args.task + '/train_manifold_format.pkl'
        dev_path = 'data/' + args.task + '/dev_manifold_format.pkl'
        test_path = 'data/' + args.task + '/test_manifold_format.pkl'
        chosen_file = args.task

        with open(train_path, 'rb') as f:
            train_hawkes_data = pickle.load(f,encoding='bytes')
        with open(dev_path, 'rb') as f:
            dev_hawkes_data = pickle.load(f,encoding='bytes')
        with open(test_path, 'rb') as f:
            test_hawkes_data = pickle.load(f,encoding='bytes')

        train_seq_times, train_seq_types, train_seq_lengths, train_tmax = \
            process_loaded_sequences(train_hawkes_data, process_dim)
        dev_seq_times, dev_seq_types, dev_seq_lengths, dev_tmax = \
            process_loaded_sequences(dev_hawkes_data, process_dim)
        test_seq_times, test_seq_types, test_seq_lengths, test_tmax = \
            process_loaded_sequences(test_hawkes_data, process_dim)

        tmax = max([train_tmax,dev_tmax,test_tmax])

        train_sample_size = train_seq_times.size(0)
        print("Train sample size: {}".format(train_sample_size))

        dev_sample_size = dev_seq_times.size(0)
        print("Dev sample size: {}".format(dev_sample_size))

        test_sample_size = test_seq_times.size(0)
        print("Test sample size: {}".format(test_sample_size))

        # Define training data
        train_times_tensor = train_seq_times.to(device)
        train_seq_types = train_seq_types.to(device)
        train_seq_lengths = train_seq_lengths.to(device)
        print("No. of event tokens in training subset:", train_seq_lengths.sum())

        # Define development data
        dev_times_tensor = dev_seq_times.to(device)
        dev_seq_types = dev_seq_types.to(device)
        dev_seq_lengths = dev_seq_lengths.to(device)
        print("No. of event tokens in development subset:", dev_seq_lengths.sum())

        # Define test data
        test_times_tensor = test_seq_times.to(device)
        test_seq_types = test_seq_types.to(device)
        test_seq_lengths = test_seq_lengths.to(device)
        print("No. of event tokens in test subset:", test_seq_lengths.sum())

    else:
        exit()


    MODEL_TOKEN = args.model
    print("Chose models {}".format(MODEL_TOKEN))
    hidden_size = args.hidden_size
    print("Hidden size: {}".format(hidden_size))
    learning_rate = args.lr
    # Training parameters
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    model = None
    if MODEL_TOKEN == 'sahp':
        with autograd.detect_anomaly():
            params = args, process_dim, device, tmax, \
                     train_times_tensor, train_seq_types, train_seq_lengths, \
                     dev_times_tensor, dev_seq_types, dev_seq_lengths, \
                     test_times_tensor, test_seq_types, test_seq_lengths, \
                     BATCH_SIZE, EPOCHS, USE_CUDA
            model = train_eval_sahp(params)

    else:
        exit()


    if args.save_model:
        # Model file dump
        SAVED_MODELS_PATH = os.path.abspath('saved_models')
        os.makedirs(SAVED_MODELS_PATH, exist_ok=True)
        # print("Saved models directory: {}".format(SAVED_MODELS_PATH))

        date_format = "%Y%m%d-%H%M%S"
        now_timestamp = datetime.datetime.now().strftime(date_format)
        extra_tag = "{}".format(args.task)
        filename_base = "{}-{}_hidden{}-{}".format(
            MODEL_TOKEN, extra_tag,
            hidden_size, now_timestamp)
        from utils.save_model import save_model
        save_model(model, chosen_file, extra_tag,
                   hidden_size, now_timestamp, MODEL_TOKEN)

    print('Done! time elapsed %.2f sec for %d epoches' % (time.time() - start_time, EPOCHS))

