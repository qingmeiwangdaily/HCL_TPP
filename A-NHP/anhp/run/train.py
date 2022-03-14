import argparse
import datetime
import os
import torch
import shutil
from anhp.esm.trainer import Trainer


def get_args():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-d', '--Domain', required=True, type=str, help='which domain to work on?'
    )
    parser.add_argument(
        '-ps', '--PathStorage', type=str, default='../..',
        help='Path of storage which stores domains (with data), logs, results, etc. \
            Must be local (e.g. no HDFS allowed)'
    )

    parser.add_argument(
        '-me', '--MaxEpoch', default=100, type=int, help='max # training epochs'
    )
    parser.add_argument(
        '-bs', '--BatchSize', default=1, type=int,
        help='# checkpoints / seqs to update parameters'
    )
    parser.add_argument(
        '-teDim', '--TimeEmbeddingDim', default=100, type=int,
        help='the dimensionality of time embedding'
    )
    parser.add_argument('-d_model', "--ModelDim", type=int, default=64)
    parser.add_argument(
        '-layer', '--Layer', default=3, type=int,
        help='the number of layers of Transformer'
    )
    parser.add_argument(
        '-sd', '--Seed', default=12345, type=int, help='random seed'
    )

    parser.add_argument('-n_head', "--NumHead", type=int, default=1)

    parser.add_argument('-dp', "--Dropout", type=float, default=0.1)
    parser.add_argument(
        '-lr', '--LearnRate', default=1e-4, type=float, help='learning rate'
    )
    parser.add_argument(
        '-np', '--NumProcess', default=1, type=int, help='# of processes used, default is 1'
    )
    parser.add_argument(
        '-nt', '--NumThread', default=1, type=int, help='OMP NUM THREADS'
    )
    parser.add_argument(
        '-ignfir', '--IgnoreFirst', action='store_true', help='whether ignore the first interval in log-like computation?'
    )
    args = parser.parse_args()
    return args


def aug_args_with_log(args):
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()

    args.Version = torch.__version__
    args.ID = id_process
    args.TIME = time_current

    path_storage = os.path.abspath(args.PathStorage)
    args.PathDomain = os.path.join(path_storage, 'domains', args.Domain)

    dict_args = vars(args)
    folder_name = get_foldername(dict_args)

    path_logs = os.path.join(args.PathDomain, 'ContKVLogs', folder_name)
    os.makedirs(path_logs)
    args.PathLog = os.path.join(path_logs, 'log.txt')
    args.PathSave = os.path.join(path_logs, 'saved_model')
    shutil.copytree(os.path.join(path_storage, "anhp", "model"), os.path.join(path_logs, "model"))
    shutil.copytree(os.path.join(path_storage, "anhp", "esm"), os.path.join(path_logs, "esm"))

    args.NumProcess = 1
    # if args.MultiProcessing:
    #    if args.NumProcess < 1:
    #        args.NumProcess = os.cpu_count()

    #if args.NumThread < 1:
        #args.NumThread = 1
#
    #print(f"mp num threads in torch : {torch.get_num_threads()}")
    #if torch.get_num_threads() != args.NumThread:
        #print(f"not equal to NumThread arg ({args.NumThread})")
        #torch.set_num_threads(args.NumThread)
        #print(f"set to {args.NumThread}")
        #assert torch.get_num_threads() == args.NumThread, "not set yet?!"


def get_foldername(dict_args):
    """
    NOTE
    adjust foler name attributes
    """
    args_used_in_name = [
        ["NumHead", "h"],
        ["MaxEpoch", "me"],
        ["ModelDim", "d_model"],
        ["Dropout", "dp"],
        ["TimeEmbeddingDim", "teDim"],
        ["Layer", "layer"],
        ['LearnRate', 'lr'],
        ['Seed', 'seed'],
        ['IgnoreFirst', "ignoreFirst"],
        ['ID', 'id'],
    ]
    folder_name = list()
    for arg_name, rename in args_used_in_name:
        folder_name.append('{}-{}'.format(rename, dict_args[arg_name]))
    folder_name = '_'.join(folder_name)
    return folder_name


def main():
    args = get_args()
    aug_args_with_log(args)
    trainer = Trainer(args)
    trainer.run()


if __name__ == "__main__": main()
