import argparse
import torch
import torch.nn as nn
import torch.optim as optim
# import TPPs.lrhp as lrhp
import TPPs.thp as thp
from Learning.hcl import train_hcl
from Learning.mle import train_mle
from Learning.utils import LabelSmoothingLoss
from data_io import prepare_dataloader
import os
import random
import numpy as np


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data_folder', type=str, default='tpp-data/data_missing')

    parser.add_argument('-epoch', type=int, default=20)
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-d_model', type=int, default=64)
    parser.add_argument('-d_rnn', type=int, default=256)
    parser.add_argument('-d_inner_hid', type=int, default=128)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-smooth', type=float, default=0.1)
    parser.add_argument('-log', type=str, default='log.txt')
    parser.add_argument('-seed', type=int, default=123456)

    # key parameters we need to try
    parser.add_argument('-w_mle', type=float, default=1.0)
    parser.add_argument('-w_dis', type=float, default=1.0)
    parser.add_argument('-w_cl1', type=float, default=0.0)
    parser.add_argument('-w_cl2', type=float, default=0.0)
    parser.add_argument('-num_neg', type=int, default=5)
    parser.add_argument('-ratio_remove', type=float, default=0.2)
    parser.add_argument('-superpose', default=False, action='store_true')

    # for result_log
    parser.add_argument('-model', type=str, default='MLE')
    parser.add_argument('-save_label', type=str, default='MLE_Reg')
    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup the log file

    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    seed_everything(opt.seed)

    """ prepare dataloader """
    dataloaders, num_types = prepare_dataloader(opt.data_folder, opt.batch_size)
    print(num_types)

    """ prepare model """
    model = thp.TransformerHawkes(
        num_types=num_types,
        d_model=opt.d_model,
        d_rnn=opt.d_rnn,
        d_inner=opt.d_inner_hid,
        n_layers=opt.n_layers,
        n_head=opt.n_head,
        d_k=opt.d_k,
        d_v=opt.d_v,
        dropout=opt.dropout,
    )
    # model = lrhp.LowRankHawkes(
    #     num_types=num_types,
    #     d_model=opt.d_model,
    #  )
    model.to(opt.device)
    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))
    """ train the model """  # TODO: pls check whether these two functions work or not on GPUs
    if opt.model == 'MLE':
        train_mle(model, dataloaders, optimizer, scheduler, pred_loss_func, opt)
    else:
        train_hcl(model, dataloaders, optimizer, scheduler, pred_loss_func, opt)

def seed_everything(seed=666):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    # import ipdb; ipdb.set_trace()
    main()

