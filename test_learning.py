import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import TPPs.thp as thp
from Learning.hcl import train_hcl
from Learning.mle import train_mle
from Learning.utils import LabelSmoothingLoss
from data_io import prepare_dataloader
import os


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data-folder', required=True, type=str, default=os.path.join('tpp-data', 'data_retweet'))

    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-batch-size', type=int, default=16)
    parser.add_argument('-d-model', type=int, default=64)
    parser.add_argument('-d-rnn', type=int, default=256)
    parser.add_argument('-d-inner-hid', type=int, default=128)
    parser.add_argument('-d-k', type=int, default=16)
    parser.add_argument('-d-v', type=int, default=16)

    parser.add_argument('-n-head', type=int, default=4)
    parser.add_argument('-n-layers', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)
    parser.add_argument('-log', type=str, default='log.txt')

    # key parameters we need to try
    parser.add_argument('-w-mle', type=float, default=1)
    parser.add_argument('-w-dis', type=float, default=1)
    parser.add_argument('-w-cl1', type=float, default=1)
    parser.add_argument('-w-cl2', type=float, default=1)
    parser.add_argument('-num-neg', type=int, default=5)
    parser.add_argument('-ratio-remove', type=float, default=0.1)
    parser.add_argument('-superpose', default=False, action='store_true')
    opt = parser.parse_args()
    # TODO: The models we can try:
    #   MLE + Reg: w-mle = 1, w-dis = 1, w-cl1 = w-cl2 = 0, superpose=False + call "train_mle"
    #   MLE + DA: w-mle = 1, w-dis = 1, w-cl1 = w-cl2 = 0, superpose=True + call "train_mle"
    #   Dis: w-mle = 0, w-dis = 1, w-cl1 = w-cl2 = 0, superpose=False + call "train_mle"
    #   HCL (event only): w-mle = 0, w-dis = 1, w-cl1 > 0, w-cl2 = 0, superpose=False + call "train_hcl"
    #   HCL (seq only): w-mle = 0, w-dis = 1, w-cl1 = 0, w-cl2 > 0, superpose=False + call "train_hcl"
    #   HCL (both): w-mle = 0, w-dis = 1, w-cl1 > 0, w-cl2 > 0, superpose=False + call "train_hcl"
    #   MLE + HCL : w-mle = 1, w-dis = 1, w-cl1 > 0, w-cl2 > 0, superpose=False + call "train_hcl"

    # TODO: Key parameters we need to try (just on one dataset and Transformer Hawkes)
    #   The number of negative sequence: num-neg in {1, 5, 10}
    #   w-cl1 in {0, 1e-1, 1, 10}
    #   w-cl2 in {0, 1e-1, 1, 10}

    # default device is CUDA
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup the log file

    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    dataloaders, num_types = prepare_dataloader(opt.data_folder, opt.batch_size)

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
    train_mle(model, dataloaders, optimizer, scheduler, pred_loss_func, opt)
    train_hcl(model, dataloaders, optimizer, scheduler, pred_loss_func, opt)


if __name__ == '__main__':
    # import ipdb; ipdb.set_trace()
    main()
