from util import *
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from model import Net
from tqdm import tqdm
import numpy as np


def evaluate():
    model.eval()
    pred_times, pred_events = [], []
    gold_times, gold_events = [], []
    for i, batch in enumerate(tqdm(test_loader)):
        gold_times.append(batch[0][:, -1].numpy())
        gold_events.append(batch[1][:, -1].numpy())
        pred_time, pred_event = model.predict(batch)
        pred_times.append(pred_time)
        pred_events.append(pred_event)
    pred_times = np.concatenate(pred_times).reshape(-1)
    gold_times = np.concatenate(gold_times).reshape(-1)
    pred_events = np.concatenate(pred_events).reshape(-1)
    gold_events = np.concatenate(gold_events).reshape(-1)
    time_error = abs_error(pred_times, gold_times)
    acc, recall, f1 = clf_metric(pred_events, gold_events, n_class=config.event_class)
    print(f"epoch {epc}")
    print(f"time_error: {time_error}, PRECISION: {acc}, RECALL: {recall}, F1: {f1}")


if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--model", type=str, default="rmtpp", help="erpp, rmtpp")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--emb_dim", type=int, default=10)
    parser.add_argument("--hid_dim", type=int, default=32)
    parser.add_argument("--mlp_dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--event_class", type=int, default=7)
    parser.add_argument("--verbose_step", type=int, default=350)
    parser.add_argument("--importance_weight", action="store_true")
    parser.add_argument("--lr", type=int, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    config = parser.parse_args()

    train_set = ATMDataset(config, subset='train')
    test_set = ATMDataset(config, subset='test')
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=ATMDataset.to_features)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=ATMDataset.to_features)

    weight = np.ones(config.event_class)
    if config.importance_weight:
        weight = train_set.importance_weight()
        print("importance weight: ", weight)
    model = Net(config, lossweight=weight)
    model.set_optimizer(total_step=len(train_loader) * config.epochs, use_bert=True)
    model.cuda()

    for epc in range(config.epochs):
        model.train()
        range_loss1 = range_loss2 = range_loss = 0
        for i, batch in enumerate(tqdm(train_loader)):
            l1, l2, l = model.train_batch(batch)
            range_loss1 += l1
            range_loss2 += l2
            range_loss += l

            if (i + 1) % config.verbose_step == 0:
                print("time loss: ", range_loss1 / config.verbose_step)
                print("event loss:", range_loss2 / config.verbose_step)
                print("total loss:", range_loss / config.verbose_step)
                range_loss1 = range_loss2 = range_loss = 0

        evaluate()
