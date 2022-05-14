import numpy as np
import torch
import time
from data_io import sampling_positive_seqs, sampling_negative_seqs, shift_and_superpose
from Learning.utils import log_likelihood, type_loss, time_loss, evaluation
from TPPs.utils import PAD


def event2seq_embedding(enc_out, non_pad_mask):
    """
    enc_out: event embedding with size batch x seq_len x d_model
    non_pad_mask: with size batch x seq_len x 1
    """
    seq_emb = torch.sum(enc_out * non_pad_mask, dim=1) / (torch.sum(non_pad_mask, dim=1) + 1e-6)  # batch * d_model
    return seq_emb


def event_contrastive_loss(all_lambda, types, non_pad_mask):
    num_types = all_lambda.shape[2]
    type_mask = torch.zeros([*types.size(), num_types], device=all_lambda.device)  # batch * seq_len * num_types
    for i in range(num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(all_lambda.device)

    pos_event_lambda = torch.sum(all_lambda * type_mask, dim=2, keepdim=True)  # batch * seq_len * 1
    neg_event_lambda = all_lambda * (1 - type_mask)  # batch * seq_len * num_types
    all_event_lambda = torch.sum(all_lambda + 1e-8, dim=2, keepdim=True)  # batch * seq_len * 1

    pos_p = (pos_event_lambda + 1e-8) / all_event_lambda  # batch * seq_len * 1
    neg_p = 1 - neg_event_lambda / all_event_lambda  # batch * seq_len * num_types
    cl1 = -torch.mean(torch.log(pos_p) * non_pad_mask)
    cl2 = -torch.mean(torch.sum(torch.log(neg_p) * non_pad_mask, dim=2))
    return cl1 + cl2


def seq_contrastive_loss(seq_emb, pos_seq_emb, neg_seq_emb, scalar: float = 0.1):
    """
    Sequence level contrastive loss
    seq_emb: batch * d_model
    pos_seq_emb: batch * d_model
    neg_seq_emb: (batch * K) * d_model
    """
    batch = seq_emb.shape[0]
    num_neg = int(neg_seq_emb.shape[0] / seq_emb.shape[0])
    pos_v = torch.exp(scalar * torch.sum(seq_emb * pos_seq_emb, dim=1, keepdim=True))  # batch x 1
    for k in range(num_neg):
        neg_seq_emb[k*batch:(k+1)*batch, :] = seq_emb * neg_seq_emb[k*batch:(k+1)*batch, :]
    neg_v = torch.exp(scalar * torch.sum(neg_seq_emb, dim=1))  # (batch * num_neg)
    neg_v = torch.reshape(neg_v, (batch, num_neg))  # batch x num_neg

    all_v = torch.sum(neg_v, dim=1, keepdim=True) + pos_v  # batch x 1
    cl1 = -torch.mean(torch.log(pos_v / all_v))
    cl2 = -torch.mean(torch.sum(torch.log(1 - neg_v / all_v), dim=1))
    return cl1 + cl2


def hcl_epoch(model, dataloader, optimizer, pred_loss_func, opt):
    """ Maximum likelihood estimation per epoch """
    model.train()
    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    for (idx, batch) in enumerate(dataloader):
        """ prepare data """
        # event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        event_time, time_gap, event_type = batch[0].to(opt.device), batch[1].to(opt.device), batch[2].to(opt.device)
        if opt.superpose:
            event_type, event_time = shift_and_superpose(event_type, event_time)

        """ forward """
        optimizer.zero_grad()
        enc_out, enc_att, all_lambda, non_pad_mask, prediction = model(event_type, event_time)
        event_significance = torch.sum(enc_att, dim=2)  # batch * seq_len
        seq_emb = event2seq_embedding(enc_out, non_pad_mask)

        pos_event_type, pos_event_time = sampling_positive_seqs(event_type,
                                                                event_time,
                                                                significance=event_significance,
                                                                ratio_remove=opt.ratio_remove)
        neg_event_type, neg_event_time = sampling_negative_seqs(event_type,
                                                                event_time,
                                                                num_neg=opt.num_neg,
                                                                ratio_remove=opt.ratio_remove)
        pos_enc_out, _, _, pos_non_pad_mask, _ = model(pos_event_type, pos_event_time)
        neg_enc_out, _, _, neg_non_pad_mask, _ = model(neg_event_type, neg_event_time)
        pos_seq_emb = event2seq_embedding(pos_enc_out, pos_non_pad_mask)
        neg_seq_emb = event2seq_embedding(neg_enc_out, neg_non_pad_mask)

        """ backward """
        # event-level contrastive loss
        nce1 = event_contrastive_loss(all_lambda, event_type, non_pad_mask)

        # sequence-level contrastive loss
        nce2 = seq_contrastive_loss(seq_emb, pos_seq_emb, neg_seq_emb)

        # negative log-likelihood
        event_ll, non_event_ll = log_likelihood(all_lambda, event_time, event_type, non_pad_mask)
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = time_loss(prediction[1], event_time)

        # SE is usually large, scale it to stabilize training
        scale_time = 100.0
        loss = opt.w_mle * event_loss + opt.w_dis * (pred_loss + se / scale_time) + opt.w_cl1 * nce1 + opt.w_cl2 * nce2
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_event_ll += -event_loss.item()
        total_time_se += se.item()
        total_event_rate += pred_num_event.item()
        total_num_event += event_type.ne(PAD).sum().item()
        # we do not predict the first event
        total_num_pred += event_type.ne(PAD).sum().item() - event_time.shape[0]

    rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


def train_hcl(model, dataloaders, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time = hcl_epoch(model, dataloaders['train'], optimizer, pred_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_type, valid_time = evaluation(model, dataloaders['val'], pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        print('  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
              .format(event=max(valid_event_losses), pred=max(valid_pred_losses), rmse=min(valid_rmse)))

        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time))

        scheduler.step()
