import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from TPPs.utils import softplus, PAD


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """
    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)
    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """
    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]
    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """ Log-likelihood of non-events, using Monte Carlo integration. """
    num_samples = 100
    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    temp_time = diff_time.unsqueeze(2) * torch.rand([*diff_time.size(), num_samples], device=data.device)
    temp_time /= (time[:, :-1] + 1).unsqueeze(2)

    temp_hid = model.linear(data)[:, 1:, :]
    temp_hid = torch.sum(temp_hid * type_mask[:, 1:, :], dim=2, keepdim=True)

    all_lambda = softplus(temp_hid + model.alpha * temp_time, model.beta)
    all_lambda = torch.sum(all_lambda, dim=2) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral


def log_likelihood(all_lambda, time, types, non_pad_mask):
    """ Log-likelihood of sequence. """
    non_pad_mask = non_pad_mask.squeeze(2)  # batch * seq_len
    num_types = all_lambda.shape[2]
    type_mask = torch.zeros([*types.size(), num_types], device=all_lambda.device)  # batch * seq_len * num_types
    for i in range(num_types):
        type_mask[:, :, i] = (types == i + 1).bool().to(all_lambda.device)

    type_lambda = torch.sum(all_lambda * type_mask, dim=2)

    # event log-likelihood
    event_ll = compute_event(type_lambda, non_pad_mask)
    event_ll = torch.sum(event_ll, dim=-1)

    # non-event log-likelihood, either numerical integration or MC integration
    non_event_ll = compute_integral_biased(type_lambda, time, non_pad_mask)
    non_event_ll = torch.sum(non_event_ll, dim=-1)
    return event_ll, non_event_ll


def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """
    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num


def time_loss(prediction, event_time):
    """ Time prediction loss. """
    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]

    # event time gap prediction
    diff = prediction - true
    se = torch.sum(diff * diff)
    return se


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """
        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = func.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = func.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss


def evaluation(model, dataloader, pred_loss_func, opt):
    """ Epoch operation in evaluation phase. """
    model.eval()
    total_event_ll = 0  # cumulative event log-likelihood
    total_time_se = 0  # cumulative time prediction squared-error
    total_event_rate = 0  # cumulative number of correct prediction
    total_num_event = 0  # number of total events
    total_num_pred = 0  # number of predictions
    with torch.no_grad():
        for (idx, batch) in enumerate(dataloader):
            """ prepare data """
            event_time, time_gap, event_type = batch[0].to(opt.device), batch[1].to(opt.device), batch[2].to(opt.device)

            """ forward """
            enc_out, enc_att, all_lambda, non_pad_mask, prediction = model(event_type, event_time)

            """ compute loss """
            event_ll, non_event_ll = log_likelihood(all_lambda, event_time, event_type, non_pad_mask)
            event_loss = -torch.sum(event_ll - non_event_ll)
            _, pred_num = type_loss(prediction[0], event_type, pred_loss_func)
            se = time_loss(prediction[1], event_time)

            """ note keeping """
            total_event_ll += -event_loss.item()
            total_time_se += se.item()
            total_event_rate += pred_num.item()
            total_num_event += event_type.ne(PAD).sum().item()
            total_num_pred += event_type.ne(PAD).sum().item() - event_time.shape[0]

    rmse = np.linalg.norm(np.array([total_time_se - total_num_pred]), ord=2) / np.linalg.norm(np.array([total_time_se]), ord=2)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse

