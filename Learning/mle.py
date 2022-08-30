import numpy as np
import torch
import time
from data_io import shift_and_superpose
from Learning.utils import log_likelihood, type_loss, time_loss, evaluation
from TPPs.utils import PAD


def mle_epoch(model, dataloader, optimizer, pred_loss_func, opt):
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

        """ backward """
        # negative log-likelihood
        event_ll, non_event_ll = log_likelihood(all_lambda, event_time, event_type, non_pad_mask)
        event_loss = -torch.sum(event_ll - non_event_ll)

        # type prediction
        pred_loss, pred_num_event = type_loss(prediction[0], event_type, pred_loss_func)

        # time prediction
        se = time_loss(prediction[1], event_time)

        # SE is usually large, scale it to stabilize training
        scale_time_loss = 100
        loss = opt.w_mle * event_loss + opt.w_dis * (pred_loss + se / scale_time_loss)
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
    rmse = np.linalg.norm(np.array([total_time_se - total_num_pred]), ord=2) / np.linalg.norm(np.array([total_time_se]), ord=2)
    # rmse = np.sqrt(total_time_se / total_num_pred)
    return total_event_ll / total_num_event, total_event_rate / total_num_pred, rmse


def train_mle(model, dataloaders, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    valid_event_losses = []  # validation log-likelihood
    valid_pred_losses = []  # validation event type prediction accuracy
    valid_rmse = []  # validation event time prediction RMSE
    test_event_losses = []  # test log-likelihood
    test_pred_losses = []  # test event type prediction accuracy
    test_rmse = []  # test event time prediction RMSE
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_event, train_type, train_time = mle_epoch(model, dataloaders['train'], optimizer, pred_loss_func, opt)
        print('  - (Training)    loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, type=train_type, rmse=train_time, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_type, valid_time = evaluation(model, dataloaders['val'], pred_loss_func, opt)
        print('  - (Validating)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_event, type=valid_type, rmse=valid_time, elapse=(time.time() - start) / 60))

        start = time.time()
        test_event, test_type, test_time = evaluation(model, dataloaders['test'], pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'accuracy: {type: 8.5f}, RMSE: {rmse: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=test_event, type=test_type, rmse=test_time, elapse=(time.time() - start) / 60))

        valid_event_losses += [valid_event]
        valid_pred_losses += [valid_type]
        valid_rmse += [valid_time]
        test_event_losses += [test_event]
        test_pred_losses += [test_type]
        test_rmse += [test_time]
        max_idx = np.argmax(valid_pred_losses)
        print('  - [Info] Maximum ll: {event: 8.5f}, '
              'Maximum accuracy: {pred: 8.5f}, Minimum RMSE: {rmse: 8.5f}'
              .format(event=test_event_losses[max_idx], pred=test_pred_losses[max_idx], rmse=test_rmse[max_idx]))

        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_event, acc=valid_type, rmse=valid_time))

        if epoch_i == opt.epoch - 1:
            with open('result.log', 'a') as f:
                paras = 'w_mle' + '-' + str(opt.w_mle) + "; " + 'w_dis' + '-' + str(opt.w_dis) + "; " + 'w_cl1' + "-" + str(opt.w_cl1) + "; " + 'w_cl2' + "-" + str(opt.w_cl2) + "; " + 'superpose' + "-" + str(opt.superpose) + "; " + 'num_neg' + "-" + str(opt.num_neg)
                f.write('[Info] Model: {}\n'.format(opt.model + ' ' + opt.save_label))
                f.write('[Info] main parameters: {}\n'.format(paras))
                f.write('[Info] all parameters: {}\n'.format(opt))
                f.write('{epoch}, {ll: 8.5f}, {acc: 8.5f}, {rmse: 8.5f}\n'
                        .format(epoch=max_idx, ll=test_event_losses[max_idx], acc=test_pred_losses[max_idx],
                                rmse=test_rmse[max_idx]))
        scheduler.step()
