import time
import torch
import pickle
import os
from torch import optim
from anhp.utils.log import LogWriter
from anhp.esm.manager import Manager
class Trainer(Manager):

    def __init__(self, args):
        tic = time.time()
        super(Trainer, self).__init__(args)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=args.LearnRate
        )
        self.optimizer.zero_grad() # init clear
        self.log = LogWriter(args.PathLog, vars(self.args) )
        # use self.args not args, cuz init function may add things to args
        self.log.initBest()
        # self.max_episode = args.MaxEpoch * self.data.sizes['train']
        print(f"time spent on initialization : {time.time()-tic:.2f}")


    def run(self):
        print("start training...")
        for _epoch in range(self.args.MaxEpoch):
            tic = time.time()
            log_lik, acc, (event_ll, non_event_ll), num_tokens, num_events, _, _, _, _ \
                = self.run_one_iteration(self.model, self.train_loader, "train", self.optimizer)
            time_train = (time.time() - tic)
            message = f"[ Epoch {_epoch} (train) ]: time to train one epoch is {time_train}, train log-like is {log_lik / num_tokens}, num_tokens: {num_tokens}, num_events: {num_events}\n" \
                      f", event_ll is {event_ll / num_tokens: .4f}, non_event_ll is {non_event_ll / num_tokens: .4f}, acc is {acc : .4f} "
            self.log.checkpoint(message)
            print(message)
            with torch.no_grad():
                tic = time.time()
                log_lik, acc, (event_ll, non_event_ll), num_tokens, num_events, _, _, _, _ \
                    = self.run_one_iteration(self.model, self.dev_loader, "eval")
                time_valid = (time.time() - tic)
                message = f"[ Epoch {_epoch} (valid) ]: time to validate is {time_valid}, valid log-like is {log_lik / num_tokens}, valid acc is {acc : .4f}, " \
                          f"valid_event_ll is {event_ll / num_tokens: .4f}, valid_non_event_ll is {non_event_ll / num_tokens: .4f}"
                self.log.checkpoint(message)
                print(message)
                updated = self.log.updateBest("loglik", log_lik / num_tokens, _epoch)
                message = "current best loglik is {:.4f} (updated at epoch-{})".format(
                    self.log.current_best['loglik'], self.log.episode_best)
                if updated:
                    message += f", best updated at this epoch"
                    torch.save(self.model.state_dict(), self.args.PathSave)
                self.log.checkpoint(message)
                print(message)
                tic = time.time()
                log_lik, acc, (event_ll, non_event_ll), num_tokens, num_events, _, _, _, _ \
                    = self.run_one_iteration(self.model, self.test_loader, "eval")
                time_valid = (time.time() - tic)
                message = f"[ Epoch {_epoch} (test) ]: time to validate is {time_valid}, valid log-like is {log_lik / num_tokens}, valid acc is {acc : .4f}, " \
                          f"valid_event_ll is {event_ll / num_tokens: .4f}, valid_non_event_ll is {non_event_ll / num_tokens: .4f}"
                self.log.checkpoint(message)
                print(message)



