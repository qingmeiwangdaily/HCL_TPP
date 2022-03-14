from anhp.utils.log import LogWriter
import pickle
import os
import torch
from anhp.esm.manager import Manager
from anhp.model.xfmr_nhp_fast import XFMRNHPFast
from tqdm import tqdm
import random
import math

class DummyDataset:
    def __init__(self, event_num):
        self.event_num = event_num
        self.pad_index = self.event_num
        self.num_types = event_num + 1

class Generator(Manager):

    def __init__(self, args):
        # do not call super -- we need more custom initialization here
        # we just need to inherit the method, not the init
        self.args = args
        if args.LoadStandardData:
            [self.train_loader, self.dev_loader, self.test_loader] \
                = self.get_dataloader(args)
            self.model = XFMRNHPFast(self.train_loader.dataset, args.ModelDim, args.Layer, args.NumHead, args.Dropout,
                                     args.TimeEmbeddingDim).cuda()
            self.EventNum = self.train_loader.dataset.event_num
        else:
            self.EventNum = args.EventNum
            self.model = XFMRNHPFast(DummyDataset(args.EventNum), args.ModelDim, args.Layer, args.NumHead, args.Dropout,
                                     args.TimeEmbeddingDim).cuda()


        if args.LoadFromPretrain:
            self.model.load_state_dict(torch.load(args.PathModel))

        self.create_thinningsampler(args.NumSample, args.NumExp)
        os.makedirs(self.args.SavePath, exist_ok=True)
        self.log = LogWriter(os.path.join(self.args.PathModelDir, "log.txt"), vars(self.args) )
        self.log.initBest()

    def generate_seqs(self, model: XFMRNHPFast, num_seqs, min_len, max_len, max_time=10):
        self.thinning_sampler.cuda()
        results = []
        seq_id = 0
        thinning_sampler = self.thinning_sampler
        bos_sets = list(range(self.EventNum))
        for i in tqdm(range(num_seqs), desc=f"   (Generate)    ", leave=False, mininterval=2):
            # thinning can only run in single instance mode, not in batch mode
            length = random.randint(min_len, max_len)
            types = [random.sample(bos_sets, 1)[0], ]
            _start_time = 0
            times = [_start_time, ]
            dtimes = [_start_time, ]
            intens = [[0] * self.args.EventNum]
            for j in range(1, length - 1):
                _time_seq, _event_seq = torch.FloatTensor(times), torch.LongTensor(types)
                time_last_event = _time_seq[-1].item()
                boundary = time_last_event + max_time
                _event_prefix, _time_prefix = _event_seq.unsqueeze(0).cuda(), _time_seq.unsqueeze(0).cuda()
                accepted_times, weights = thinning_sampler.draw_next_time(
                    [[_event_prefix, _time_prefix],
                     time_last_event, boundary, model, self.args.Patience],
                    mode="unbiased"
                )
                _time_uncond = torch.sum(accepted_times * weights)
                time_uncond = float(_time_uncond)
                dtime_uncond = time_uncond - time_last_event
                times.append(time_uncond)
                dtimes.append(dtime_uncond)
                intensities_at_times = model.compute_intensities_at_sampled_times(
                    _event_prefix, _time_prefix,
                    _time_uncond.reshape(1, 1).cuda()
                )[0, 0]
                intens.append(intensities_at_times.tolist())
                next_event_name = torch.multinomial(intensities_at_times, 1)[0].item()
                types.append(next_event_name)
            rst = [
                {
                    "time_since_start": times[x],
                    "type_event": types[x],
                    "time_since_last_event": dtimes[x],
                    "intensities": intens[x]
                } for x in range(len(times))
            ]
            results.append(rst)
            seq_id += 1
        return results

    def run(self):
        splits = ["train", "test", "dev"]
        num_seqs = self.args.NumSeqs
        train_num = int(num_seqs)
        dev_num = 1000
        test_num = 1000
        torch.save(self.model.state_dict(), self.args.PathSave)
        total_tokens = 0
        for _split, _num_seqs in zip(splits, [train_num, dev_num, test_num]):
            save_path = os.path.join(self.args.SavePath, f"{_split}.pkl")
            with open(save_path, "wb") as f_out:
                data = self.generate_seqs(self.model, _num_seqs, self.args.MinLength, self.args.MaxLength,
                                          self.args.LookAheadTime)
                pickle.dump(
                    {
                        "dim_process": self.EventNum,
                        _split: data
                    }, f_out
                )
                total_tokens += sum([len(x) - 1 for x in data])
        print("impatient rate: {}".format(self.thinning_sampler.patience_counter / total_tokens))


